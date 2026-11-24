// The photo mode's capture on the Vulkan back end (photo.c).
//
// OP_PHOTO: the game's output, as the replay has it where the capture was recorded (after the frame's scaling pass,
// before the HUD and the photo's own passes), is copied into a host-visible buffer, and the middle of the picture as
// shown (photo_middle) is blitted, linear-filtered, into the album image (1280x720 RGBA8, made at the first photo
// and kept). Once the GPU is done, the completion thread hands the buffer to photo.c, whose worker releases it.
// OP_PHOTO_FILL: the album image is copied into the texture the game's photo passes drew into (blitted when that
// texture has another size or format), so "save this picture?" shows it and the game's read-back gets it. The
// copy waits until the passes are recorded because they overwrite their targets, the output too under an upscaler's
// smaller scene; the album image holds the picture meanwhile. Per photo: one copy of 8 MiB (1920x1080) to 32 MiB
// (3840x2160) and two small blits, nothing on other frames.
//
// Each follows the replay's own transfers (vk_exec_xfer.c): a full barrier (it ends the open pass), the transfers on
// images in GENERAL, then a barrier from the transfers to every later command and to host reads, so the passes that
// write those images next wait for them whatever barrier the pass chain picks.
#include <stdlib.h>

#include "host.h"
#include "nvn_api.h"
#include "photo.h"
#include "vk.h"
#include "vk_exec.h"

typedef struct {
    VkBuffer buf;
    VmaAllocation alloc;
    const uint8_t *map;
    uint32_t seq, w, h;
    float shape;
    int bgra;
} Copy;

// The album image (replay thread only): made at the first photo, kept for the run.
static VkImage album;
static VmaAllocation album_mem;
static uint32_t album_seq;   // the photo the album image holds, 0 for none
static int album_failed;

static void release(void *arg) {
    Copy *j = arg;
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static void done(void *arg) {
    Copy *j = arg;
    vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);
    photo_captured(j->seq, j->map, j->w, j->h, (size_t)j->w * 4, j->bgra, j->shape, release, j);
}

static void after_transfers(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                                            VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
}

// The album image, made and taken to GENERAL in this command buffer the first time; 0 when it cannot be had.
static int album_get(VkExec *ex) {
    if (album) return 1;
    if (album_failed) return 0;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {PHOTO_ALBUM_W, PHOTO_ALBUM_H, 1},
                            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ci, &ai, &album, &album_mem, NULL) != VK_SUCCESS) {
        album = VK_NULL_HANDLE, album_failed = 1;
        rt_log("photo: no album image: the game's photo textures keep their own pictures");
        return 0;
    }
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = album, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    return 1;
}

void vk_photo_copy(VkExec *ex, const p_photo *p) {
    const VkTexImage *img = vk_texture_get(p->tex);
    if (!img || img->samples > 1 || !(img->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
        photo_capture_failed(p->seq, "the output has no single-sampled colour image");
        return;
    }
    int bgra = img->format == VK_FORMAT_B8G8R8A8_UNORM || img->format == VK_FORMAT_B8G8R8A8_SRGB;
    if (!bgra && img->format != VK_FORMAT_R8G8B8A8_UNORM && img->format != VK_FORMAT_R8G8B8A8_SRGB) {
        photo_capture_failed(p->seq, "the output's image is not 8-bit RGBA or BGRA");
        return;
    }
    Copy *j = calloc(1, sizeof *j);
    if (!j) {
        photo_capture_failed(p->seq, "out of memory");
        return;
    }
    *j = (Copy){.seq = p->seq, .w = img->width, .h = img->height, .shape = p->shape, .bgra = bgra};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)j->w * j->h * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) {
        free(j);
        photo_capture_failed(p->seq, "no host-visible buffer for the copy");
        return;
    }
    j->map = info.pMappedData;
    vk_exec_barrier(ex, VK_BAR_COPY);
    VkBufferImageCopy r = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {j->w, j->h, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, j->buf, 1, &r);
    ex->recorded++;
    if (album_get(ex)) {
        double m[4];
        photo_middle(j->w, j->h, p->shape, (double)PHOTO_ALBUM_W / PHOTO_ALBUM_H, m);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}};
        bl.srcOffsets[0] = (VkOffset3D){(int32_t)(m[0] + 0.5), (int32_t)(m[1] + 0.5), 0};
        bl.srcOffsets[1] = (VkOffset3D){(int32_t)(m[2] + 0.5), (int32_t)(m[3] + 0.5), 1};
        bl.dstOffsets[1] = (VkOffset3D){PHOTO_ALBUM_W, PHOTO_ALBUM_H, 1};
        vkCmdBlitImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, album, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       VK_FILTER_LINEAR);
        ex->recorded++;
        album_seq = p->seq;
    }
    after_transfers(ex);
    ex->recorded++;
    vk_completion_add(ex->fence, done, j);
}

void vk_photo_fill(VkExec *ex, const p_photo_fill *p) {
    if (!album || album_seq != p->seq) return;   // the capture failed: the game's own picture stays
    const VkTexImage *d = vk_texture_get(p->dst);
    if (!d || d->samples > 1 || !(d->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
        rt_log("photo: photo %u: the album texture has no single-sampled colour image; its own picture shows", p->seq);
        return;
    }
    vk_exec_barrier(ex, VK_BAR_COPY);
    if (d->width == PHOTO_ALBUM_W && d->height == PHOTO_ALBUM_H && d->format == VK_FORMAT_R8G8B8A8_UNORM) {
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .extent = {PHOTO_ALBUM_W, PHOTO_ALBUM_H, 1}};
        vkCmdCopyImage(ex->cb, album, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    } else {   // another size, or BGRA / sRGB: the blit converts
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}};
        bl.srcOffsets[1] = (VkOffset3D){PHOTO_ALBUM_W, PHOTO_ALBUM_H, 1};
        bl.dstOffsets[1] = (VkOffset3D){(int32_t)d->width, (int32_t)d->height, 1};
        vkCmdBlitImage(ex->cb, album, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       VK_FILTER_LINEAR);
    }
    after_transfers(ex);
    ex->recorded += 2;
    vk_texture_gpu_written(p->dst);
}
