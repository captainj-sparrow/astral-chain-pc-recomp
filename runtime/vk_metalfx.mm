// Apple MetalFX Temporal Scaler over MoltenVK (VK_EXT_metal_objects).
// Compiled only on macOS (Apple Silicon, macOS 14.4+) with Objective-C++ ARC (-fobjc-arc).
#ifdef __APPLE__

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "vk_mem_alloc.h"
#pragma clang diagnostic pop

#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "host.h"
#include "vk.h"
#include "vk_metalfx.h"
#include "platform/platform.h"
}

enum { MFX_CTX_MAX = 4, MFX_DEAD_MAX = 16 };

struct MfxSlot {
    int active;
    uint32_t w, h, ow, oh;
    int hdr;
    int staged_init;               // 1 once color_img and depth_img have been transitioned out of UNDEFINED
    uint64_t used;
    id<MTLFXTemporalScaler> scaler;
    VkImage color_img;             // R16G16B16A16_SFLOAT (w x h), exportable to Metal
    VmaAllocation color_mem;
    id<MTLTexture> color_tex;
    VkImage depth_img;             // R32_SFLOAT (w x h), exportable to Metal
    VmaAllocation depth_mem;
    id<MTLTexture> depth_tex;
    VkBuffer depth_buf;            // w * h * 4 bytes, staging for D32_SFLOAT(_S8_UINT) depth aspect -> R32_SFLOAT
    VmaAllocation depth_buf_mem;
};

struct MfxDead {
    int active;
    uint64_t after;
    id<MTLFXTemporalScaler> scaler;
    id<MTLTexture> color_tex;
    id<MTLTexture> depth_tex;
    VkImage color_img;
    VmaAllocation color_mem;
    VkImage depth_img;
    VmaAllocation depth_mem;
    VkBuffer depth_buf;
    VmaAllocation depth_buf_mem;
};

static struct {
    int inited, available, failed;
    PFN_vkExportMetalObjectsEXT export_fn;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    VkSemaphore sem;               // timeline semaphore backed by MTLSharedEvent
    id<MTLSharedEvent> event;
    uint64_t sem_val;
    uint64_t clock;
    MfxSlot slots[MFX_CTX_MAX];
    MfxDead dead[MFX_DEAD_MAX];
} M;

static id<MTLTexture> export_texture(VkImage img) {
    if (!img || !M.export_fn) return nil;
    VkExportMetalTextureInfoEXT ti = {};
    ti.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT;
    ti.image = img;
    ti.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT oi = {};
    oi.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    oi.pNext = &ti;
    M.export_fn(vk_ctx()->device, &oi);
    return ti.mtlTexture;
}

static int create_export_image(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                               VkImage *out_img, VmaAllocation *out_mem, id<MTLTexture> *out_tex) {
    const VkCtx *c = vk_ctx();
    VkExportMetalObjectCreateInfoEXT em = {};
    em.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT;
    em.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT;

    VkImageCreateInfo ii = {};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.pNext = &em;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = fmt;
    ii.extent = {w, h, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai = {};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(c->vma, &ii, &ai, out_img, out_mem, NULL) != VK_SUCCESS) {
        *out_img = VK_NULL_HANDLE;
        *out_mem = VK_NULL_HANDLE;
        return 0;
    }
    id<MTLTexture> tex = export_texture(*out_img);
    if (!tex) {
        vmaDestroyImage(c->vma, *out_img, *out_mem);
        *out_img = VK_NULL_HANDLE;
        *out_mem = VK_NULL_HANDLE;
        return 0;
    }
    *out_tex = tex;
    return 1;
}

static void destroy_dead_entry(MfxDead *d) {
    const VkCtx *c = vk_ctx();
    d->scaler = nil;
    d->color_tex = nil;
    d->depth_tex = nil;
    if (d->color_img) vmaDestroyImage(c->vma, d->color_img, d->color_mem);
    if (d->depth_img) vmaDestroyImage(c->vma, d->depth_img, d->depth_mem);
    if (d->depth_buf) vmaDestroyBuffer(c->vma, d->depth_buf, d->depth_buf_mem);
    d->color_img = VK_NULL_HANDLE;
    d->color_mem = VK_NULL_HANDLE;
    d->depth_img = VK_NULL_HANDLE;
    d->depth_mem = VK_NULL_HANDLE;
    d->depth_buf = VK_NULL_HANDLE;
    d->depth_buf_mem = VK_NULL_HANDLE;
    d->after = 0;
    d->active = 0;
}

static void sweep(int all) {
    uint64_t done = vk_sync_fence_done();
    for (int i = 0; i < MFX_DEAD_MAX; i++) {
        if (M.dead[i].active && (all || M.dead[i].after <= done)) destroy_dead_entry(&M.dead[i]);
    }
}

static void retire_slot(MfxSlot *s) {
    if (!s->active) return;
    int idx = -1;
    for (int i = 0; i < MFX_DEAD_MAX; i++) {
        if (!M.dead[i].active) { idx = i; break; }
    }
    if (idx < 0) {
        vk_queue_lock();
        vkDeviceWaitIdle(vk_ctx()->device);
        vk_queue_unlock();
        sweep(1);
        idx = 0;
    }
    MfxDead *d = &M.dead[idx];
    d->active = 1;
    d->after = vk_sync_issued() + 1;
    d->scaler = s->scaler;
    d->color_tex = s->color_tex;
    d->depth_tex = s->depth_tex;
    d->color_img = s->color_img;
    d->color_mem = s->color_mem;
    d->depth_img = s->depth_img;
    d->depth_mem = s->depth_mem;
    d->depth_buf = s->depth_buf;
    d->depth_buf_mem = s->depth_buf_mem;

    s->scaler = nil;
    s->color_tex = nil;
    s->depth_tex = nil;
    s->color_img = VK_NULL_HANDLE;
    s->color_mem = VK_NULL_HANDLE;
    s->depth_img = VK_NULL_HANDLE;
    s->depth_mem = VK_NULL_HANDLE;
    s->depth_buf = VK_NULL_HANDLE;
    s->depth_buf_mem = VK_NULL_HANDLE;
    s->active = 0;
    s->staged_init = 0;
    s->w = s->h = s->ow = s->oh = 0;
    s->hdr = 0;
    s->used = 0;
}

static MfxSlot *slot_for(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int hdr, int sync_init) {
    int free_idx = -1, lru = 0;
    M.clock++;
    for (int i = 0; i < MFX_CTX_MAX; i++) {
        if (M.slots[i].active && M.slots[i].w == w && M.slots[i].h == h &&
            M.slots[i].ow == ow && M.slots[i].oh == oh && M.slots[i].hdr == hdr) {
            M.slots[i].used = M.clock;
            return &M.slots[i];
        }
        if (!M.slots[i].active && free_idx < 0) free_idx = i;
        if (M.slots[i].used < M.slots[lru].used) lru = i;
    }
    if (free_idx < 0) {
        rt_log("metalfx: retiring %ux%u to %ux%u scaler for another size",
               M.slots[lru].w, M.slots[lru].h, M.slots[lru].ow, M.slots[lru].oh);
        retire_slot(&M.slots[lru]);
        free_idx = lru;
    }
    MfxSlot *s = &M.slots[free_idx];
    uint64_t t0 = plat_now_ns();
    id<MTLFXTemporalScaler> scaler = nil;
    @autoreleasepool {
        MTLFXTemporalScalerDescriptor *desc = [[MTLFXTemporalScalerDescriptor alloc] init];
        desc.inputWidth = w;
        desc.inputHeight = h;
        desc.outputWidth = ow;
        desc.outputHeight = oh;
        desc.colorTextureFormat = MTLPixelFormatRGBA16Float;
        desc.depthTextureFormat = MTLPixelFormatR32Float;
        desc.motionTextureFormat = MTLPixelFormatRG16Float;
        desc.outputTextureFormat = MTLPixelFormatRGBA16Float;
        desc.autoExposureEnabled = hdr ? YES : NO;
        desc.inputContentPropertiesEnabled = NO;
        desc.requiresSynchronousInitialization = sync_init ? YES : NO;
        scaler = [desc newTemporalScalerWithDevice:M.device];
    }
    if (!scaler) {
        rt_log("metalfx: newTemporalScalerWithDevice failed for %ux%u to %ux%u (%s)",
               w, h, ow, oh, hdr ? "linear" : "LDR");
        return NULL;
    }
    // Staging inputs: color (RGBA16F) and depth (R32F), plus a device-local buffer for depth-aspect extraction.
    const VkImageUsageFlags img_u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (!create_export_image(w, h, VK_FORMAT_R16G16B16A16_SFLOAT, img_u,
                             &s->color_img, &s->color_mem, &s->color_tex) ||
        !create_export_image(w, h, VK_FORMAT_R32_SFLOAT, img_u,
                             &s->depth_img, &s->depth_mem, &s->depth_tex)) {
        rt_log("metalfx: staging image creation failed (%ux%u)", w, h);
        s->scaler = scaler;
        s->active = 1;
        retire_slot(s);
        return NULL;
    }
    VkBufferCreateInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = (VkDeviceSize)w * h * 4;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo bai = {};
    bai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &bai, &s->depth_buf, &s->depth_buf_mem, NULL) != VK_SUCCESS) {
        rt_log("metalfx: depth staging buffer creation failed (%ux%u)", w, h);
        s->scaler = scaler;
        s->active = 1;
        retire_slot(s);
        return NULL;
    }
    s->scaler = scaler;
    s->w = w;
    s->h = h;
    s->ow = ow;
    s->oh = oh;
    s->hdr = hdr;
    s->staged_init = 0;
    s->used = M.clock;
    s->active = 1;
    rt_log("metalfx: temporal scaler %ux%u to %ux%u (%s colour, sync %d) created in %.1f ms",
           w, h, ow, oh, hdr ? "linear" : "LDR", sync_init, (double)(plat_now_ns() - t0) / 1e6);
    return s;
}

extern "C" void vk_metalfx_init(void) {
    if (M.inited) return;
    M.inited = 1;
    const VkCtx *c = vk_ctx();
    if (!c || !c->device || !c->has_metal_objects) {
        rt_log("metalfx: unavailable (VK_EXT_metal_objects not enabled)");
        return;
    }
    M.export_fn = (PFN_vkExportMetalObjectsEXT)vkGetDeviceProcAddr(c->device, "vkExportMetalObjectsEXT");
    if (!M.export_fn) {
        rt_log("metalfx: unavailable (vkExportMetalObjectsEXT missing)");
        return;
    }
    @autoreleasepool {
        VkExportMetalDeviceInfoEXT di = {};
        di.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT;
        VkExportMetalCommandQueueInfoEXT qi = {};
        qi.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT;
        qi.pNext = &di;
        qi.queue = c->gfx_queue;
        VkExportMetalObjectsInfoEXT oi = {};
        oi.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
        oi.pNext = &qi;
        M.export_fn(c->device, &oi);
        M.device = di.mtlDevice;
        M.queue = qi.mtlCommandQueue;
        if (!M.device || !M.queue) {
            rt_log("metalfx: unavailable (could not export MTLDevice or MTLCommandQueue)");
            M.device = nil;
            M.queue = nil;
            return;
        }
        if (![MTLFXTemporalScalerDescriptor supportsDevice:M.device]) {
            rt_log("metalfx: unavailable (MTLFXTemporalScaler not supported on %s)",
                   M.device.name ? M.device.name.UTF8String : "device");
            M.device = nil;
            M.queue = nil;
            return;
        }
        // Dedicated timeline semaphore backed by an exported MTLSharedEvent for GPU-side ordering.
        VkExportMetalObjectCreateInfoEXT em = {};
        em.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT;
        em.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT;
        VkSemaphoreTypeCreateInfo st = {};
        st.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        st.pNext = &em;
        st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        st.initialValue = 0;
        VkSemaphoreCreateInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        si.pNext = &st;
        if (vkCreateSemaphore(c->device, &si, NULL, &M.sem) != VK_SUCCESS) {
            rt_log("metalfx: unavailable (timeline semaphore creation failed)");
            M.device = nil;
            M.queue = nil;
            return;
        }
        VkExportMetalSharedEventInfoEXT ei = {};
        ei.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT;
        ei.semaphore = M.sem;
        VkExportMetalObjectsInfoEXT eoi = {};
        eoi.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
        eoi.pNext = &ei;
        M.export_fn(c->device, &eoi);
        M.event = ei.mtlSharedEvent;
        if (!M.event) {
            rt_log("metalfx: unavailable (could not export MTLSharedEvent from timeline semaphore)");
            vkDestroySemaphore(c->device, M.sem, NULL);
            M.sem = VK_NULL_HANDLE;
            M.device = nil;
            M.queue = nil;
            return;
        }
        M.available = 1;
        float min_s = [MTLFXTemporalScalerDescriptor supportedInputContentMinScaleForDevice:M.device];
        float max_s = [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:M.device];
        rt_log("metalfx: available on %s (scale %.2fx..%.2fx)",
               M.device.name ? M.device.name.UTF8String : "Metal GPU", min_s, max_s);
    }
}

extern "C" int vk_metalfx_available(void) { return M.available && !M.failed; }

extern "C" void vk_metalfx_warm(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int hdr) {
    if (!vk_metalfx_available() || !w || !h) return;
    if (!ow) ow = w;
    if (!oh) oh = h;
    if (!slot_for(w, h, ow, oh, hdr, 1))
        rt_log("metalfx: pre-warm failed for %ux%u to %ux%u (will retry on first frame)", w, h, ow, oh);
}

extern "C" int vk_metalfx_evaluate(VkCommandBuffer *cb_io, const MetalFxEval *e) {
    if (!vk_metalfx_available() || !cb_io || !*cb_io || !e) return 0;
    uint32_t w = e->w, h = e->h, ow = e->ow ? e->ow : w, oh = e->oh ? e->oh : h;
    MfxSlot *s = slot_for(w, h, ow, oh, e->hdr, 0);
    if (!s) {
        M.failed = 1;
        return 0;
    }
    id<MTLTexture> mv_tex = export_texture(e->mv);
    id<MTLTexture> out_tex = export_texture(e->out);
    if (!mv_tex || !out_tex) {
        rt_log("metalfx: could not export mv (%p) or out (%p) MTLTexture: MetalFX off for this run",
               (__bridge void *)mv_tex, (__bridge void *)out_tex);
        M.failed = 1;
        return 0;
    }

    // Record staging copies directly into *cb_io (vk_dlaa.c already drained the recorder thread).
    // Parenthesizing vkCmd* bypasses vk_rec.h's macros so all commands go directly into cb in exact order.
    VkCommandBuffer cb = *cb_io;
    if (!s->staged_init) {
        VkImageMemoryBarrier2 ib[2] = {};
        VkImage imgs[2] = {s->color_img, s->depth_img};
        for (int i = 0; i < 2; i++) {
            ib[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            ib[i].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            ib[i].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            ib[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            ib[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ib[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib[i].image = imgs[i];
            ib[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 2;
        di.pImageMemoryBarriers = ib;
        (vkCmdPipelineBarrier2)(cb, &di);
        s->staged_init = 1;
    }

    // 1. Copy/convert scene color (any format, level, layer) -> RGBA16F staging image.
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, e->color.level, e->color.layer, 1};
    blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    (vkCmdBlitImage)(cb, e->color.image, VK_IMAGE_LAYOUT_GENERAL,
                     s->color_img, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);

    // 2. Extract depth aspect (works for both D32_SFLOAT_S8_UINT and D32_SFLOAT) -> depth_buf -> R32F staging image.
    VkBufferImageCopy d2b = {};
    d2b.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, e->depth.level, e->depth.layer, 1};
    d2b.imageExtent = {w, h, 1};
    (vkCmdCopyImageToBuffer)(cb, e->depth.image, VK_IMAGE_LAYOUT_GENERAL, s->depth_buf, 1, &d2b);

    VkMemoryBarrier2 tb = {};
    tb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    tb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    tb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    tb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    tb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo tdi = {};
    tdi.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    tdi.memoryBarrierCount = 1;
    tdi.pMemoryBarriers = &tb;
    (vkCmdPipelineBarrier2)(cb, &tdi);

    VkBufferImageCopy b2d = {};
    b2d.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b2d.imageExtent = {w, h, 1};
    (vkCmdCopyBufferToImage)(cb, s->depth_buf, s->depth_img, VK_IMAGE_LAYOUT_GENERAL, 1, &b2d);

    // 3. Submit pre-MetalFX commands signalling M.sem = sig_val, and arm the next Vulkan submit to wait on wait_val.
    uint64_t sig_val = ++M.sem_val;
    uint64_t wait_val = ++M.sem_val;
    *cb_io = vk_sync_split_ex(M.sem, sig_val, M.sem, wait_val);

    // 4. Encode MetalFX onto MoltenVK's MTLCommandQueue between sig_val and wait_val.
    @autoreleasepool {
        vk_queue_lock();
        id<MTLCommandBuffer> mcb = [M.queue commandBuffer];
        if (!mcb) {
            vk_queue_unlock();
            rt_log("metalfx: commandBuffer allocation failed: MetalFX off for this run");
            // Still signal wait_val from CPU so the next Vulkan submit does not deadlock on M.sem.
            M.event.signaledValue = wait_val;
            M.failed = 1;
            return 0;
        }
        [mcb encodeWaitForEvent:M.event value:sig_val];

        id<MTLFXTemporalScaler> sc = s->scaler;
        sc.colorTexture = s->color_tex;
        sc.depthTexture = s->depth_tex;
        sc.motionTexture = mv_tex;
        sc.outputTexture = out_tex;
        sc.inputContentWidth = w;
        sc.inputContentHeight = h;
        sc.jitterOffsetX = e->jx;
        sc.jitterOffsetY = e->jy;
        sc.motionVectorScaleX = 1.0f;
        sc.motionVectorScaleY = 1.0f;
        sc.depthReversed = NO;
        sc.preExposure = 1.0f;
        sc.reset = e->reset ? YES : NO;

        [sc encodeToCommandBuffer:mcb];
        [mcb encodeSignalEvent:M.event value:wait_val];
        [mcb commit];
        vk_queue_unlock();
    }
    return 1;
}

extern "C" void vk_metalfx_sweep(void) {
    if (M.available) sweep(0);
}

extern "C" void vk_metalfx_trim(void) {
    int n = 0;
    for (int i = 0; i < MFX_CTX_MAX; i++) {
        if (M.slots[i].active) retire_slot(&M.slots[i]), n++;
    }
    if (n) rt_log("metalfx: %d scaler%s released", n, n == 1 ? "" : "s");
}

extern "C" void vk_metalfx_release(void) {
    for (int i = 0; i < MFX_CTX_MAX; i++) {
        if (M.slots[i].active) retire_slot(&M.slots[i]);
    }
    sweep(1);
    const VkCtx *c = vk_ctx();
    if (M.sem && c && c->device) vkDestroySemaphore(c->device, M.sem, NULL);
    M.sem = VK_NULL_HANDLE;
    M.event = nil;
    M.queue = nil;
    M.device = nil;
    M.export_fn = NULL;
    M.available = M.inited = M.failed = 0;
    M.sem_val = 0;
}

#endif // __APPLE__
