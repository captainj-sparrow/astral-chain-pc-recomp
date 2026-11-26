// NVIDIA DLSS Frame Generation (AC_FRAME_GENERATION = off, auto, 2x, 3x, 4x; astralchain.ini frame_generation, live
// once NGX was loaded at start): NGX's frame generation feature (NVSDK_NGX_Feature_FrameGeneration) makes one to
// three frames between two of the game's, and the present thread shows them between the game's presents.
//
// The inputs, per game frame (vk_backend.c present_image, on the replay thread, the frame's images still holding it):
// - the picture the game presents, with its HUD: its window texture (RGBA8);
// - the same picture without the HUD: the window texture copied when the game's last scene pass (SMAA's blend, which
//   vk_hdr.c finds: vk_hdr_draw) has drawn it and before the first HUD draw (vk_hdr_before_draw, vk_dlssg_capture);
//   NGX compares the two to tell the HUD from the scene, so the HUD is not moved along with the scene. The float scene
//   that pass reads is from before its blend: NGX took every edge the blend changed for HUD and held it still;
// - the scene depth and the motion vectors of the temporal upscaler (vk_dlaa.c vk_dlaa_fg_inputs: render pixels, the
//   camera's and the objects' own), the camera's matrices and the jitter. So frame generation needs a temporal
//   upscaler running (DLSS, FSR 3, FSR 4, in any mode): without one the frame has no motion vectors and is shown alone.
// NGX keeps the previous frame itself and makes the frames between it and this one, each in its own evaluation
// (MultiFrameIndex 1..MultiFrameCount), recorded by the present thread at the start of the command buffer of the blit
// that shows the first of them (gfx_present.c vk_present_pre_record): that submission comes before the next frame's
// work (vk_present_drain), so the depth, the motion vectors and the scene still hold this frame there. The same
// command buffer copies the window texture, so the game frame can be shown last, after the game drew into the window
// texture again.
//
// Pacing (vk_backend.c present_main): the game frame due at its display time T is shown n/(n+1) of a frame period
// later, and the n generated frames before it at even steps from T on; a frame without generated frames (no motion
// vectors, a new history) is shown at the same delayed time, so the game's frames keep even intervals when frame
// generation starts and stops. The delay is what frame generation costs in latency: 8.3 ms at 60 fps with 2x, 12.5 ms
// with 4x. Under FIFO a fixed multiplier is capped at the pictures the display shows in a game frame (6x at 60 fps on
// 240 Hz: 4x; at least 2x): FIFO shows one per refresh, and more would queue up behind it.
//
// NGX also gets each game frame's number (DLSSG.BackbufferFrameID; a gap is a new history here too) and a buffer it
// marks when it sees a scene change between the two frames (DLSSG.OutputDisableInterpolation): the generated frames of
// such a set are replaced by the game's frame on the GPU (shaders/dlssg_hold.comp), and the log counts them. A set NGX
// refuses shows the game's frame alone and its feature is remade; five refused sets in a row, or three failed
// creations, turn frame generation off for the run.
//
// AC_DLSSG_MVDEBUG = 1 .. 8 (test, no settings row): frame generation's inputs drawn over every frame it shows, after
// NGX evaluated (so its inputs stay as they were): 1 the motion vectors as a colour tint over the game's frame, 2 the
// vectors alone, 3 the depth, 4 where the picture differs from the HUD-less one, what NGX keeps still as HUD, 5 and 6
// the alpha channel of the picture and of the HUD-less one, 7 HUD that UI recomposition misses, 8 HUD alpha over the bare
// scene (shaders/dlssg_mvdebug.comp has the key). The generated
// frames show the game's frame too, so the picture holds still and the vectors sit on the surfaces they belong to.
// AC_DLSSG_MVDEBUG_SCALE: the motion in render pixels that saturates the colour (default 8). Read at each present.
//
// With HDR output (an HDR10 swapchain): the present shows the window texture's PQ encode (vk_hdr.c vk_hdr_encode: the
// scene's pixels from the float scene, highlights and all, the HUD's at the HUD's white). NGX gets that encode instead
// of the 8-bit pictures: the window texture and the HUD-less picture are copied into images of ours and each encoded
// with the frame's scene (vk_hdr_encode_fg) into an A2B10G10R10 image of PQ code values, what the present would show of
// it. The HUD-less one's pixels are all scene, so the two differ only under the HUD, as in SDR. NGX is told the
// pictures are HDR (DLSSG.ColorBuffersHDR) and writes its frames in the same format, and the present sends them and the
// game's frame to the swapchain without encoding them again (vk_dlssg_encoded). UI recomposition's alpha comes from the
// 8-bit copy (the HDR10 alpha has 2 bits).
#include "vk_dlssg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "dlssg_alpha0_spv.h"
#include "dlssg_hold_spv.h"
#include "dlssg_mvdebug_spv.h"
#include "dlssg_uialpha_spv.h"
#include "gfx.h"
#include "host.h"
#include "vk_dlaa.h"
#include "platform/platform.h"

#define MAX_GEN 5   // generated frames per game frame: NGX's MultiFrameCountMax on the RTX 50 series
#define HDR_FMT VK_FORMAT_A2B10G10R10_UNORM_PACK32   // the pictures with an HDR10 swapchain: PQ code values
#define HINTS 8     // NGX's "do not show" buffers, a ring: one is read back HINTS - 1 sets after it was written, when
                    // its blit (8 in flight at most, gfx_present.c) has long finished
#define EVAL_FAILS_OFF 5     // sets in a row NGX refused: off for the run (each refused set remakes the feature)
#define CREATE_FAILS_OFF 3   // failed feature creations in a row: off for the run

typedef struct { VkImage img; VmaAllocation mem; VkImageView view; } Img;

static struct {
    // replay thread
    int last_on;              // the previous frame had what frame generation needs
    // present thread
    void *feat;               // NGX's feature, for:
    uint32_t w, h, rw, rh;    // the picture's size and the scene's
    VkFormat fmt;             // the pictures NGX reads and writes: the window texture's, HDR_FMT with HDR output
    VkFormat wfmt;            // the window texture's
    int hdr;                  // this job's swapchain is HDR10: the pictures are encoded before NGX (vk_hdr_encode_fg)
    VkFormat img_fmt, img_wfmt;   // the formats the images below were made in
    Img gen[MAX_GEN], hud, real;
    Img sdr, hud8;            // HDR output: the window texture and the HUD-less picture in its format (encode input)
    Img dbg;                  // AC_DLSSG_MVDEBUG: a copy of the game's frame the overlay reads
    Img uia;                  // AC_DLSSG_UI_ALPHA: the HUD's coverage from the picture's alpha (R8)
    int uir;                  // the feature was made with UI recomposition
    int fmt_ok;               // -1 not checked, else for ok_wfmt and ok_hdr
    VkFormat ok_wfmt;
    int ok_hdr;
    int armed, made, n;       // this job: the hook is set, generated frames made, asked for
    int evaluated;            // this job: NGX evaluated the frame (its history now ends with it)
    int prev_shown;           // the previous job showed frame generation's frames: its history follows on
    uint64_t last_id;         // the game frame NGX last evaluated (VkFgFrame.frame_id): a gap starts a new history
    int eval_fails, create_fails;   // in a row (EVAL_FAILS_OFF, CREATE_FAILS_OFF)
    int drop_feat;            // a set was refused: the feature is remade at the next job, once its blit is done
    int capped;               // the multiplier last capped to under FIFO (logged on a change), 0 none
    struct { VkBuffer buf; VmaAllocation mem; void *map; int pending; } hint[HINTS];
    uint64_t sets, held;      // sets evaluated, and of those read back, the ones NGX said not to show (logged)
    VkFgFrame fr;             // this job's inputs
    VkImage win;              // and its window texture, of pw x ph
    uint32_t pw, ph;
    // any thread
    volatile int32_t failed;  // off for the run (image creation, or NGX failing again and again)
    volatile int32_t shown;   // the multiplier last shown, 0 none
    uint64_t shown_ns;        // when (plat_now_ns)
} G = {.fmt_ok = -1};

// The setting: 0 off, 1 auto, else the multiplier (2..4).
static int setting(void) {
    const char *v = cfg_value("AC_FRAME_GENERATION");
    if (!v || !*v || !strcmp(v, "off")) return 0;
    if (!strcmp(v, "auto")) return 1;
    int m = v[0] - '0';
    return m >= 2 && m <= 6 && v[1] == 'x' && !v[2] ? m : 0;
}

int vk_dlssg_max_multiplier(void) {
    unsigned max = 0;
    if (!vk_ngx_fg_caps(&max) || !max) return 0;
    return max >= MAX_GEN ? MAX_GEN + 1 : (int)max + 1;
}

void vk_dlssg_init(void) {
    int s = setting();
    if (!s) return;
    int m = vk_dlssg_max_multiplier();
#ifdef _WIN32
    const char *model = "nvngx_dlssg.dll next to astralchain.exe";
#else
    const char *model = "libnvidia-ngx-dlssg.so.<version> next to astralchain";
#endif
    if (!m) rt_log("dlssg: frame generation is set but NGX does not offer it here (an NVIDIA RTX 40 or newer GPU, a "
                   "recent driver and %s): off", model);
    else rt_log("dlssg: frame generation %s, up to %dx here; it needs a temporal upscaler running (upscaling)",
                s == 1 ? "auto" : cfg_value("AC_FRAME_GENERATION"), m);
}

int vk_dlssg_set_at_start(void) {
    static int on = -1;
    if (on < 0) on = setting() != 0;
    return on;
}

int vk_dlssg_scene_wanted(void) {
    unsigned max;
    return setting() && !plat_cas32(&G.failed, 0, 0) && vk_ngx_fg_caps(&max);
}

void vk_dlssg_collect(VkFgFrame *f) {
    f->on = 0;
    if (!vk_dlssg_scene_wanted()) {
        G.last_on = 0;
        return;
    }
    if (!vk_lowlat_oob_ok()) {   // Reflex without frame generation's own present queue (set after the start)
        static int logged;
        if (!logged)
            logged = 1, rt_log("dlssg: NVIDIA Reflex runs without frame generation's present queue (vk_device.c): "
                               "frame generation waits for a restart");
        G.last_on = 0;
        return;
    }
    f->on = f->depth && f->mv && f->hudless && f->rw && f->rh && f->hw && f->hh;
    if (f->on && !G.last_on) f->reset = 1;   // after frames without it: the history NGX holds is from another moment
    G.last_on = f->on;
}

// ---------------------------------------------------------------- the present thread

static int has_features(VkFormat f, VkFormatFeatureFlags need) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &fp);
    return (fp.optimalTilingFeatures & need) == need;
}

// The window texture's format wf, and with HDR output the HDR10 pictures', usable for frame generation's images.
static int format_ok(VkFormat wf, int hdr) {
    if (wf != VK_FORMAT_R8G8B8A8_UNORM && wf != VK_FORMAT_B8G8R8A8_UNORM) return 0;
    if (G.fmt_ok < 0 || G.ok_wfmt != wf || G.ok_hdr != hdr) {
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                    VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        G.ok_wfmt = wf, G.ok_hdr = hdr;
        G.fmt_ok = has_features(wf, need);
        if (!G.fmt_ok) rt_log("dlssg: the window texture's format %d cannot be a storage image here: off", (int)wf);
        else if (hdr && !(G.fmt_ok = has_features(HDR_FMT, need | VK_FORMAT_FEATURE_BLIT_SRC_BIT)))
            rt_log("dlssg: HDR10 pictures (format %d) cannot be storage images here: off with HDR output",
                   (int)HDR_FMT);
    }
    return G.fmt_ok;
}

static void img_free(Img *i) {
    const VkCtx *c = vk_ctx();
    if (i->view) vkDestroyImageView(c->device, i->view, NULL);
    if (i->img) vmaDestroyImage(c->vma, i->img, i->mem);
    memset(i, 0, sizeof *i);
}

static int img_make(VkCommandBuffer cb, Img *i, uint32_t w, uint32_t h, VkFormat f) {
    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &i->img, &i->mem, NULL) != VK_SUCCESS) return i->img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = i->img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &i->view) != VK_SUCCESS) {
        img_free(i);
        return 0;
    }
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = i->img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    return 1;
}

// ---------------------------------------------------------------- the HUD-less picture (the replay thread)
// The window texture as the game's last scene pass left it, before the first HUD draw (vk_hdr.c vk_hdr_before_draw):
// the scene exactly as the picture shows it, anti-aliased, without the HUD. The float scene that pass read is from
// before its SMAA blend; NGX took every edge the blend changed for HUD and held it still (a fence smeared at any
// multiplier). The copy is read by the present of the same frame, which is queued before the next frame's work, as the
// float scene was.
static struct {
    VkImage img;
    VmaAllocation mem;
    uint32_t w, h;
    VkFormat fmt;
    // AC_DLSSG_UI_ALPHA: the copy's view, and the copy with alpha 0 that goes back into the window texture
    VkImageView view;
    Img zero;
} CAP;
static void barrier(VkCommandBuffer cb);
// Compute passes with push descriptors (the passes of ours, below)
typedef struct Pass {
    int tried, ok;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
} Pass;
static Pass alpha_pass;
static VkSampler nearest;
static int pass_init(Pass *ps, const char *name, const uint32_t *spv, size_t bytes, const VkDescriptorType *types,
                     uint32_t n);
static void pass_run(VkCommandBuffer cb, Pass *ps, const VkImageView *views, const VkSampler *samplers, uint32_t n,
                     uint32_t out, const void *pc, uint32_t pc_bytes, uint32_t w, uint32_t h);

// AC_DLSSG_UI_ALPHA=1 (test): the window texture's alpha set to 0 at the HUD-less copy, so after the HUD's draws it
// holds their coverage (where they blend their alpha in over the destination's), for frame generation's UI
// recomposition. The game's scene leaves 1 there, and "over" onto 1 stays 1 (AC_DLSSG_MVDEBUG=5 shows it).
int vk_dlssg_ui_alpha(void) {
    const char *v = cfg_get("AC_DLSSG_UI_ALPHA");
    return v && !strcmp(v, "1");
}

static void zero_alpha(VkCommandBuffer cb, VkImage win) {
    static const VkDescriptorType t[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    const VkCtx *c = vk_ctx();
    if (!pass_init(&alpha_pass, "UI alpha", dlssg_alpha0_spv, sizeof dlssg_alpha0_spv, t, 2)) return;
    if (!CAP.view) {
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = CAP.img,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = CAP.fmt,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &CAP.view) != VK_SUCCESS) CAP.view = VK_NULL_HANDLE;
    }
    if (!CAP.view || (!CAP.zero.img && !img_make(cb, &CAP.zero, CAP.w, CAP.h, CAP.fmt))) return;
    VkImageView v[2] = {CAP.view, CAP.zero.view};
    VkSampler sm[2] = {nearest, VK_NULL_HANDLE};
    int32_t pc[2] = {(int32_t)CAP.w, (int32_t)CAP.h};
    pass_run(cb, &alpha_pass, v, sm, 2, 1, pc, sizeof pc, CAP.w, CAP.h);
    barrier(cb);
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {CAP.w, CAP.h, 1}};
    vkCmdCopyImage(cb, CAP.zero.img, VK_IMAGE_LAYOUT_GENERAL, win, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
}

int vk_dlssg_capture(VkCommandBuffer cb, VkImage win, VkFormat fmt, uint32_t w, uint32_t h) {
    const VkCtx *c = vk_ctx();
    if (CAP.img && (CAP.w != w || CAP.h != h || CAP.fmt != fmt)) {   // a new game resolution: rare
        vk_queue_lock();
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
        if (CAP.view) vkDestroyImageView(c->device, CAP.view, NULL);
        img_free(&CAP.zero);
        vmaDestroyImage(c->vma, CAP.img, CAP.mem);
        CAP.img = VK_NULL_HANDLE, CAP.view = VK_NULL_HANDLE;
    }
    if (!CAP.img) {
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = fmt, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &CAP.img, &CAP.mem, NULL) != VK_SUCCESS) {
            CAP.img = VK_NULL_HANDLE;
            static int logged;
            if (!logged)
                logged = 1, rt_log("dlssg: the HUD-less picture's image could not be made: the float scene instead");
            return 0;
        }
        CAP.w = w, CAP.h = h, CAP.fmt = fmt;
        VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = CAP.img,
                                   .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(cb, &di);
        rt_log("dlssg: the HUD-less picture is the window texture before the HUD (%ux%u, format %d)", w, h, (int)fmt);
    }
    barrier(cb);   // the scene pass's writes, and the present before still reading last frame's copy
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {w, h, 1}};
    vkCmdCopyImage(cb, win, VK_IMAGE_LAYOUT_GENERAL, CAP.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    barrier(cb);   // before the HUD draws write the window texture again
    if (vk_dlssg_ui_alpha()) {
        zero_alpha(cb, win);
        barrier(cb);
    }
    return 1;
}

VkImage vk_dlssg_capture_image(void) { return CAP.img; }

static void capture_free(void) {
    if (CAP.view) vkDestroyImageView(vk_ctx()->device, CAP.view, NULL);
    img_free(&CAP.zero);
    if (CAP.img) vmaDestroyImage(vk_ctx()->vma, CAP.img, CAP.mem);
    memset(&CAP, 0, sizeof CAP);
}

// Everything goes: the GPU may still read the images and NGX's feature (a resize, the release), so the device is
// waited for first. Rare: a change of the game resolution or the scene's size.
static void free_all(int wait) {
    if (wait && (G.feat || G.real.img)) {
        vk_queue_lock();   // vkDeviceWaitIdle needs every queue externally synchronised
        vkDeviceWaitIdle(vk_ctx()->device);
        vk_queue_unlock();
    }
    if (G.feat) vk_ngx_fg_release(G.feat);
    G.feat = NULL;
    for (int i = 0; i < MAX_GEN; i++) img_free(&G.gen[i]);
    img_free(&G.hud), img_free(&G.real), img_free(&G.dbg), img_free(&G.uia), img_free(&G.sdr), img_free(&G.hud8);
    G.w = G.h = G.rw = G.rh = 0;
}

static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void fail(const char *what) {
    rt_log("dlssg: %s failed: frame generation off for this run", what);
    plat_xchg32(&G.failed, 1);
}

// NGX's "do not show this set" byte (DLSSG.OutputDisableInterpolation): a small host-visible buffer per set, from a
// ring. The set's own frames hear of it on the GPU (shaders/dlssg_hold.comp); the CPU reads it HINTS - 1 sets later,
// only for the log.
static VkBuffer hint_take(VkCommandBuffer cb) {
    const VkCtx *c = vk_ctx();
    unsigned i = (unsigned)(G.sets % HINTS), old = (unsigned)((G.sets + 1) % HINTS);
    if (G.hint[old].pending) {
        vmaInvalidateAllocation(c->vma, G.hint[old].mem, 0, VK_WHOLE_SIZE);
        G.held += ((const uint8_t *)G.hint[old].map)[0] != 0;
        G.hint[old].pending = 0;
    }
    if (G.sets && G.sets % 600 == 0) {
        if (G.held)
            rt_log("dlssg: NGX asked not to show %llu of the last 600 sets (a scene change): the game's frame went out "
                   "in their place", (unsigned long long)G.held);
        G.held = 0;
    }
    if (!G.hint[i].buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 16,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &G.hint[i].buf, &G.hint[i].mem, &info) != VK_SUCCESS)
            return G.hint[i].buf = VK_NULL_HANDLE;
        G.hint[i].map = info.pMappedData;
    }
    vkCmdFillBuffer(cb, G.hint[i].buf, 0, 16, 0);
    G.hint[i].pending = 1;
    return G.hint[i].buf;
}

static void hints_free(void) {
    for (int i = 0; i < HINTS; i++)
        if (G.hint[i].buf) vmaDestroyBuffer(vk_ctx()->vma, G.hint[i].buf, G.hint[i].mem);
    memset(G.hint, 0, sizeof G.hint);
}

// Waits for the device, then lets NGX's feature go (a refused set's, which the GPU may still be running)
static void feat_drop(void) {
    vk_queue_lock();
    vkDeviceWaitIdle(vk_ctx()->device);
    vk_queue_unlock();
    if (G.feat) vk_ngx_fg_release(G.feat);
    G.feat = NULL;
    G.drop_feat = 0;
}

// The images and the feature for this job's sizes, made in cb when missing. 1: ready; *fresh: a new feature (its
// history starts with this frame).
static int ensure(VkCommandBuffer cb, int *fresh) {
    const VkFgFrame *f = &G.fr;
    *fresh = 0;
    // another size, or HDR output turned on or off (the pictures' format, and NGX's feature made for it)
    if (G.real.img && (G.w != G.pw || G.h != G.ph || G.rw != f->rw || G.rh != f->rh || G.img_fmt != G.fmt ||
                       G.img_wfmt != G.wfmt))
        free_all(1);
    if (!G.real.img) {
        G.w = G.pw, G.h = G.ph, G.rw = f->rw, G.rh = f->rh, G.img_fmt = G.fmt, G.img_wfmt = G.wfmt;
        if (!img_make(cb, &G.real, G.w, G.h, G.fmt) || !img_make(cb, &G.hud, G.w, G.h, G.fmt)) {
            free_all(0);
            fail("image creation");
            return 0;
        }
    }
    if (G.hdr && !G.sdr.img &&
        (!img_make(cb, &G.sdr, G.w, G.h, G.wfmt) || !img_make(cb, &G.hud8, G.w, G.h, G.wfmt))) {
        free_all(0);
        fail("image creation");
        return 0;
    }
    for (int i = 0; i < G.n; i++)
        if (!G.gen[i].img && !img_make(cb, &G.gen[i], G.w, G.h, G.fmt)) {
            free_all(0);
            fail("image creation");
            return 0;
        }
    if (G.drop_feat) feat_drop();
    if (!G.feat) {
        barrier(cb);   // the new images' layouts before NGX records anything
        G.uir = vk_dlssg_ui_alpha();
        if (G.uir && !G.uia.img && !img_make(cb, &G.uia, G.w, G.h, VK_FORMAT_R8_UNORM)) G.uir = 0;
        if (!vk_ngx_fg_create(cb, G.w, G.h, G.fmt, G.rw, G.rh, G.uir, &G.feat)) {
            G.feat = NULL;
            if (++G.create_fails >= CREATE_FAILS_OFF) fail("NGX's feature creation, three times in a row,");
            return 0;   // tried again at the next frame
        }
        G.create_fails = 0;
        *fresh = 1;
    }
    return 1;
}

// ---------------------------------------------------------------- the passes of ours
// Compute passes with push descriptors: combined image samplers and one storage image (the output), in binding order.
static PFN_vkCmdPushDescriptorSetKHR push_fn;
static Pass dbg_pass, ua_pass, hold_pass;

static VkSampler sampler(VkFilter f) {
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = f, .minFilter = f,
                              .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkSampler s = VK_NULL_HANDLE;
    vkCreateSampler(vk_ctx()->device, &si, NULL, &s);
    return s;
}

// types[i]: binding i's descriptor type; push constants up to 48 bytes
static int pass_init(Pass *ps, const char *name, const uint32_t *spv, size_t bytes, const VkDescriptorType *types,
                     uint32_t n) {
    if (ps->tried) return ps->ok;
    ps->tried = 1;
    const VkCtx *c = vk_ctx();
    if (!push_fn) push_fn = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!nearest) nearest = sampler(VK_FILTER_NEAREST);
    VkDescriptorSetLayoutBinding b[8];
    for (uint32_t i = 0; i < n; i++)
        b[i] = (VkDescriptorSetLayoutBinding){i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = n, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes,
                                   .pCode = spv};
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r = VK_ERROR_INITIALIZATION_FAILED;
    if (push_fn && nearest && vkCreateDescriptorSetLayout(c->device, &dci, NULL, &ps->dsl) == VK_SUCCESS) {
        VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &ps->dsl, .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &pr};
        if (vkCreatePipelineLayout(c->device, &li, NULL, &ps->layout) == VK_SUCCESS &&
            vkCreateShaderModule(c->device, &mi, NULL, &m) == VK_SUCCESS) {
            VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                              .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                        .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m,
                                                        .pName = "main"},
                                              .layout = ps->layout};
            r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &ps->pipe);
            vkDestroyShaderModule(c->device, m, NULL);
        }
    }
    ps->ok = r == VK_SUCCESS;
    if (!ps->ok) rt_log("dlssg: the %s pass could not be created", name);
    return ps->ok;
}

static void pass_free(Pass *ps) {
    const VkCtx *c = vk_ctx();
    if (ps->pipe) vkDestroyPipeline(c->device, ps->pipe, NULL);
    if (ps->layout) vkDestroyPipelineLayout(c->device, ps->layout, NULL);
    if (ps->dsl) vkDestroyDescriptorSetLayout(c->device, ps->dsl, NULL);
    memset(ps, 0, sizeof *ps);
}

static void passes_release(void) {
    const VkCtx *c = vk_ctx();
    pass_free(&dbg_pass), pass_free(&alpha_pass), pass_free(&ua_pass), pass_free(&hold_pass);
    if (nearest) vkDestroySampler(c->device, nearest, NULL);
    nearest = VK_NULL_HANDLE;
}

// One dispatch over the picture: views[i] for binding i (the storage image at binding out), samplers[i] for the rest.
static void pass_run(VkCommandBuffer cb, Pass *ps, const VkImageView *views, const VkSampler *samplers, uint32_t n,
                     uint32_t out, const void *pc, uint32_t pc_bytes, uint32_t gw, uint32_t gh) {
    VkDescriptorImageInfo ii[8];
    VkWriteDescriptorSet wr[8];
    for (uint32_t i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){i == out ? VK_NULL_HANDLE : samplers[i], views[i], VK_IMAGE_LAYOUT_GENERAL};
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i,
                                      .descriptorCount = 1,
                                      .descriptorType = i == out ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipe);
    rec_push(push_fn, cb, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, n, wr);
    vkCmdPushConstants(cb, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    vkCmdDispatch(cb, (gw + 7) / 8, (gh + 7) / 8, 1);
}

// AC_DLSSG_MVDEBUG
static int mvdebug_mode(void) {
    const char *v = cfg_get("AC_DLSSG_MVDEBUG");
    int m = v && *v ? atoi(v) : 0;
    return m >= 1 && m <= 8 ? m : 0;
}

// After NGX's evaluations in cb: the overlay into the game's frame (G.real) and each generated frame, all from a copy
// of the game's frame, so every frame shown is the game's frame with its inputs.
static void mvdebug(VkCommandBuffer cb, int mode) {
    static const VkDescriptorType t[5] = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    const VkFgFrame *f = &G.fr;
    if (!dbg_pass.tried)
        rt_log("dlssg: AC_DLSSG_MVDEBUG=%d: frame generation's inputs drawn over the frames it shows", mode);
    if (!pass_init(&dbg_pass, "AC_DLSSG_MVDEBUG", dlssg_mvdebug_spv, sizeof dlssg_mvdebug_spv, t, 5)) return;
    if (!G.dbg.img && !img_make(cb, &G.dbg, G.w, G.h, G.fmt)) return;
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {G.w, G.h, 1}};
    vkCmdCopyImage(cb, G.real.img, VK_IMAGE_LAYOUT_GENERAL, G.dbg.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    barrier(cb);
    const char *e = cfg_get("AC_DLSSG_MVDEBUG_SCALE");
    double sc = e ? atof(e) : 0.0;
    for (int k = 0; k <= G.n; k++) {
        VkImageView v[5] = {G.dbg.view, f->mv_view, f->depth_view, k ? G.gen[k - 1].view : G.real.view, G.hud.view};
        VkSampler sm[5] = {nearest, nearest, nearest, VK_NULL_HANDLE, nearest};
        struct {
            int32_t size[2];
            float rsize[2], scale;
            int32_t mode;
            float near_z, far_z;
            int32_t generated;
        } pc = {{(int32_t)G.w, (int32_t)G.h}, {(float)f->rw, (float)f->rh}, sc > 0 ? (float)sc : 8.0f, mode,
                f->near_z, f->far_z, k};
        pass_run(cb, &dbg_pass, v, sm, 5, 3, &pc, sizeof pc, G.w, G.h);
    }
    barrier(cb);
}

// A set NGX said not to show (its byte in hint): the game's frame goes into each generated frame on the GPU
static void hold(VkCommandBuffer cb, VkBuffer hint) {
    static const VkDescriptorType t[3] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    if (!pass_init(&hold_pass, "scene change hold", dlssg_hold_spv, sizeof dlssg_hold_spv, t, 3)) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, hold_pass.pipe);
    int32_t pc[2] = {(int32_t)G.w, (int32_t)G.h};
    vkCmdPushConstants(cb, hold_pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    for (int k = 0; k < G.n; k++) {
        VkDescriptorImageInfo ri = {nearest, G.real.view, VK_IMAGE_LAYOUT_GENERAL},
                              oi = {VK_NULL_HANDLE, G.gen[k].view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo bi = {hint, 0, 16};
        VkWriteDescriptorSet wr[3] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ri},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi}};
        rec_push(push_fn, cb, VK_PIPELINE_BIND_POINT_COMPUTE, hold_pass.layout, 0, 3, wr);
        vkCmdDispatch(cb, (G.w + 7) / 8, (G.h + 7) / 8, 1);
    }
    barrier(cb);
}

// vk_present_pre_record's hook: at the start of the blit's command buffer, before anything reads src. Returns the
// image the blit shows: the first generated frame, or src when none was made.
static VkImage record(VkCommandBuffer cb, VkImage src, unsigned slot, VkFormat *fmt) {
    if (!G.armed || src != G.win) return src;
    G.armed = 0;
    int fresh = 0;
    if (!ensure(cb, &fresh)) return src;
    const VkFgFrame *f = &G.fr;
    barrier(cb);   // the game's writes to the window texture, its scene, depth and motion vectors
    // the HUD-less scene in the window texture's format and size (a blit converts the float and scales a smaller scene
    // of the picture's shape), and the picture itself, kept for its own present after the generated ones. With HDR
    // output both go into 8-bit images of ours first and are encoded from there, with the frame's scene, into the HDR10
    // pictures NGX reads (vk_hdr.c vk_hdr_encode_fg: what the present would show of each).
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {G.w, G.h, 1}};
    vkCmdCopyImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, G.hdr ? G.sdr.img : G.real.img, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    barrier(cb);
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{0, 0, 0}, {(int32_t)f->hw, (int32_t)f->hh, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)G.w, (int32_t)G.h, 1}}};
    vkCmdBlitImage(cb, f->hudless, VK_IMAGE_LAYOUT_GENERAL, G.hdr ? G.hud8.img : G.hud.img, VK_IMAGE_LAYOUT_GENERAL, 1,
                   &b, VK_FILTER_LINEAR);
    barrier(cb);
    if (G.hdr) {
        VkImageView in[2] = {G.sdr.view, G.hud8.view}, out[2] = {G.real.view, G.hud.view};
        if (!vk_hdr_encode_fg(cb, slot, in, out, 2, G.wfmt, G.w, G.h)) return src;   // the present's encode fails too
        barrier(cb);
    }
    if (G.uir) {   // the HUD's coverage the picture's alpha holds (vk_exec_draw.c), as NGX's single channel
        static const VkDescriptorType t[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        if (pass_init(&ua_pass, "UI alpha extraction", dlssg_uialpha_spv, sizeof dlssg_uialpha_spv, t, 2)) {
            VkImageView v[2] = {G.hdr ? G.sdr.view : G.real.view, G.uia.view};
            VkSampler sm[2] = {nearest, VK_NULL_HANDLE};
            int32_t pc[2] = {(int32_t)G.w, (int32_t)G.h};
            pass_run(cb, &ua_pass, v, sm, 2, 1, pc, sizeof pc, G.w, G.h);
            barrier(cb);
        }
    }
    int reset = f->reset || fresh || !G.prev_shown || f->frame_id != G.last_id + 1;
    VkBuffer hint = hint_take(cb);
    if (hint) barrier(cb);   // its fill before NGX writes it
    for (int k = 1; k <= G.n; k++) {
        VkFgEval e = {.back = G.real.img, .back_view = G.real.view, .hudless = G.hud.img, .hudless_view = G.hud.view,
                      .out = G.gen[k - 1].img, .out_view = G.gen[k - 1].view, .format = G.fmt, .w = G.w, .h = G.h,
                      .f = f, .count = (uint32_t)G.n, .index = (uint32_t)k, .reset = reset && k == 1,
                      .ui_alpha = G.uir ? G.uia.img : VK_NULL_HANDLE, .ui_alpha_view = G.uia.view,
                      .hdr = G.hdr, .no_interp = hint, .no_interp_bytes = 16};
        int r = vk_ngx_fg_evaluate(cb, G.feat, &e);
        if (!r || ((unsigned)r & 0xFFF00000u) == 0xBAD00000u) {   // NVSDK_NGX_Result: 1 success, 0xBAD0xxxx failures
            // The set is not shown and the feature is remade (feat_drop, at the next job): a new history. Only a run of
            // refused sets turns frame generation off.
            static int logged;
            if (logged++ < 8)
                rt_log("dlssg: NGX refused frame %d of %d (%08x): the game's frame alone", k, G.n, (unsigned)r);
            if (k > 1) barrier(cb);   // the frames NGX did write, before the blit reads src
            G.drop_feat = 1;
            if (++G.eval_fails >= EVAL_FAILS_OFF) fail("NGX's evaluation, five sets in a row,");
            return src;
        }
    }
    barrier(cb);
    if (hint) {   // NGX's byte, for the CPU's read HINTS - 1 sets later (after the blit's fence)
        VkMemoryBarrier2 hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                               .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                               .pMemoryBarriers = &hb};
        vkCmdPipelineBarrier2(cb, &di);
    }
    G.eval_fails = 0;
    G.last_id = f->frame_id;
    G.sets++;
    G.evaluated = 1;
    if (hint && !reset) hold(cb, hint);
    int dm = mvdebug_mode();
    if (dm) mvdebug(cb, dm);
    *fmt = G.fmt;
    // a new history gives no frame between the last one and this: only the game's frame is shown, at its delayed time
    if (reset) return G.real.img;
    G.made = G.n;
    return G.gen[0].img;
}

int vk_dlssg_begin(const VkFgFrame *f, VkImage win, VkFormat fmt, uint32_t w, uint32_t h, uint64_t period_ns) {
    G.armed = 0, G.made = 0, G.evaluated = 0;
    int s = f->on && !plat_cas32(&G.failed, 0, 0) ? setting() : 0;
    int max = s ? vk_dlssg_max_multiplier() : 0, hdr = vk_hdr_active();
    if (!s || !max || !format_ok(fmt, hdr)) {
        G.prev_shown = 0;
        vk_present_pre_record(NULL);
        return 0;
    }
    int m = s;
    double hz = gfx_window_refresh_hz(), slots = hz > 0 ? hz * (double)period_ns / 1e9 : 0.0;
    if (s == 1) {   // auto: as many frames as the display shows in a game frame
        m = slots > 0 ? (int)(slots + 0.25) : 2;
    } else if (slots > 0 && !strcmp(gfx_window_present_mode(), "fifo") && (int)(slots + 0.25) < m) {
        // FIFO shows one picture per refresh: more than fit in a game frame would queue up behind it, each game frame
        // shown later than the last. As many as fit (at least 2x, the game frame and one between)
        int fit = (int)(slots + 0.25);
        m = fit < 2 ? 2 : fit;
        if (G.capped != m)
            rt_log("dlssg: %dx is more than the display shows at this frame rate under V-Sync (%.0f Hz, %.1f fps): %dx",
                   s, hz, 1e9 / (double)period_ns, m);
        G.capped = m;
    } else {
        G.capped = 0;
    }
    if (m > max) m = max;
    if (m < 2) {
        G.prev_shown = 0;
        vk_present_pre_record(NULL);
        return 0;
    }
    G.fr = *f, G.win = win, G.wfmt = fmt, G.hdr = hdr, G.fmt = hdr ? HDR_FMT : fmt, G.n = m - 1, G.pw = w, G.ph = h;
    G.armed = 1;
    vk_present_pre_record(record);
    return G.n;
}

int vk_dlssg_end(void) {
    vk_present_pre_record(NULL);
    int made = G.armed ? -1 : G.made;   // still armed: the blit did not record (no swapchain image)
    G.armed = 0;
    G.prev_shown = G.evaluated;
    if (made > 0) {
        plat_xchg32(&G.shown, made + 1);
        __atomic_store_n(&G.shown_ns, plat_now_ns(), __ATOMIC_RELAXED);
    }
    return made;
}

VkImage vk_dlssg_image(int k) { return k <= 0 ? G.real.img : k <= G.n ? G.gen[k - 1].img : VK_NULL_HANDLE; }
VkFormat vk_dlssg_format(void) { return G.fmt; }

int vk_dlssg_encoded(VkImage img) {
    if (!G.hdr || !img) return 0;
    int hit = img == G.real.img;
    for (int i = 0; i < MAX_GEN; i++) hit |= img == G.gen[i].img;
    return hit;
}

void vk_dlssg_release(void) {
    free_all(0);
    hints_free();
    capture_free();
    passes_release();
}

// ---------------------------------------------------------------- the menu and the overlay
const char *vk_dlssg_running(void) {
    static const char *const names[] = {"", "", "2x", "3x", "4x", "5x", "6x"};
    int m = plat_cas32(&G.shown, 0, 0);
    uint64_t at = __atomic_load_n(&G.shown_ns, __ATOMIC_RELAXED);
    if (m < 2 || m > 6 || !at || plat_now_ns() - at > 250000000ull) return NULL;
    return names[m];
}

int vk_dlssg_choice_state(void) {
    if (!setting()) return 0;
    if (!vk_dlaa_wanted()) return 1;   // NGX was not loaded at start
    if (!vk_dlssg_max_multiplier() || plat_cas32(&G.failed, 0, 0)) return 2;
    if (!vk_lowlat_oob_ok()) return 1;   // Reflex without frame generation's present queue: made at the next start
    if (!vk_dlaa_running()) return 3;
    return 0;
}
