// AMD FSR 3.1.4 frame generation (frame_generation with the FSR 3 method, vk_dlssg.c): our own Vulkan host for AMD's
// optical flow and frame interpolation passes.
//
// The passes are AMD's, unmodified: tools/ffx_sdk.py fetches them from the FidelityFX SDK v1.1.4 (MIT, pinned commit and
// file hashes), compiles them into ffxfg_spv.h in the build directory and reads back the descriptor bindings each one kept
// (ffxfg_<pass>_bind). This file does what the SDK's ffx_opticalflow.cpp (opticalflowCreate, dispatch) and
// ffx_frameinterpolation.cpp (frameinterpolationCreate, ffxFrameInterpolationPrepare, ffxFrameInterpolationDispatch) do
// for FSR 3's "interpolation only" use (ffx_fsr3.cpp, an upscaler of the game's own), without the SDK's back end: it makes
// the internal images, fills the constant blocks and records the passes in AMD's order, all into the command buffer of
// the blit that shows the frame made:
//   prepare (ffxFrameInterpolationPrepare): the dilated depth and motion vectors and the previous frame's nearest depth
//     from the scene depth and the temporal upscaler's motion vectors (reconstruct and dilate);
//   the optical flow (ffxOpticalflowContextDispatch) on the HUD-less picture: its luma and luma pyramid, the scene
//     change detection's histogram and divergence, then per pyramid level from the coarsest the block search, the filter
//     and the scale to the next level, the last filter writing the flow FSR 3 shares with the interpolation;
//   the interpolation (ffxFrameInterpolationDispatch): setup, then unless the history starts anew the previous frame's
//     depth for the frame between, the game's motion vector field, its inpainting pyramid (SPD), the optical flow's
//     vector field and the disocclusion mask; the interpolation itself into the output, the output's inpainting pyramid
//     and the inpainting, which also puts the HUD back from the picture with it; last, the HUD-less picture is kept as
//     the next frame's previous interpolation source.
// One pass is ours (shaders/ffxfg_check.comp, before the previous frame's depth for the frame between): AMD's passes
// trust the game's motion vectors, and what a pixel shows does not always move with its surface (a light's glow on the
// road that follows the bike, a wet road's reflections). It checks each dilated vector against the two pictures and
// gives the pixel no vector, or the optical flow's, where that explains the pixel and the game's does not; the two
// passes that place this frame's pixels in the frame between read its vectors, and the vector field its copy of the
// depth. The measurements are with CHECK_EPS below.
// The SDK's scheduling details are AMD's as written there: dispatch sizes, the ping-pong of the optical flow's images
// by frame and by level, the clears on a new history (the reconstructed depths to the bit pattern of 1.0, as the SDK's
// back end clears an R32_UINT image with a float), the constant values. Left out: the debug view, the tear and pacing
// lines, the distortion field (a 1x1 zero image, as the SDK binds when none is given), the async path.
//
// Each image is kept in GENERAL. Every pass is followed by a full memory barrier, as vk_fsr3.c does. The constant blocks
// come from a host-visible ring (CB_SLOTS blocks, about 14 a frame). One set of images for the picture's size, the
// scene's and the pictures' format; another makes them again (the old ones go to the deferred destruction).
//
// The optical flow's search pass sums and takes minima over subgroups of 32 or 64 lanes only (ffxWaveLaneCount() == 32
// or the two halves of a group of 64): on a device whose compute runs another width (Intel's 8 or 16) its subgroup
// passes are made with a required size of 32 and full subgroups (VK_EXT_subgroup_size_control, core in Vulkan 1.3);
// where neither is possible, FSR 3 frame generation is not offered.
//
// Adapted from AMD FidelityFX SDK sdk/src/components/opticalflow/ffx_opticalflow.cpp and
// sdk/src/components/frameinterpolation/ffx_frameinterpolation.cpp @ c6efa6bf7f2027b3ec94f28578bb5965eabb9e55 (MIT,
// Copyright (C) 2024 Advanced Micro Devices, Inc.).
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "host.h"
#include "vk.h"
#include "vk_ffxfg.h"
#include "vk_rec.h"
#include "ffxfg_spv.h"   // build directory: tools/ffx_sdk.py
#include "ffxfg_check_spv.h"   // shaders/ffxfg_check.comp: ours
#include "platform/platform.h"

// ---------------------------------------------------------------- the passes
enum { FG_OF_PREPARE_LUMA, FG_OF_LUMA_PYRAMID, FG_OF_SCD_HISTOGRAM, FG_OF_SCD_DIVERGENCE, FG_OF_SEARCH, FG_OF_FILTER,
       FG_OF_SCALE, FG_FI_RECONSTRUCT_AND_DILATE, FG_FI_SETUP, FG_FI_RECONSTRUCT_PREV_DEPTH, FG_FI_GAME_MV_FIELD,
       FG_FI_GAME_PYRAMID, FG_FI_OF_VECTOR_FIELD, FG_FI_DISOCCLUSION, FG_FI_INTERPOLATION, FG_FI_INPAINTING_PYRAMID,
       FG_FI_INPAINTING, FG_CHECK, FG_N };
enum { K_SAMPLED, K_STORAGE, K_UNIFORM, K_SSBO, K_SAMPLER };   // tools/ffx_sdk.py KIND_*
// shaders/ffxfg_check.comp's bindings, as tools/ffx_sdk.py lists AMD's
static const uint16_t ffxfg_check_bind[][2] = {{0, K_SAMPLED}, {1, K_SAMPLED}, {2, K_SAMPLED}, {3, K_SAMPLED},
                                               {4, K_STORAGE}, {5, K_STORAGE}, {6, K_UNIFORM}, {7, K_SAMPLER},
                                               {8, K_SAMPLED}};
#define PASS(n, sg) {#n, ffxfg_##n##_spv, sizeof ffxfg_##n##_spv, ffxfg_##n##_bind, \
                     (uint32_t)(sizeof ffxfg_##n##_bind / sizeof ffxfg_##n##_bind[0]), sg}
// sg: the pass uses subgroup operations (SPD's quad reductions, the search's sums and minima), local size a multiple of
// 64: made with the required subgroup size where one is needed
static const struct {
    const char *name;
    const uint32_t *spv;
    size_t bytes;
    const uint16_t (*bind)[2];
    uint32_t nb;
    int sg;
} PASSES[FG_N] = {
    PASS(of_prepare_luma, 0), PASS(of_luma_pyramid, 1), PASS(of_scd_histogram, 0), PASS(of_scd_divergence, 0),
    PASS(of_search, 1), PASS(of_filter, 0), PASS(of_scale, 0), PASS(fi_reconstruct_and_dilate, 0), PASS(fi_setup, 0),
    PASS(fi_reconstruct_previous_depth, 0), PASS(fi_game_motion_vector_field, 0), PASS(fi_game_vector_field_pyramid, 1),
    PASS(fi_optical_flow_vector_field, 0), PASS(fi_disocclusion_mask, 0), PASS(fi_interpolation, 0),
    PASS(fi_inpainting_pyramid, 1), PASS(fi_inpainting, 0), PASS(check, 0),
};
#undef PASS
#define MAX_BIND 24

// ---------------------------------------------------------------- the images
// The optical flow's (AMD's FFX_OF_RESOURCE_IDENTIFIER_*): its luma input and the flow, two of each (this frame's and
// the last), each a pyramid of OF_LEVELS images; the scene change detection's histograms; the flow and the scene change
// result it shares with the interpolation. The interpolation's (FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_*): the
// prepare pass's dilated depth and motion vectors and previous nearest depth (the shared resources an FSR 3 upscaler
// would have made), its own fields at the scene's size, the inpainting pyramid (half the picture's size, every mip), the
// previous interpolation source and the default distortion field. Ours (ffxfg_check.comp): the dilated motion vectors
// after the check and the dilated depth the vector field reads.
#define OF_LEVELS 7
#define OF_BLOCK 8
#define SCD_BINS (256 * 3 * 3)   // GetSCDHistogramTextureWidth: 256 bins, 3 x 3 histograms
enum { I_OF_IN1 = 0, I_OF_IN2 = I_OF_IN1 + OF_LEVELS, I_OF_FLOW1 = I_OF_IN2 + OF_LEVELS, I_OF_FLOW2 = I_OF_FLOW1 + OF_LEVELS,
       I_OF_HIST = I_OF_FLOW2 + OF_LEVELS, I_OF_HIST_PREV, I_OF_SCD_TEMP, I_OF_VECTOR, I_OF_SCD, I_DDEPTH, I_DMV,
       I_RECON_PREV, I_RECON_INTERP, I_GMV_X, I_GMV_Y, I_OFMV_X, I_OFMV_Y, I_DISOCC, I_PYRAMID, I_PREV_SRC, I_DEF_DIST,
       I_DMV_CHECKED, I_DDEPTH_FIELD, I_N };
#define PYR_VIEWS 13   // the inpainting pyramid's storage bindings (mips 0..12)
#define CB_SLOTS 1024u

// AMD's cbFI (FrameInterpolationConstants), std140: every member on its natural offset here
typedef struct {
    int32_t render[2], display[2];
    float display_rcp[2], near_z, far_z;
    int32_t upscaler_target[2], mode, reset;
    float device_to_view[4];
    float dt;
    int32_t hudless;   // HUDLessAttachedFactor
    int32_t distortion_size[2];
    float of_scale[2];
    int32_t of_block;
    uint32_t dispatch_flags;
    int32_t max_render[2], of_half_res, instances;
    int32_t rect_base[2], rect_size[2];
    float debug_bar[3];
    uint32_t transfer;
    float min_max_lum[2], tan_half_fov, pad1;
    float jitter[2], mv_scale[2];
} FiConsts;
_Static_assert(sizeof(FiConsts) == 176, "cbFI");
typedef struct { uint32_t mips, groups, offset[2]; } PyrConsts;   // cbInpaintingPyramid
typedef struct {                                                   // cbOF
    int32_t luma_res[2];
    uint32_t level, levels, frame, transfer;
    float min_max_lum[2];
} OfConsts;
typedef struct { uint32_t mips, groups, offset[2], groups_pyramid, pad[3]; } OfSpdConsts;   // cbOF_SPD
typedef struct { float k[4], d[4]; } CheckConsts;                                            // ffxfg_check.comp K
// ffxfg_check.comp: a vector explains its pixel when the colour it reads is within CHECK_EPS of the pixel's (of 1: 6
// of 255); the game's gives way when it is off by 2 times the other's error plus CHECK_OFF; no vector replaces one of
// up to CHECK_FROM scene pixels; and a pixel given another vector goes back by CHECK_BACK in view space for the vector
// field. Measured on the new game's motorway (Arc 140T, 2560x1440, scene 1696x954, 46 fps falling to 32, the same 7
// frames of two runs, 46 to 58 s after the first frame made), without the check and with it:
//   still content (pixels within 2 of 255 in both pictures over a 2x2 box, the HUD left out) that the frame made shows
//   off by over 8: 12.1 % of it, then 4.2 %; by over 24: 4.3 %, then 1.3 %;
//   the side view the sequence starts on, around the bike: pixels over 24 from both pictures in place and from every
//   vector's answer: 4.57 %, then 0.29 %;
//   moving detail that the game's vectors do explain, off by over 16: 0.19 %, then 0.33 % (what the check costs).
// The choice per pixel was modelled on those frames against the pictures' own motion (48x48 blocks matched between
// the two pictures): in the blocks where the game's vector was off, the pixels carrying the picture's motion went from
// 13 % to 57 % where the picture is still and from 9 % to 28 % where it moves, and from 96.0 % to 97.1 % where the
// game's vector was right. Deciding per 8x8 block over 16 reads first gave 62 %, 31 % and 96.9 %: not worth a second
// pass. Thresholds of 4 and 4 or 8 and 8 moved the still content's figure by under 2 points. CHECK_BACK was tried at
// 1.05, 1.5 and 4 (with the still test alone): still content off 3.6 %, 4.0 % and 4.0 %, moving detail off 0.49 %,
// 0.31 % and 0.24 %; 1.5 is 29 of the field's 1023 priority levels at 10 m (a level is (1 + d^0.33)^-1 / 1023), so
// a pixel of the same surface whose vector holds wins the place. The pass costs about
// 0.3 ms of the 9.7 ms a frame made takes there.
#define CHECK_EPS (6.0f / 255.0f)
#define CHECK_OFF (6.0f / 255.0f)
#define CHECK_FROM 8.0f
#define CHECK_BACK 1.5f

static struct {
    int tried, ok, failed, logged_why;
    int supported;               // -1 not checked
    uint32_t req_sg;             // the required subgroup size for the subgroup passes, 0 none
    int full_sg;                 // and with full subgroups
    VkSampler linear;
    VkDescriptorSetLayout dsl[FG_N];
    VkPipelineLayout layout[FG_N];
    VkPipeline pipe[FG_N];
    VkBuffer cb;
    VmaAllocation cb_mem;
    uint8_t *cb_map;
    uint32_t cb_stride, cb_next;
    VkBuffer counters;           // FI_Counters: two uints
    VmaAllocation counters_mem;
    // the images, for one picture size, scene size and pictures' format
    uint32_t w, h, rw, rh;
    VkFormat fmt;
    VkImage img[I_N];
    VmaAllocation mem[I_N];
    VkImageView view[I_N];
    VkImageView pyr[PYR_VIEWS];
    uint32_t of_w[OF_LEVELS], of_h[OF_LEVELS];   // the flow's size per level
    int fresh;                   // the images were just made: a new history
    uint32_t of_frame;           // AMD's resourceFrameIndex
    uint32_t of_index;           // the optical flow constants' frameIndex
    uint64_t dispatches, prev_id;
    // GPU time (TS_SLOTS evaluations of three timestamps: start, after the optical flow, end), read TS_SLOTS - 1
    // evaluations later when the blit's fence has long passed, averaged and logged every TS_LOG evaluations
    VkQueryPool ts;
    double ts_period_ms;
    uint32_t ts_next;
    double ts_of, ts_all;
    uint32_t ts_n;
} X = {.supported = -1};
#define TS_SLOTS 16u
#define TS_LOG 600u

static VkImageView make_view(VkImage img, VkFormat f, uint32_t level, uint32_t levels) {
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, levels, 0, 1}};
    VkImageView v = VK_NULL_HANDLE;
    return vkCreateImageView(vk_ctx()->device, &vi, NULL, &v) == VK_SUCCESS ? v : VK_NULL_HANDLE;
}

static void destroy_images(void) {
    for (int i = 0; i < PYR_VIEWS; i++) vk_defer_view(X.pyr[i]), X.pyr[i] = VK_NULL_HANDLE;
    for (int i = 0; i < I_N; i++) {
        vk_defer_view(X.view[i]), vk_defer_image(X.img[i], X.mem[i]);
        X.view[i] = VK_NULL_HANDLE, X.img[i] = VK_NULL_HANDLE, X.mem[i] = NULL;
    }
    X.w = X.h = X.rw = X.rh = 0;
}

void vk_ffxfg_release(void) {
    if (X.w) {
        destroy_images();
        rt_log("ffxfg: work images released");
    }
}

void vk_ffxfg_shutdown(void) {
    const VkCtx *c = vk_ctx();
    if (!c || !c->device) return;
    for (int i = 0; i < PYR_VIEWS; i++)
        if (X.pyr[i]) vkDestroyImageView(c->device, X.pyr[i], NULL);
    for (int i = 0; i < I_N; i++) {
        if (X.view[i]) vkDestroyImageView(c->device, X.view[i], NULL);
        if (X.img[i]) vmaDestroyImage(c->vma, X.img[i], X.mem[i]);
    }
    if (X.cb) vmaDestroyBuffer(c->vma, X.cb, X.cb_mem);
    if (X.ts) vkDestroyQueryPool(c->device, X.ts, NULL);
    if (X.counters) vmaDestroyBuffer(c->vma, X.counters, X.counters_mem);
    for (int p = 0; p < FG_N; p++) {
        if (X.pipe[p]) vkDestroyPipeline(c->device, X.pipe[p], NULL);
        if (X.layout[p]) vkDestroyPipelineLayout(c->device, X.layout[p], NULL);
        if (X.dsl[p]) vkDestroyDescriptorSetLayout(c->device, X.dsl[p], NULL);
    }
    if (X.linear) vkDestroySampler(c->device, X.linear, NULL);
    memset(&X, 0, sizeof X);
    X.supported = -1;
}

// image i's format and size for a picture of w x h, a scene of rw x rh and pictures of format f
static void image_desc(int i, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh, VkFormat f, VkFormat *fmt, uint32_t *iw,
                       uint32_t *ih, uint32_t *levels) {
    *levels = 1;
    if (i < I_OF_FLOW1) {   // the luma pyramid: the picture's size >> level, R8_UINT
        uint32_t l = (uint32_t)(i - (i < I_OF_IN2 ? I_OF_IN1 : I_OF_IN2));
        *fmt = VK_FORMAT_R8_UINT, *iw = w >> l ? w >> l : 1, *ih = h >> l ? h >> l : 1;
        return;
    }
    if (i < I_OF_HIST) {    // the flow: a vector per 8 x 8 block, halved per level
        uint32_t l = (uint32_t)(i - (i < I_OF_FLOW2 ? I_OF_FLOW1 : I_OF_FLOW2));
        *fmt = VK_FORMAT_R16G16_SINT, *iw = X.of_w[l], *ih = X.of_h[l];
        return;
    }
    switch (i) {
    case I_OF_HIST: *fmt = VK_FORMAT_R32_UINT, *iw = SCD_BINS, *ih = 1; return;
    case I_OF_HIST_PREV: *fmt = VK_FORMAT_R32_SFLOAT, *iw = SCD_BINS, *ih = 1; return;
    case I_OF_SCD_TEMP: case I_OF_SCD: *fmt = VK_FORMAT_R32_UINT, *iw = 3, *ih = 1; return;
    case I_OF_VECTOR: *fmt = VK_FORMAT_R16G16_SINT, *iw = X.of_w[0], *ih = X.of_h[0]; return;
    case I_DDEPTH: case I_DDEPTH_FIELD: *fmt = VK_FORMAT_R32_SFLOAT, *iw = rw, *ih = rh; return;
    case I_DMV: case I_DMV_CHECKED: *fmt = VK_FORMAT_R16G16_SFLOAT, *iw = rw, *ih = rh; return;
    case I_DISOCC: *fmt = VK_FORMAT_R8G8_UNORM, *iw = rw, *ih = rh; return;
    case I_PYRAMID: {
        uint32_t hw = w / 2 ? w / 2 : 1, hh = h / 2 ? h / 2 : 1, big = hw > hh ? hw : hh;
        *fmt = VK_FORMAT_R16G16B16A16_SFLOAT, *iw = hw, *ih = hh;
        while (big >>= 1) ++*levels;
        return;
    }
    case I_PREV_SRC: *fmt = f, *iw = w, *ih = h; return;
    case I_DEF_DIST: *fmt = VK_FORMAT_R8G8_UNORM, *iw = 1, *ih = 1; return;
    default: *fmt = VK_FORMAT_R32_UINT, *iw = rw, *ih = rh; return;   // the depths and the motion vector fields
    }
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

static void clear_u(VkCommandBuffer cb, int i, uint32_t v) {
    VkClearColorValue c = {.uint32 = {v, v, v, v}};
    VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
    vkCmdClearColorImage(cb, X.img[i], VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r);
}

static int make_images(VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh, VkFormat f) {
    const VkCtx *c = vk_ctx();
    destroy_images();
    X.of_w[0] = (w + OF_BLOCK - 1) / OF_BLOCK, X.of_h[0] = (h + OF_BLOCK - 1) / OF_BLOCK;
    for (int l = 1; l < OF_LEVELS; l++)   // FFX_ALIGN_UP(size, 2) / 2
        X.of_w[l] = (X.of_w[l - 1] + 1) / 2, X.of_h[l] = (X.of_h[l - 1] + 1) / 2;
    VkImageMemoryBarrier2 ib[I_N];
    uint64_t bytes = 0;
    for (int i = 0; i < I_N; i++) {
        VkFormat fmt;
        uint32_t iw, ih, levels;
        image_desc(i, w, h, rw, rh, f, &fmt, &iw, &ih, &levels);
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = fmt, .extent = {iw, ih, 1}, .mipLevels = levels, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VmaAllocationInfo info;
        if (vmaCreateImage(c->vma, &ci, &ai, &X.img[i], &X.mem[i], &info) != VK_SUCCESS) {
            rt_log("ffxfg: image %d (%ux%u, format %d) could not be made", i, iw, ih, (int)fmt);
            X.img[i] = VK_NULL_HANDLE, X.mem[i] = NULL;
            destroy_images();
            return 0;
        }
        bytes += info.size;
        if (!(X.view[i] = make_view(X.img[i], fmt, 0, levels))) {
            destroy_images();
            return 0;
        }
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = X.img[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1}};
    }
    VkFormat pf;
    uint32_t pw, ph, plev;
    image_desc(I_PYRAMID, w, h, rw, rh, f, &pf, &pw, &ph, &plev);
    for (uint32_t m = 0; m < PYR_VIEWS; m++)   // a smaller pyramid has fewer mips: the last one stands in for the rest
        if (!(X.pyr[m] = make_view(X.img[I_PYRAMID], pf, m < plev ? m : plev - 1, 1))) {
            destroy_images();
            return 0;
        }
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = I_N,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    clear_u(cb, I_DEF_DIST, 0);   // no distortion
    vkCmdFillBuffer(cb, X.counters, 0, VK_WHOLE_SIZE, 0);
    barrier(cb);
    X.w = w, X.h = h, X.rw = rw, X.rh = rh, X.fmt = f, X.fresh = 1, X.of_frame = 0, X.of_index = 0;
    rt_log("ffxfg: work images for a %ux%u picture (format %d) and a %ux%u scene: %u images, %.1f MiB", w, h, (int)f, rw,
           rh, (unsigned)I_N, (double)bytes / (1024.0 * 1024.0));
    return 1;
}

// ---------------------------------------------------------------- the device
static int format_has(VkFormat f, VkFormatFeatureFlags want) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &fp);
    return (fp.optimalTilingFeatures & want) == want;
}

static int why_not(const char *what) {
    if (!X.logged_why) X.logged_why = 1, rt_log("ffxfg: FSR 3 frame generation not offered here: %s", what);
    return X.supported = 0;
}

int vk_ffxfg_supported(void) {
    if (X.supported >= 0) return X.supported;
    const VkCtx *c = vk_ctx();
    if (!c || !c->device) return 0;
    VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    if (!f2.features.shaderStorageImageReadWithoutFormat || !f2.features.shaderStorageImageWriteWithoutFormat)
        return why_not("no storage image reads and writes without a format");
    if (!f2.features.shaderStorageImageExtendedFormats) return why_not("no extended storage image formats");
    VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sg = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = &p13};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg};
    vkGetPhysicalDeviceProperties2(c->phys, &p2);
    VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                 VK_SUBGROUP_FEATURE_QUAD_BIT;
    if (!(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || (sg.supportedOperations & ops) != ops)
        return why_not("no subgroup arithmetic and quad operations in compute shaders");
    // 32 or 64 lanes, as the search pass's reductions need: the device's own width, or 32 required per pipeline
    X.req_sg = 0;
    int ctl = f13.subgroupSizeControl && (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
              p13.minSubgroupSize <= 32 && p13.maxSubgroupSize >= 32;
    if (sg.subgroupSize != 32 && sg.subgroupSize != 64) {
        if (!ctl) return why_not("compute subgroups are neither 32 nor 64 lanes wide and cannot be made so");
        X.req_sg = 32;
    }
    uint32_t lanes = X.req_sg ? X.req_sg : (ctl ? p13.maxSubgroupSize : sg.subgroupSize);
    X.full_sg = f13.computeFullSubgroups && 64 % lanes == 0;   // the subgroup passes' groups are 64 or 256 wide
    if (c->props.limits.maxPerStageDescriptorStorageImages < 16) return why_not("too few storage images per stage");
    static const struct { VkFormat f; int atomic; } FMT[] = {
        {VK_FORMAT_R8_UINT, 0}, {VK_FORMAT_R16G16_SINT, 0}, {VK_FORMAT_R32_UINT, 1}, {VK_FORMAT_R32_SFLOAT, 0},
        {VK_FORMAT_R16G16_SFLOAT, 0}, {VK_FORMAT_R8G8_UNORM, 0}, {VK_FORMAT_R16G16B16A16_SFLOAT, 0}};
    for (size_t i = 0; i < sizeof FMT / sizeof FMT[0]; i++) {
        VkFormatFeatureFlags want = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                    VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                    (FMT[i].atomic ? VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT : 0);
        if (!format_has(FMT[i].f, want)) {
            char b[96];
            snprintf(b, sizeof b, "format %d cannot be the storage image AMD's passes need", (int)FMT[i].f);
            return why_not(b);
        }
    }
    rt_log("ffxfg: FSR 3 frame generation offered here (compute subgroups of %u lanes%s%s)", sg.subgroupSize,
           X.req_sg ? ", 32 required for AMD's subgroup passes" : "", X.full_sg ? ", full subgroups" : "");
    return X.supported = 1;
}

int vk_ffxfg_failed(void) { return (X.tried && !X.ok) || X.failed; }

// The pipelines, at the first evaluation (on the present thread, once the setting asks for it).
static int init(void) {
    X.tried = 1;
    const VkCtx *c = vk_ctx();
    if (!vk_ffxfg_supported() || !vk_pipeline_push_fn()) return 0;
    uint64_t t0 = plat_now_ns();
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                              .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = VK_LOD_CLAMP_NONE};
    if (vkCreateSampler(c->device, &si, NULL, &X.linear) != VK_SUCCESS) return 0;   // s_LinearClamp
    static const VkDescriptorType T[5] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          VK_DESCRIPTOR_TYPE_SAMPLER};
    for (int p = 0; p < FG_N; p++) {
        VkDescriptorSetLayoutBinding b[MAX_BIND];
        if (PASSES[p].nb > MAX_BIND) return 0;
        for (uint32_t i = 0; i < PASSES[p].nb; i++)
            b[i] = (VkDescriptorSetLayoutBinding){PASSES[p].bind[i][0], T[PASSES[p].bind[i][1]], 1,
                                                  VK_SHADER_STAGE_COMPUTE_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                               .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                               .bindingCount = PASSES[p].nb, .pBindings = b};
        if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &X.dsl[p]) != VK_SUCCESS) return 0;
        VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &X.dsl[p]};
        if (vkCreatePipelineLayout(c->device, &li, NULL, &X.layout[p]) != VK_SUCCESS) return 0;
        VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = PASSES[p].bytes,
                                       .pCode = PASSES[p].spv};
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return 0;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .requiredSubgroupSize = X.req_sg};
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                          .layout = X.layout[p]};
        if (PASSES[p].sg && X.req_sg) ci.stage.pNext = &rs;
        if (PASSES[p].sg && X.full_sg) ci.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        VkResult r = vkCreateComputePipelines(c->device, vk_pipeline_cache(), 1, &ci, NULL, &X.pipe[p]);
        vkDestroyShaderModule(c->device, m, NULL);
        if (r != VK_SUCCESS) {
            rt_log("ffxfg: the %s pipeline could not be made (%d)", PASSES[p].name, (int)r);
            X.pipe[p] = VK_NULL_HANDLE;
            return 0;
        }
    }
    VkDeviceSize al = c->props.limits.minUniformBufferOffsetAlignment;
    X.cb_stride = (uint32_t)((sizeof(FiConsts) + al - 1) / al * al);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)CB_SLOTS * X.cb_stride,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo minfo;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &X.cb, &X.cb_mem, &minfo) != VK_SUCCESS) return 0;
    X.cb_map = minfo.pMappedData;
    VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 16,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo di = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(c->vma, &ci, &di, &X.counters, &X.counters_mem, NULL) != VK_SUCCESS) return 0;
    if (c->props.limits.timestampComputeAndGraphics && c->props.limits.timestampPeriod > 0) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                    .queryCount = TS_SLOTS * 3};
        if (vkCreateQueryPool(c->device, &qi, NULL, &X.ts) != VK_SUCCESS) X.ts = VK_NULL_HANDLE;
        X.ts_period_ms = c->props.limits.timestampPeriod / 1e6;
    }
    rt_log("ffxfg: AMD FSR 3.1.4 frame generation ready, %d pipelines in %.1f ms", FG_N, (plat_now_ns() - t0) / 1e6);
    return X.ok = 1;
}

// ---------------------------------------------------------------- recording
typedef struct { uint16_t b; VkImageView v; VkDescriptorBufferInfo buf; } Bind;
#define IMG(bnd, view) {(bnd), (view), {0}}
#define BUF(bnd, info) {(bnd), VK_NULL_HANDLE, (info)}

static VkDescriptorBufferInfo upload(const void *data, uint32_t bytes) {
    uint32_t slot = X.cb_next++ % CB_SLOTS;
    memcpy(X.cb_map + (size_t)slot * X.cb_stride, data, bytes);
    vmaFlushAllocation(vk_ctx()->vma, X.cb_mem, (VkDeviceSize)slot * X.cb_stride, bytes);   // no-op on coherent memory
    return (VkDescriptorBufferInfo){X.cb, (VkDeviceSize)slot * X.cb_stride, bytes};
}

static VkDescriptorBufferInfo counters(void) { return (VkDescriptorBufferInfo){X.counters, 0, 8}; }

// One pass: every binding glslc kept (ffxfg_<pass>_bind) from bs (an unused one may be given and is left out), the
// dispatch and a barrier. 0: a binding has nothing given, which is a fault of this file: logged, frame generation off.
static int run(VkCommandBuffer cb, int p, const Bind *bs, int nbs, uint32_t gx, uint32_t gy, uint32_t gz) {
    VkWriteDescriptorSet w[MAX_BIND];
    VkDescriptorImageInfo ii[MAX_BIND];
    VkDescriptorBufferInfo bi[MAX_BIND];
    static const VkDescriptorType T[5] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          VK_DESCRIPTOR_TYPE_SAMPLER};
    for (uint32_t i = 0; i < PASSES[p].nb; i++) {
        uint16_t b = PASSES[p].bind[i][0], k = PASSES[p].bind[i][1];
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b,
                                      .descriptorCount = 1, .descriptorType = T[k]};
        if (k == K_SAMPLER) {
            ii[i] = (VkDescriptorImageInfo){X.linear, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            w[i].pImageInfo = &ii[i];
            continue;
        }
        int j = 0;
        while (j < nbs && bs[j].b != b) j++;
        if (j == nbs || ((k == K_SAMPLED || k == K_STORAGE) ? !bs[j].v : !bs[j].buf.buffer)) {
            rt_log("ffxfg: pass %s: binding %u has nothing bound: FSR 3 frame generation off for this run",
                   PASSES[p].name, b);
            return 0;
        }
        if (k == K_SAMPLED || k == K_STORAGE) {
            ii[i] = (VkDescriptorImageInfo){VK_NULL_HANDLE, bs[j].v, VK_IMAGE_LAYOUT_GENERAL};
            w[i].pImageInfo = &ii[i];
        } else {
            bi[i] = bs[j].buf;
            w[i].pBufferInfo = &bi[i];
        }
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, X.pipe[p]);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, X.layout[p], 0, PASSES[p].nb, w);
    vkCmdDispatch(cb, gx ? gx : 1, gy ? gy : 1, gz ? gz : 1);
    barrier(cb);
    return 1;
}

// AMD's ffxSpdSetup (ffx_spd.h): thread groups over the rectangle {0, 0, w, h} and its mip count (mips < 0: from its size)
static void spd_setup(uint32_t w, uint32_t h, int mips, uint32_t groups[2], uint32_t *ngroups, uint32_t *nmips) {
    groups[0] = (w - 1) / 64 + 1, groups[1] = (h - 1) / 64 + 1;
    *ngroups = groups[0] * groups[1];
    if (mips >= 0) {
        *nmips = (uint32_t)mips;
    } else {
        uint32_t r = w > h ? w : h, l = 0;
        while (r >>= 1) l++;
        *nmips = l < 12 ? l : 12;
    }
}

#define N(a) ((int)(sizeof a / sizeof a[0]))

// The GPU time of the evaluation TS_SLOTS - 1 before this one, into the averages; logged every TS_LOG evaluations
static void timing(VkCommandBuffer cb, int stamp) {
    if (!X.ts) return;
    uint32_t slot = X.ts_next % TS_SLOTS;
    if (stamp == 0) {
        uint64_t old[3];
        uint32_t prev = (X.ts_next + 1) % TS_SLOTS;   // the oldest slot, about to be reused
        if (X.ts_next >= TS_SLOTS - 1 &&
            vkGetQueryPoolResults(vk_ctx()->device, X.ts, prev * 3, 3, sizeof old, old, sizeof old[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && old[2] >= old[0]) {
            X.ts_of += (double)(old[1] - old[0]) * X.ts_period_ms, X.ts_all += (double)(old[2] - old[0]) * X.ts_period_ms;
            if (++X.ts_n == TS_LOG) {
                rt_log("ffxfg: GPU time per frame made, mean of %u: %.2f ms (prepare and optical flow %.2f ms, the "
                       "interpolation %.2f ms) at %ux%u, scene %ux%u", X.ts_n, X.ts_all / X.ts_n, X.ts_of / X.ts_n,
                       (X.ts_all - X.ts_of) / X.ts_n, X.w, X.h, X.rw, X.rh);
                X.ts_of = X.ts_all = 0, X.ts_n = 0;
            }
        }
        vkCmdResetQueryPool(cb, X.ts, slot * 3, 3);
    }
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, X.ts, slot * 3 + (uint32_t)stamp);
    if (stamp == 2) X.ts_next++;
}

// ffxOpticalflowContextDispatch on the HUD-less picture
static int optical_flow(VkCommandBuffer cb, const FfxFgEval *e, int reset) {
    uint32_t w = X.w, h = X.h;
    if (reset) {
        clear_u(cb, I_OF_SCD_TEMP, 0), clear_u(cb, I_OF_SCD, 0), clear_u(cb, I_OF_HIST, 0), clear_u(cb, I_OF_HIST_PREV, 0);
        for (int l = 0; l < OF_LEVELS; l++) clear_u(cb, I_OF_IN1 + l, 0), clear_u(cb, I_OF_IN2 + l, 0);
        barrier(cb);
        X.of_index = 0;
    } else {
        X.of_index++;
    }
    int odd = X.of_frame & 1;
    int in_a = odd ? I_OF_IN2 : I_OF_IN1, in_b = odd ? I_OF_IN1 : I_OF_IN2;
    OfConsts k = {.luma_res = {(int32_t)w, (int32_t)h}, .level = 0, .levels = OF_LEVELS, .frame = X.of_index,
                  .transfer = e->pq ? 1u : 0u, .min_max_lum = {0.0f, e->pq ? e->peak_nits : 0.0f}};
    VkDescriptorBufferInfo cbk = upload(&k, sizeof k);
    uint32_t g[2], ng, nm;
    spd_setup(w, h, 4, g, &ng, &nm);
    OfSpdConsts sk = {.mips = nm, .groups = ng, .groups_pyramid = ng};
    VkDescriptorBufferInfo cbs = upload(&sk, sizeof sk);
    {
        const Bind b[] = {IMG(0, e->hudless_view), IMG(1, X.view[in_a]), BUF(2, cbk)};
        if (!run(cb, FG_OF_PREPARE_LUMA, b, N(b), ((w + 1) / 2 + 15) / 16, ((h + 1) / 2 + 15) / 16, 1)) return 0;
    }
    {
        Bind b[OF_LEVELS + 2];
        for (int l = 0; l < OF_LEVELS; l++) b[l] = (Bind)IMG((uint16_t)l, X.view[in_a + l]);
        b[OF_LEVELS] = (Bind)BUF(7, cbk), b[OF_LEVELS + 1] = (Bind)BUF(8, cbs);
        if (!run(cb, FG_OF_LUMA_PYRAMID, b, N(b), g[0], g[1], 1)) return 0;
    }
    {
        const Bind b[] = {IMG(0, X.view[in_a]), IMG(1, X.view[I_OF_HIST]), BUF(2, cbk)};
        if (!run(cb, FG_OF_SCD_HISTOGRAM, b, N(b), ((w / 4) / 3 + 31) / 32, 16, 9)) return 0;
    }
    {
        const Bind b[] = {IMG(0, X.view[I_OF_HIST]), IMG(1, X.view[I_OF_HIST_PREV]), IMG(2, X.view[I_OF_SCD_TEMP]),
                          IMG(3, X.view[I_OF_SCD]), BUF(4, cbk)};
        if (!run(cb, FG_OF_SCD_DIVERGENCE, b, N(b), 9, 3, 1)) return 0;
    }
    for (int level = OF_LEVELS - 1; level >= 0; level--) {
        int odd_level = level & 1;
        int fa = (odd != odd_level) ? I_OF_FLOW2 : I_OF_FLOW1, fb = (odd != odd_level) ? I_OF_FLOW1 : I_OF_FLOW2;
        k.level = (uint32_t)level;
        VkDescriptorBufferInfo cl = upload(&k, sizeof k);
        uint32_t lw = w >> level ? w >> level : 1, lh = h >> level ? h >> level : 1;
        {
            const Bind b[] = {IMG(0, X.view[in_a + level]), IMG(1, X.view[in_b + level]), IMG(2, X.view[fa + level]),
                              IMG(3, X.view[I_OF_SCD]), BUF(4, cl)};
            if (!run(cb, FG_OF_SEARCH, b, N(b), (((lw + 3) / 4) * 16 + 63) / 64, (lh + 15) / 16, 1)) return 0;
        }
        {
            const Bind b[] = {IMG(0, X.view[fa + level]), IMG(1, X.view[level ? fb + level : I_OF_VECTOR]), BUF(2, cl)};
            if (!run(cb, FG_OF_FILTER, b, N(b), (X.of_w[level] + 15) / 16, (X.of_h[level] + 3) / 4, 1)) return 0;
        }
        if (level > 0) {
            const Bind b[] = {IMG(0, X.view[in_a + level]), IMG(1, X.view[in_b + level]), IMG(2, X.view[fb + level]),
                              IMG(3, X.view[fb + level - 1]), IMG(4, X.view[I_OF_SCD]), BUF(5, cl)};
            if (!run(cb, FG_OF_SCALE, b, N(b), (X.of_w[level - 1] + 3) / 4, (X.of_h[level - 1] + 3) / 4, 1)) return 0;
        }
    }
    X.of_frame = (X.of_frame + 1) % 16;   // FFX_OPTICALFLOW_MAX_QUEUED_FRAMES
    return 1;
}

int vk_ffxfg_evaluate(VkCommandBuffer cb, const FfxFgEval *e) {
    if (X.failed) return 0;
    if (!X.tried && !init()) {
        rt_log("ffxfg: the FSR 3 frame generation host could not be set up: off for this run");
        X.failed = 1;
        return 0;
    }
    if ((X.w != e->w || X.h != e->h || X.rw != e->rw || X.rh != e->rh || X.fmt != e->format) &&
        !make_images(cb, e->w, e->h, e->rw, e->rh, e->format)) {
        rt_log("ffxfg: the work images could not be made: FSR 3 frame generation off for this run");
        X.failed = 1;
        return 0;
    }
    // a new history: asked for, new images, the first dispatch, or a frame number out of sequence (AMD's disjoint IDs)
    int reset = e->reset || X.fresh || !X.dispatches || e->frame_id < X.prev_id || e->frame_id - X.prev_id > 1;
    X.fresh = 0, X.prev_id = e->frame_id, X.dispatches++;
    uint32_t w = X.w, h = X.h, rw = X.rw, rh = X.rh;

    // the constants (ffxFrameInterpolationPrepare, ffxFrameInterpolationDispatch, setupDeviceDepthToViewSpaceDepthParams
    // for depth near 0 to far 1, not inverted, not infinite, one view unit a metre)
    FiConsts k;
    memset(&k, 0, sizeof k);
    k.render[0] = k.max_render[0] = (int32_t)rw, k.render[1] = k.max_render[1] = (int32_t)rh;
    k.display[0] = k.upscaler_target[0] = k.rect_size[0] = (int32_t)w;
    k.display[1] = k.upscaler_target[1] = k.rect_size[1] = (int32_t)h;
    k.display_rcp[0] = 1.0f / (float)w, k.display_rcp[1] = 1.0f / (float)h;
    float n = fminf(e->near_z, e->far_z), f = fmaxf(e->near_z, e->far_z), aspect = (float)rw / (float)rh;
    float q = f / (n - f), cot = 1.0f / tanf(0.5f * e->fov_y);
    k.near_z = e->near_z, k.far_z = e->far_z;
    k.device_to_view[0] = -q, k.device_to_view[1] = q * n, k.device_to_view[2] = aspect / cot, k.device_to_view[3] = 1.0f / cot;
    k.reset = reset;
    k.dt = e->dt_ms;
    k.distortion_size[0] = k.distortion_size[1] = 1;
    k.of_scale[0] = 1.0f / (float)w, k.of_scale[1] = 1.0f / (float)h, k.of_block = OF_BLOCK;
    k.transfer = e->pq ? 1u : 0u;
    k.min_max_lum[0] = 0.0f, k.min_max_lum[1] = e->pq ? e->peak_nits : 0.0f;
    k.tan_half_fov = tanf(atanf(tanf(0.5f * e->fov_y) * aspect));   // tan of half the horizontal field of view
    k.jitter[0] = e->jitter[0], k.jitter[1] = e->jitter[1];
    k.mv_scale[0] = 1.0f / (float)rw, k.mv_scale[1] = 1.0f / (float)rh;   // render pixels to UV
    k.debug_bar[0] = 0.0f, k.debug_bar[1] = 1.0f, k.debug_bar[2] = 1.0f;
    k.hudless = 1;   // the HUD-less picture is given: the interpolation source, the HUD put back from the picture
    VkDescriptorBufferInfo cbk = upload(&k, sizeof k);
    uint32_t rgx = (rw + 7) / 8, rgy = (rh + 7) / 8, dgx = (w + 7) / 8, dgy = (h + 7) / 8;
    VkImageView dmv = X.view[I_DMV], ddepth = X.view[I_DDEPTH], prev = X.view[I_PREV_SRC], dist = X.view[I_DEF_DIST];

    timing(cb, 0);
    // prepare: the dilated depth and motion vectors, the previous frame's nearest depth
    clear_u(cb, I_RECON_PREV, 0x3f800000u);   // 1.0 (far) as the bits the atomics compare
    barrier(cb);
    {
        const Bind b[] = {IMG(0, e->mv), IMG(1, e->depth), IMG(2, X.view[I_RECON_PREV]), IMG(3, dmv), IMG(4, ddepth),
                          BUF(5, cbk)};
        if (!run(cb, FG_FI_RECONSTRUCT_AND_DILATE, b, N(b), rgx, rgy, 1)) goto fail;
    }
    if (!optical_flow(cb, e, reset)) goto fail;
    timing(cb, 1);
    {
        const Bind b[] = {IMG(0, X.view[I_OF_SCD]), IMG(1, X.view[I_GMV_X]), IMG(2, X.view[I_GMV_Y]),
                          IMG(3, X.view[I_OFMV_X]), IMG(4, X.view[I_OFMV_Y]), IMG(5, X.view[I_DISOCC]),
                          BUF(6, counters()), BUF(7, cbk)};
        if (!run(cb, FG_FI_SETUP, b, N(b), rgx, rgy, 1)) goto fail;
    }
    if (!reset) {
        // ours: the game's vectors checked against the two pictures (ffxfg_check.comp), for the two passes that place
        // this frame's pixels in the frame between; the vector field also gets the depth with the pixels given another
        // vector pushed back
        {
            CheckConsts ck = {.k = {CHECK_EPS, CHECK_OFF, 1.0f / CHECK_BACK, CHECK_FROM},
                              .d = {k.device_to_view[0], 0.0f, (float)OF_BLOCK}};
            const Bind b[] = {IMG(0, dmv), IMG(1, ddepth), IMG(2, e->hudless_view), IMG(3, prev),
                              IMG(4, X.view[I_DMV_CHECKED]), IMG(5, X.view[I_DDEPTH_FIELD]),
                              BUF(6, upload(&ck, sizeof ck)), IMG(8, X.view[I_OF_VECTOR])};
            if (!run(cb, FG_CHECK, b, N(b), rgx, rgy, 1)) goto fail;
            dmv = X.view[I_DMV_CHECKED];
        }
        clear_u(cb, I_RECON_INTERP, 0x3f800000u);
        barrier(cb);
        {
            const Bind b[] = {IMG(0, dmv), IMG(1, ddepth), IMG(2, e->hudless_view), IMG(3, dist),
                              IMG(4, X.view[I_RECON_INTERP]), BUF(5, cbk)};
            if (!run(cb, FG_FI_RECONSTRUCT_PREV_DEPTH, b, N(b), rgx, rgy, 1)) goto fail;
        }
        {
            const Bind b[] = {IMG(0, dmv), IMG(1, X.view[I_DDEPTH_FIELD]), IMG(2, prev), IMG(3, e->hudless_view),
                              IMG(4, dist), IMG(5, X.view[I_GMV_X]), IMG(6, X.view[I_GMV_Y]), BUF(7, cbk)};
            if (!run(cb, FG_FI_GAME_MV_FIELD, b, N(b), rgx, rgy, 1)) goto fail;
        }
        {
            uint32_t g[2], ng, nm;
            spd_setup(rw, rh, -1, g, &ng, &nm);
            PyrConsts pk = {.mips = nm, .groups = ng};
            Bind b[PYR_VIEWS + 5];
            int nb = 0;
            b[nb++] = (Bind)IMG(0, X.view[I_GMV_X]), b[nb++] = (Bind)IMG(1, X.view[I_GMV_Y]);
            b[nb++] = (Bind)BUF(2, counters());
            for (int m = 0; m < PYR_VIEWS; m++) b[nb++] = (Bind)IMG((uint16_t)(3 + m), X.pyr[m]);
            b[nb++] = (Bind)BUF(16, cbk), b[nb++] = (Bind)BUF(17, upload(&pk, sizeof pk));
            if (!run(cb, FG_FI_GAME_PYRAMID, b, nb, g[0], g[1], 1)) goto fail;
        }
        {
            const Bind b[] = {IMG(0, X.view[I_OF_VECTOR]), IMG(2, ddepth), IMG(3, prev), IMG(4, e->hudless_view),
                              IMG(6, X.view[I_OFMV_X]), IMG(7, X.view[I_OFMV_Y]), BUF(8, cbk)};
            if (!run(cb, FG_FI_OF_VECTOR_FIELD, b, N(b), (w / OF_BLOCK + 7) / 8, (h / OF_BLOCK + 7) / 8, 1)) goto fail;
        }
        {
            const Bind b[] = {IMG(0, X.view[I_GMV_X]), IMG(1, X.view[I_GMV_Y]), IMG(2, X.view[I_RECON_PREV]),
                              IMG(3, ddepth), IMG(4, X.view[I_RECON_INTERP]), IMG(5, X.view[I_PYRAMID]), IMG(6, dist),
                              IMG(7, X.view[I_DISOCC]), BUF(8, cbk)};
            if (!run(cb, FG_FI_DISOCCLUSION, b, N(b), rgx, rgy, 1)) goto fail;
        }
    }
    {
        const Bind b[] = {IMG(0, X.view[I_GMV_X]), IMG(1, X.view[I_GMV_Y]), IMG(2, X.view[I_OFMV_X]),
                          IMG(3, X.view[I_OFMV_Y]), IMG(4, prev), IMG(5, e->hudless_view), IMG(6, X.view[I_DISOCC]),
                          IMG(7, X.view[I_PYRAMID]), BUF(8, counters()), IMG(9, e->out_view), BUF(10, cbk)};
        if (!run(cb, FG_FI_INTERPOLATION, b, N(b), dgx, dgy, 1)) goto fail;
    }
    {
        uint32_t g[2], ng, nm;
        spd_setup(w, h, -1, g, &ng, &nm);
        PyrConsts pk = {.mips = nm, .groups = ng};
        Bind b[PYR_VIEWS + 4];
        int nb = 0;
        b[nb++] = (Bind)IMG(0, e->out_view), b[nb++] = (Bind)BUF(1, counters());
        for (int m = 0; m < PYR_VIEWS; m++) b[nb++] = (Bind)IMG((uint16_t)(2 + m), X.pyr[m]);
        b[nb++] = (Bind)BUF(15, cbk), b[nb++] = (Bind)BUF(16, upload(&pk, sizeof pk));
        if (!run(cb, FG_FI_INPAINTING_PYRAMID, b, nb, g[0], g[1], 1)) goto fail;
    }
    {
        const Bind b[] = {IMG(0, X.view[I_OF_SCD]), IMG(1, X.view[I_PYRAMID]), IMG(2, e->back_view),
                          IMG(3, e->hudless_view), IMG(7, e->out_view), BUF(8, cbk)};
        if (!run(cb, FG_FI_INPAINTING, b, N(b), dgx, dgy, 1)) goto fail;
    }
    // the HUD-less picture, the next frame's previous interpolation source
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {w, h, 1}};
    vkCmdCopyImage(cb, e->hudless, VK_IMAGE_LAYOUT_GENERAL, X.img[I_PREV_SRC], VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    barrier(cb);
    timing(cb, 2);
    return 1;
fail:
    X.failed = 1;
    return 0;
}
