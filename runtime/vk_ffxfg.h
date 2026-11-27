// AMD FSR 3 frame generation (vk_ffxfg.c): AMD FidelityFX 3.1.4's optical flow and frame interpolation, one frame made
// between two of the game's, on any GPU the port runs on. vk_dlssg.c picks it as frame generation's method where NVIDIA
// DLSS Frame Generation is not offered or not wanted (frame_generation_method), and calls vk_ffxfg_evaluate where it
// would have called NGX: at the start of the first generated frame's blit, on the present thread.
#pragma once
#include "vk.h"

typedef struct {
    VkImage back, hudless, out;        // the frame with its HUD, the same without it, the frame made between: w x h
    VkImageView back_view, hudless_view, out_view;   // sampled (back, hudless); sampled and storage (out)
    VkFormat format;                   // of all three: the window texture's (RGBA8), or with HDR output A2B10G10R10
    uint32_t w, h;
    VkImageView depth, mv;             // the scene depth (depth aspect, 0 near .. 1 far) and the temporal upscaler's
    uint32_t rw, rh;                   // motion vectors (RG16F render pixels, last frame's position minus this one's)
    float jitter[2];                   // the scene's jitter, render pixels, as the upscalers get it
    float near_z, far_z, fov_y;        // the camera: near and far plane, vertical field of view (radians)
    float dt_ms;                       // the game's frame period
    int reset;                         // a new history: nothing to interpolate from (a cut, a size change, a gap)
    uint64_t frame_id;                 // the game frame's number (a gap is a new history too)
    int pq;                            // the pictures hold PQ code values (HDR10), else sRGB codes
    float peak_nits;                   // with pq: the display's peak luminance
} FfxFgEval;

// Whether this device runs AMD's passes (their formats as storage images, storage images without a format, subgroup
// arithmetic and quad operations in compute, 32 or 64 lanes): any thread once the device exists; the reason it does not
// is logged once.
int vk_ffxfg_supported(void);
int vk_ffxfg_failed(void);   // set-up or a frame failed: FSR 3 frame generation is off for the run
// Records the optical flow and the interpolation into cb (no render pass open, every image in GENERAL before and
// after) and writes the frame between into e->out. 0: it failed and is off for the rest of the run (logged).
int vk_ffxfg_evaluate(VkCommandBuffer cb, const FfxFgEval *e);
void vk_ffxfg_release(void);    // its images go (deferred); the next evaluation makes them again
void vk_ffxfg_shutdown(void);   // the device idle: every object goes at once
