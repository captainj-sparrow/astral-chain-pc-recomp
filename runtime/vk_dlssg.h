// NVIDIA DLSS Frame Generation (vk_dlssg.c, AC_FRAME_GENERATION = off, auto, 2x, 3x, 4x): frames made by NGX between the
// game's own, from the frame's picture with its HUD, the HUD-less scene, the scene depth and the temporal upscaler's
// motion vectors, presented by the present thread between the game's presents.
#pragma once
#include "vk.h"

// What one game frame gives frame generation, taken on the replay thread when the game presents it (vk_backend.c
// present_image), while the frame's images still hold that frame: the temporal upscaler's part (vk_dlaa_fg_inputs)
// and the HUD-less scene (vk_hdr_present_source). Matrices are row-major for row vectors (v * M), NGX's convention,
// and hold no jitter.
typedef struct VkFgFrame {
    int on;                                // the frame has everything below and frame generation is to run
    int reset;                             // a new history: a camera cut, a new scene size, frames without the AA
    VkImage depth;                         // the scene depth (D32F_S8 or D32F), GENERAL, its depth aspect
    VkImageView depth_view;
    VkFormat depth_format;
    VkImage mv;                            // motion vectors, RG16F render pixels (last frame's position minus this one's)
    VkImageView mv_view;
    uint32_t rw, rh;                       // the size of the depth and the motion vectors (the scene's)
    VkImage hudless;                       // the scene the game's last pass copied into the window texture, before the
    VkFormat hudless_format;               // HUD (float, the window's size or smaller with its shape)
    uint32_t hw, hh;
    float jitter[2];                       // the scene's jitter, render pixels
    float view_to_clip[16], clip_to_view[16], clip_to_prev[16], prev_to_clip[16];
    float cam_pos[3], cam_up[3], cam_right[3], cam_fwd[3];
    float near_z, far_z, fov_y;
    uint64_t frame_id;                     // the game frame's number: one more than the last presented
} VkFgFrame;

void vk_dlssg_init(void);             // vk_device.c after vk_dlaa_init: what NGX offers here, logged once
int vk_dlssg_scene_wanted(void);      // vk_hdr.c: the HUD-less scene is to be found even without HDR output
// vk_hdr.c, on the replay thread, before the first draw into the window texture after the game's last scene pass (the
// render pass ended): the window texture copied into frame generation's HUD-less picture, recorded into cb. 1: done,
// and vk_dlssg_capture_image() holds it for this frame's present.
int vk_dlssg_capture(VkCommandBuffer cb, VkImage win, VkFormat fmt, uint32_t w, uint32_t h);
VkImage vk_dlssg_capture_image(void);
// AC_DLSSG_UI_ALPHA=1 (test): the window texture's alpha is 0 from the HUD-less copy on, and the HUD's draws blend their
// coverage into it (vk_hdr_hud_targets, vk_exec_draw.c), for frame generation's UI recomposition.
int vk_dlssg_ui_alpha(void);
int vk_dlssg_set_at_start(void);      // vk_device.c: the setting at start asked for it (its present queue)
// vk_backend.c present_image, on the replay thread: fills f->on and f->reset from the setting and this frame's inputs
// (the rest of *f is filled by vk_dlaa_fg_inputs and vk_hdr_present_source before).
void vk_dlssg_collect(VkFgFrame *f);
// The present thread. vk_dlssg_begin: the generated frames to show before the game's frame, 0 for none; with some,
// the next vk_present_blit of the window texture win records NGX's evaluations first and blits the first generated
// frame in its place (vk_present_pre_record). period_ns: the game's frame period (auto's multiplier). vk_dlssg_end
// after that blit: the generated frames made and shown from 1 on; 0: the blit showed the game's frame (a new history,
// or a call failed), still at frame generation's delayed time; -1: the blit recorded nothing (no swapchain image).
// vk_dlssg_image(k): generated frame k (1..n), or 0 the game's frame as it was then (a copy: the game may draw into
// its window texture again meanwhile).
int vk_dlssg_begin(const VkFgFrame *f, VkImage win, VkFormat fmt, uint32_t w, uint32_t h, uint64_t period_ns);
int vk_dlssg_end(void);
VkImage vk_dlssg_image(int k);
VkFormat vk_dlssg_format(void);       // of vk_dlssg_image's images: the window texture's, or HDR10 (A2B10G10R10)
// gfx_present.c: img is one of frame generation's HDR10 pictures (made with an HDR10 swapchain), which hold the
// present's PQ encode already
int vk_dlssg_encoded(VkImage img);
void vk_dlssg_release(void);          // vk_dlaa_release, the device idle, before NGX shuts down
// For the menu and the overlay: the multiplier in use ("2x" .. "6x") while generated frames were shown in the last
// presents, else NULL. vk_dlssg_choice_state: 0 runs or can, 1 after a restart (NGX was not loaded at start), 2 not
// here (no NVIDIA GPU or driver with it, its library missing, or it failed), 3 waits for a temporal upscaler.
const char *vk_dlssg_running(void);
int vk_dlssg_choice_state(void);
int vk_dlssg_max_multiplier(void);    // 2 .. 6, from NGX's MultiFrameCountMax; 0 before NGX answered or without it

// vk_dlaa.c, for vk_dlssg.c: NGX's frame generation feature, through the core vk_dlaa.c loaded and initialised.
typedef struct VkFgEval {
    VkImage back, hudless, out;            // the frame with its HUD, the same without it, the generated frame: w x h
    VkImageView back_view, hudless_view, out_view;
    VkFormat format;                       // of all three
    uint32_t w, h;
    const VkFgFrame *f;
    uint32_t count, index;                 // frames generated per game frame, and this one (1..count)
    int reset;
    VkImage ui_alpha;                      // AC_DLSSG_UI_ALPHA: the HUD's coverage (R8, w x h), or VK_NULL_HANDLE
    VkImageView ui_alpha_view;
    int hdr;                               // the pictures are HDR10 (PQ code values): DLSSG.ColorBuffersHDR
    VkBuffer no_interp;                    // DLSSG.OutputDisableInterpolation: NGX writes 1 to its first byte for a set
    uint32_t no_interp_bytes;              // not to show; VK_NULL_HANDLE for none
} VkFgEval;
int vk_ngx_fg_caps(unsigned *max_count);   // 1: NGX is up and offers frame generation; *max_count its MultiFrameCountMax
// uir: UI recomposition (DLSSG.UserInterfaceRecompositionEnabled): each evaluation gives the HUD's alpha (ui_alpha)
int vk_ngx_fg_create(VkCommandBuffer cb, uint32_t w, uint32_t h, VkFormat fmt, uint32_t rw, uint32_t rh, int uir,
                     void **feat);
int vk_ngx_fg_evaluate(VkCommandBuffer cb, void *feat, const VkFgEval *e);   // NGX's result
void vk_ngx_fg_release(void *feat);
// The frame's temporal upscaler inputs, at the game's present before vk_dlaa_frame: 1 when the upscaler resolved this
// frame with camera motion (depth, motion vectors, camera and jitter filled), else 0.
int vk_dlaa_fg_inputs(VkFgFrame *f);
