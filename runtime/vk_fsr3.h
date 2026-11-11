// AMD FSR 3 (vk_fsr3.c): AMD FidelityFX Super Resolution 3.1.4 as the scene's temporal anti-aliasing at its own size
// (Native AA), behind AC_ANTI_ALIASING=fsr3 (astralchain.ini anti_aliasing = fsr3). vk_dlaa.c finds the scene, jitters it, makes its motion
// vectors and calls vk_fsr3_evaluate where DLAA would run. Unlike DLSS and XeSS there is no library: AMD's passes are
// compiled into the exe (tools/ffx_sdk.py), so it runs on any GPU and OS the port does.
#pragma once
#include "vk.h"

typedef struct {
    VkImageView color;           // the light target (sampled): linear scene light
    VkImageView depth;           // the scene depth, depth aspect
    VkImageView mv;              // RG16F: last frame's window position minus this one's, render pixels, without jitter
    VkImageView out;             // RGBA16F of the same size, GENERAL: FSR 3 writes the picture here
    uint32_t w, h;               // the scene's size (input and output: Native AA)
    float jx, jy;                // this frame's jitter, pixels (window x right, y down)
    uint32_t phases;             // the jitter sequence's length
    float dt_ms;                 // time since the last evaluation
    float near_z, far_z, fov_y;  // the camera: near and far plane distance, vertical field of view (radians)
    int reset;                   // a new history (cut, size change, gap)
} Fsr3Eval;

int vk_fsr3_available(void);   // the pipelines are made (at the first call) and FSR 3 has not failed in this run
int vk_fsr3_failed(void);      // set-up or a frame failed: FSR 3 is off for the run (any thread; makes nothing)
// Records the passes into cb (the replay's command buffer, through the recorder; no render pass open; every image in
// GENERAL before and after). 0: FSR 3 failed and is off for the rest of the run.
int vk_fsr3_evaluate(VkCommandBuffer cb, const Fsr3Eval *e);
void vk_fsr3_release(void);    // the work images go (deferred); the next evaluation makes them again
void vk_fsr3_shutdown(void);   // vk_dlaa_release, the device idle: every object of FSR 3 goes at once
