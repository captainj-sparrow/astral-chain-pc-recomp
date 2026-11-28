// Apple MetalFX Temporal Scaler (vk_metalfx.mm): native MetalFX temporal upscaling and Native AA on macOS via
// MoltenVK's VK_EXT_metal_objects interop, behind AC_UPSCALING=metalfx (astralchain.ini upscaling = metalfx, or auto
// on macOS). vk_dlaa.c finds the scene, jitters it, makes its motion vectors and calls vk_metalfx_evaluate where the
// upscaler runs; this file stages the scene color and depth into Metal-exportable images, splits the Vulkan command
// buffer with a shared MTLSharedEvent timeline semaphore, and encodes MTLFXTemporalScaler directly onto MoltenVK's
// MTLCommandQueue.
#pragma once
#include "vk.h"

typedef struct {
    VkImage image;
    VkFormat format;
    uint32_t width, height;   // the image's own extent (level 0)
    uint32_t level, layer;    // the subresource the scene is in
} MetalFxImage;

typedef struct {
    MetalFxImage color, depth;   // the scene (light target or composite) and its depth (D32F or D32F_S8)
    VkImage mv, out;             // exportable RG16F motion (w x h) and RGBA16F output (ow x oh), level 0 layer 0
    uint32_t w, h;               // the scene's size (the input)
    uint32_t ow, oh;             // the output's size, 0: the scene's (Native AA)
    float jx, jy;                // this frame's jitter, pixels (window x right, y down)
    int reset;                   // a new history (cut, size change, gap)
    int hdr;                     // color is linear scene light (auto exposure), else 0..1 (tonemapped)
} MetalFxEval;

#ifdef __APPLE__
#ifdef __cplusplus
extern "C" {
#endif

void vk_metalfx_init(void);        // vk_dlaa_init, the device up: exports MTLDevice/MTLCommandQueue, checks MetalFX
int vk_metalfx_available(void);    // MetalFX temporal scaling is supported and has not failed in this run
void vk_metalfx_warm(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int hdr);
// Evaluates MetalFX. Records staging copies into *cb_io, splits the submission with a shared MTLSharedEvent so
// MetalFX runs strictly between the pre- and post-upscaler Vulkan commands, and updates *cb_io to the new command
// buffer. 0: MetalFX failed and is off for the rest of the run.
int vk_metalfx_evaluate(VkCommandBuffer *cb_io, const MetalFxEval *e);
void vk_metalfx_sweep(void);       // vk_dlaa_frame: frees retired scalers whose GPU work has completed
void vk_metalfx_trim(void);        // vk_dlaa_frame: retires all cached scalers after MetalFX has been idle
void vk_metalfx_release(void);     // vk_dlaa_release, the device idle: frees everything

#ifdef __cplusplus
}
#endif
#else
static inline void vk_metalfx_init(void) {}
static inline int vk_metalfx_available(void) { return 0; }
static inline void vk_metalfx_warm(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int hdr) {
    (void)w; (void)h; (void)ow; (void)oh; (void)hdr;
}
static inline int vk_metalfx_evaluate(VkCommandBuffer *cb_io, const MetalFxEval *e) {
    (void)cb_io; (void)e;
    return 0;
}
static inline void vk_metalfx_sweep(void) {}
static inline void vk_metalfx_trim(void) {}
static inline void vk_metalfx_release(void) {}
#endif
