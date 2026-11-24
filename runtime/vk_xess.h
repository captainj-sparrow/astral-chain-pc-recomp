// XeSS (vk_xess.c): Intel XeSS Super Resolution as the scene's temporal upscaler, at its own size (Native AA) or from a
// smaller scene, behind AC_UPSCALING=xess (astralchain.ini upscaling = xess). vk_dlaa.c finds the scene, jitters it,
// makes its motion vectors and calls vk_xess_evaluate where the upscaler runs; this file only talks to libxess.dll.
#pragma once
#include "vk.h"

int vk_xess_wanted(void);   // the setting is on at start (the device may still lack what XeSS needs)
int vk_xess_tried(void);    // the library was looked for in this run (it was chosen at start)
// vk_device.c, before the instance and device exist: XeSS's instance and device extensions join the lists (room for
// max names) once vk_xess_device_exts has checked that the device supports every extension and feature XeSS asks for. A
// device that lacks any of them keeps XeSS off and is made exactly as without it.
void vk_xess_instance_exts(const char **ext, uint32_t *n, uint32_t max);
int vk_xess_device_exts(VkInstance inst, VkPhysicalDevice pd, const VkExtensionProperties *have, uint32_t nhave,
                        const char **ext, uint32_t *n, uint32_t max);   // 1: XeSS can run, mutable descriptors needed
int vk_xess_available(void);   // loaded, and the device was made with what it needs
// One evaluation, recorded into cb (the replay's command buffer, no render pass open; vk_xess_evaluate drains the
// recorder thread before XeSS records into cb itself): color is the scene (w x h): its light (linear), or with ldr the
// tonemapped composite (0..1); depth its depth (depth aspect), mv the motion vectors (RG16F, render pixels, last frame's
// position minus this frame's, without the jitter), out an RGBA16F image of ow x oh (0: w x h) that receives the
// result. Every image is in GENERAL before and after. jx, jy: the jitter the scene was drawn with, in render pixels
// (window x right, y down). 0: XeSS failed and is off for the rest of the run.
typedef struct {
    VkImage image;
    VkImageView view;
    VkFormat format;
    uint32_t level, layer;
} XessImage;
typedef struct {
    XessImage color, depth, mv, out;
    uint32_t w, h;     // the scene (input)
    uint32_t ow, oh;   // the output, 0: the scene's size
    float jx, jy;
    int reset;
    int ldr;           // color is 0..1 (tonemapped), not linear scene light
} XessFrame;
int vk_xess_evaluate(VkCommandBuffer cb, const XessFrame *f);
// vk_dlaa.c vk_dlaa_warm, while the game starts (one thread, before the first evaluation): the context, its kernels and
// xessVKInit for input w x h, output ow x oh (0: w x h) and the colour kind, so the first evaluation finds them made;
// with cb (an open upload batch, submitted by the caller) also a throw-away evaluation of f. 1: done; 0: XeSS not
// available, or a step failed, which is logged and left to the first evaluation (XeSS stays on).
int vk_xess_warm(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int ldr, VkCommandBuffer cb, const XessFrame *f);
// The same without the evaluation, on a thread of its own that the first vk_xess_evaluate (or vk_xess_release) joins.
void vk_xess_warm_async(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int ldr);
void vk_xess_release(void);   // vk_device.c destroy_all, the device idle: the context goes before the device
