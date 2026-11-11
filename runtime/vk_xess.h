// XeSS (vk_xess.c): Intel XeSS Super Resolution as the scene's temporal anti-aliasing at its own size (Native AA),
// behind AC_ANTI_ALIASING=xess (astralchain.ini anti_aliasing = xess). vk_dlaa.c finds the scene, jitters it, makes its
// motion vectors and calls vk_xess_evaluate where DLAA would run; this file only talks to libxess.dll.
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
// recorder thread before XeSS records into cb itself): color is
// the scene's light (linear), depth its depth (depth aspect), mv the motion vectors (RG16F, render pixels, last frame's
// position minus this frame's, without the jitter), out an RGBA16F image of the same size that receives the result.
// Every image is in GENERAL before and after. jx, jy: the jitter the scene was drawn with, in pixels (window x right,
// y down). 0: XeSS failed and is off for the rest of the run.
typedef struct {
    VkImage image;
    VkImageView view;
    VkFormat format;
    uint32_t level, layer;
} XessImage;
typedef struct {
    XessImage color, depth, mv, out;
    uint32_t w, h;
    float jx, jy;
    int reset;
} XessFrame;
int vk_xess_evaluate(VkCommandBuffer cb, const XessFrame *f);
void vk_xess_release(void);   // vk_device.c destroy_all, the device idle: the context goes before the device
