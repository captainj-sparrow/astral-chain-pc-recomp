// DLAA (vk_dlaa.c): NVIDIA DLSS at the scene's own size, behind AC_DLAA (astralchain.ini dlaa = 1).
#pragma once
#include "vk.h"
#include "vk_exec.h"   // VkExec, VkExecTarget, p_bind_buf

int vk_dlaa_wanted(void);   // the setting is on (the device may still lack DLSS)
// vk_device.c: NGX's instance and device extensions join the lists (room for max names); vk_dlaa_init after the
// device exists brings NGX up and checks that DLSS is available.
void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_init(void);
// vk_exec.c vk_exec_pass_begin, once the pass's targets are resolved and before its barrier: arms the frame on the
// scene's light target or, at the first quarter-width pass after it, runs DLAA on that target; with a scene smaller than
// the output, runs DLSS on the composite at the begin of the game's scaling pass instead. Each present ends the frame
// (vk_dlaa_frame).
void vk_dlaa_pass(VkExec *ex);
void vk_dlaa_frame(void);
// vk_exec_draw.c, each draw before its fixed state is recorded: notes the scene camera block (vk_dlaa_draw), and moves
// the viewport of the scene's draws by the frame's jitter (vk_dlaa_viewport, when the viewport is recorded).
void vk_dlaa_draw(VkExec *ex);
void vk_dlaa_viewport(VkExec *ex, VkViewport *v);
// vk_exec_draw.c, once a draw's fragment shader is known: 1 skips the draw (the game's SMAA edge and weight passes while
// AC_SMAA is 0 or DLSS upscales).
int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs);

// ---- per-object motion (vk_exec_draw.c "per-object motion", vk_mvprev.c)
int vk_dlaa_available(void);   // DLSS is up and the dlaa setting on
// A draw of this frame's scene, not yet at the DLAA point: its depth is the scene depth (the armed pair's, or before
// arming the one DLAA ran on last frame).
int vk_dlaa_scene_draw(const VkExec *ex);
void vk_dlaa_jitter(float j[2]);   // the jitter this frame's scene draws take, render pixels
// vk_dlaa.c mv_pass, after the camera's motion: draws the frame's listed objects again into the motion image mv
// (render size w x h) over the scene depth (target z), with last frame's blocks. 1: something was drawn.
int vk_mvobj_execute(VkExec *ex, uint64_t cam, VkImageView mv, const VkExecTarget *z, uint32_t w, uint32_t h);
// vk_mvprev.c: last frame's copies of an object's blocks. key: the draw's identity; banks: the position's (bit n: cn);
// ubo: the vertex stage's bound blocks from c3; max: the largest uniform range. prev[b] (bank b) and *have: the banks
// with last frame's copy; *changed: those whose bytes differ. 0: no last-frame object (new, or a block not snapshotted).
int vk_mvprev_on(void);
int vk_mvprev_blocks(uint64_t key, uint32_t banks, const p_bind_buf ubo[], uint64_t max, VkDescriptorBufferInfo prev[17],
                     uint32_t *have, uint32_t *changed);
void vk_mvprev_swap(void);   // after DLSS ran: this frame's copies become last frame's
// vk_mvcheck.c (AC_MV_CHECK, test): mv_pass hands it the motion image after the camera pass and, after the object
// pass, the motion image again and the scene light target it ran on; it measures how well the vectors fit the picture.
void vk_mvcheck_camera(VkCommandBuffer cb, VkImage mv, uint32_t w, uint32_t h, float jx, float jy);
void vk_mvcheck_final(VkCommandBuffer cb, uint64_t fence, VkImage mv, VkImage cimg, VkFormat cfmt, uint32_t level,
                      uint32_t layer, uint32_t w, uint32_t h, float jx, float jy);
void vk_mvcheck_output(VkCommandBuffer cb, VkImage out, uint32_t ow, uint32_t oh);   // after DLSS on a check frame
const char *vk_mvobj_frame_masks(void);   // vk_exec_draw.c: the last motion list's draw count
