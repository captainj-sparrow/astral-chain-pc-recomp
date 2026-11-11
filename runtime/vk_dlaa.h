// The scene's temporal anti-aliasing (vk_dlaa.c) at the scene's own size, chosen by AC_ANTI_ALIASING (astralchain.ini
// anti_aliasing = smaa, dlaa, xess or fsr3): NVIDIA DLAA, Intel XeSS Native AA (vk_xess.c) or AMD FSR 3 Native AA
// (vk_fsr3.c) in place of the game's SMAA.
#pragma once
#include "vk.h"
#include "vk_exec.h"   // VkExec, VkExecTarget, p_bind_buf

int vk_dlaa_wanted(void);   // dlaa is on at start (the device may still lack DLSS)
// vk_device.c, after the instance exists (the NGX core is found through the loaded NVIDIA Vulkan driver): NGX's
// device extensions join the list (room for max names); vk_dlaa_init after the device exists brings NGX up and checks
// that DLSS is available.
void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_init(void);
void vk_dlaa_release(void);   // vk_device.c destroy_all, the device idle
// vk_exec.c vk_exec_pass_begin, once the pass's targets are resolved and before its barrier: follows the scene's passes
// and, at the first pass a quarter as wide as the scene's light target, runs the AA on that target. Each present ends
// the frame and picks the backend (vk_dlaa_frame).
void vk_dlaa_pass(VkExec *ex);
void vk_dlaa_frame(void);
// vk_exec_draw.c, each draw before its fixed state is recorded: notes the scene camera block (vk_dlaa_draw), and moves
// the viewport of the scene's draws by the frame's jitter (vk_dlaa_viewport, when the viewport is recorded).
void vk_dlaa_draw(VkExec *ex);
void vk_dlaa_viewport(VkExec *ex, VkViewport *v);
// vk_exec_draw.c, once a draw's fragment shader is known: 1 skips the draw (the game's SMAA while the AA runs).
int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs);
const char *vk_dlaa_running(void);   // "DLAA", "XeSS Native AA" or "FSR 3 Native AA" while it runs, else NULL
int vk_dlaa_choice_state(const char *aa);   // the menu: 0 runs or can, 1 after a restart, 2 not available here

// ---- per-object motion (vk_exec_draw.c "per-object motion", vk_mvprev.c), for either backend
int vk_dlaa_available(void);   // a temporal AA backend is chosen and up
// A draw of this frame's scene, not yet at the AA point: its depth is the scene depth (the armed pair's, or before
// arming the one the AA ran on last frame).
int vk_dlaa_scene_draw(const VkExec *ex);
void vk_dlaa_jitter(const VkExec *ex, float j[2]);   // the jitter this scene draw's viewport took, render pixels
// vk_dlaa.c mv_pass, after the camera's motion: draws the frame's listed objects again into the motion image mv
// (render size w x h) over the scene depth (target z), with last frame's blocks; only draws whose camera block (c3) is
// cam. 1: something was drawn. vk_mvobj_drop: the list is not used this frame (no camera motion, a cut, no AA point).
int vk_mvobj_execute(VkExec *ex, uint64_t cam, VkImageView mv, const VkExecTarget *z, uint32_t w, uint32_t h);
void vk_mvobj_drop(void);
// vk_mvprev.c: last frame's copies of an object's blocks. key: the draw's identity; banks: the position's (bit n: cn);
// ubo: the vertex stage's bound blocks from c3; max: the largest uniform range. prev[b] (bank b) and *have: the banks
// with last frame's copy; *changed: those whose bytes differ. 0: no last-frame object (new, or a block not snapshotted).
int vk_mvprev_on(void);
int vk_mvprev_blocks(uint64_t key, uint32_t banks, const p_bind_buf ubo[], uint64_t max, VkDescriptorBufferInfo prev[17],
                     uint32_t *have, uint32_t *changed);
// At each AA point: this frame's copies become last frame's for the next (vk_mvprev_swap). vk_mvprev_forget(0) after a
// frame without the AA point drops that frame's copies (the next is matched against the last frame the AA ran on);
// vk_mvprev_forget(1), a new history, drops last frame's too.
void vk_mvprev_swap(void);
void vk_mvprev_forget(int both);
