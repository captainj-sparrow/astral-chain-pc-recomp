// The op-stream replay of the Vulkan back end. vk_exec.c walks the command
// handles of a submission, keeps render targets, passes, clears and fixed-function state; draws, binds and dispatches
// (vk_exec_draw.c) and copies, counters, events and conditional rendering (vk_exec_xfer.c) plug in through the two
// hooks at the bottom.
// Runs on the submitting thread under vk_sync_submit's lock: no locking inside.
// The op loop follows MacNativeGames nvn_mtl.m (MIT, Copyright (c) 2026 jhaizhou) and the pass handling LAS Recomp
// (MIT, Copyright (c) 2026 Peiikai); vk_exec.c names the sources.
#pragma once
#include <stdint.h>

#include "nvn_api.h"
#include "vk.h"

// Dynamic state the next draw must record (VkExec.dirty). vk_exec_record sets them all at the start of each command
// buffer; the state ops set theirs; vk_exec_draw.c clears what it recorded.
enum {
    VK_EXEC_DIRTY_VIEWPORT = 1 << 0,     // viewports or depth range
    VK_EXEC_DIRTY_SCISSOR = 1 << 1,
    VK_EXEC_DIRTY_DEPTH_BIAS = 1 << 2,   // polygon offset
    VK_EXEC_DIRTY_BLEND_CONST = 1 << 3,
    VK_EXEC_DIRTY_STENCIL = 1 << 4,      // compare and write masks, references
    VK_EXEC_DIRTY_ALL = (1 << 5) - 1,
};

#define VK_EXEC_CLEARS 16   // whole-target clears kept for the next pass (VkExec.pclear)

// One render target of OP_SET_RT, resolved to its vk_texture.c image for the current command buffer.
typedef struct {
    const tex_t *tex;                    // payload copy (lives for the run); NULL: no target in this slot
    VkImage image;                       // VK_NULL_HANDLE: no target or not renderable (logged once)
    VkFormat format;                     // view format: the view's when it reinterprets a same-size format
    VkImageViewType view_type;           // 2D, or 2D_ARRAY for more than one layer
    VkImageAspectFlags aspect;
    uint32_t level, base_layer, layers;  // subresource of the view; of a 3D image: depth planes
    uint8_t is3d;                        // a 3D image (colour only), viewed through 2D views of its slices
    uint8_t blocks;                      // a BC image viewed as one texel per block (VkTexImage.block_views)
    uint32_t width, height, samples;     // extent of that level; with blocks, in blocks
    int32_t nvn_format;                  // NVNformat of the view (VkPipeState.color_fmt / depth_fmt)
    const VkTexImage *img;               // the image (fb_usage, fb_learned)
    float scale;                         // VkTexImage.scale
} VkExecTarget;

// The replay state. One instance lives for the run: NVN state is queue state and carries over from one submission to
// the next, as in MacNativeGames' and LAS Recomp's replays. The "per command buffer" fields are reset by vk_exec_record.
typedef struct VkExec {
    // ---- per command buffer
    VkCommandBuffer cb;                  // being recorded
    uint64_t fence;                      // timeline value the submission signals (vk_completion_add)
    int recorded;                        // Vulkan commands recorded so far; every handler adds the ones it records
    int in_pass;                         // inside vkCmdBeginRendering
    uint32_t dirty;                      // VK_EXEC_DIRTY_*

    // ---- render targets (OP_SET_RT) and the current pass
    p_set_rt rt;                         // last OP_SET_RT payload (a change ends the open pass); rt.n 0, no depth: none
    int rt_resolved;                   // color[] and depth hold this command buffer's images for rt
    VkExecTarget color[8], depth;        // resolved targets
    uint32_t pass_w, pass_h, pass_layers;            // render area and layer count of the open (or next) pass
    float pass_scale;                                // its targets' VkTexImage.scale: game viewports and scissors
                                                     // are multiplied by it
    int pass_noatt;                                   // the pass has no attachment: its size follows the draws (vk_exec_noatt_dims)
    VkFormat pass_color[8], pass_depth, pass_stencil; // attachment formats of the pass (VkPipelineRenderingCreateInfo);
                                                      // VK_FORMAT_UNDEFINED: none
    uint32_t pass_ncolor, pass_samples;               // colorAttachmentCount and rasterization samples of the pass

    // ---- fixed-function state (vk_exec.c keeps it; vk_exec_draw.c records it before draws)
    float vp[16][6];                     // OP_VIEWPORTS: x, y, w, h, near, far
    int has_vp;                          // a viewport was set; else the render area
    p_scissor scissor;                   // OP_SCISSOR (scissor 0)
    int has_scissor;                     // a scissor was set; else unbounded
    float depth_range[2];                // OP_DEPTH_RANGE: overrides the viewports' near and far
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];   // OP_STENCIL per face (0 front, 1 back): compare mask, write mask, ref
    float point_size;                    // OP_BIND_STATE kind 6: NVN point size for the support buffer; 0 = never set = 1
    p_poly_offset poly_offset;           // OP_POLY_OFFSET: depth bias factor, units, clamp
    p_prim_restart prim_restart;         // OP_PRIM_RESTART: applied to indexed strip and fan draws (op_draw_idx)
    float blend_color[4];                // OP_BLEND_COLOR
    int64_t copy_stride;                 // OP_COPY_STRIDE: row stride of buffer <-> texture copies (vk_exec_xfer.c)
    const dpool_t *tex_pool, *samp_pool; // OP_SET_TEX_POOL, OP_SET_SAMP_POOL
    // Pipeline inputs. vk_exec.c writes depth_clamp (OP_DEPTH_CLAMP), raster_discard (OP_RASTER_DISCARD) and, at pass
    // begin, color_fmt, depth_fmt and samples; vk_exec_draw.c owns every other field (shaders, prim, y_flip, state
    // objects, vertex input).
    VkPipeState pipe;

    // ---- vk_exec_draw.c slots (draws, binds, dispatch): left empty by vk_exec.c
    const prog_t *prog;                  // OP_BIND_PROGRAM: the last bound program, or the one left bound for a
                                         // stage after a NULL bind unbound the others; NULL: no stage bound
    const prog_t *prog_stage[6];         // the program bound for each NVN shader stage
    uint32_t prog_unbound;               // stage bits a BindProgram(NULL, stages) unbound: no fallback to prog
    int32_t prog_stages;
    NVNbufferRange vbuf[16];             // OP_BIND_VBUF
    p_bind_buf ubo[6][18], ssbo[6][16];  // OP_BIND_UBO / OP_BIND_SSBO per stage and index
    uint64_t tex[6][128], samp[6][128], img[6][32];   // OP_BIND_TEX / OP_BIND_SAMP / OP_BIND_IMG handles per stage

    // ---- vk_exec_xfer.c slots (render enable, conditional rendering): left empty by vk_exec.c
    int render_disabled;                 // OP_RENDER_ENABLE false or a failed OP_RENDER_COND: clears and draws skip
    p_render_cond render_cond;           // last OP_RENDER_COND
    int has_render_cond;

    // ---- internal to vk_exec.c
    void *views;                         // image views created for this command buffer (freed at its fence)
    int nviews, cap_views;
    int barrier_at;                      // recorded right after the last barrier, full or narrow (-1: none yet)
    int bar_prev;                        // VK_BAR_* of that barrier (VK_BAR_N: none yet in this submission)
    // Whole-target clears recorded outside a pass (vk_exec.c clear_color, clear_ds) wait here: the next pass that
    // attaches the subresource clears it with its load op, and anything else recorded first records them as image
    // clears (vk_exec_clears_flush). Empty at the end of every submission.
    struct VkExecClear {
        VkImage image;
        const tex_t *tex;
        VkImageAspectFlags aspect;       // colour, or the depth and stencil aspects cleared
        uint32_t level, base_layer, layers;
        VkClearValue v;
    } pclear[VK_EXEC_CLEARS];
    int npclear;
    uint32_t rt_serial, pass_rt;         // bumped when OP_SET_RT changes the targets; rt_serial of the last pass begun

    // The pass chain (vk_exec.c "the pass chain"): what was recorded since the last barrier, so that a pass begin can
    // take a barrier of attachment writes only, or none.
    uint32_t sync_epoch;                 // bumped by every barrier recorded; the stamps of VkTexImage compare to it
    int chain_at;                        // `recorded` right after the last barrier or pass end: a pass begin that finds
                                         // `recorded` there follows nothing but passes since the last barrier
    uint8_t chain_other;                 // a draw since the last barrier wrote a storage buffer or image: the next
                                         // barrier is a full one
    uint8_t bar_narrow;                  // the last barrier recorded was a narrow one
    uint8_t pass_how;                    // VK_PASS_*: the barrier the open pass began behind
    uint8_t pass_force;                  // the next pass begin takes a barrier (a restart for a hazard)
    uint8_t pass_hinted, hint_hit;       // the open pass took its barrier for its hint; a draw confirmed the hint
    uint32_t hint_epoch;                 // sync_epoch before that barrier
    uint64_t rt_hash, last_rt_hash;      // hash of rt; of the targets of the pass before the last target change
    uint64_t pass_key;                   // names a pass by its targets and those of the pass before (the hints)
    const void *sy_vs, *sy_fs;           // vk_exec_draw.c chain_draw's memo: the shaders, texture bind serial and pass
    uint32_t sy_serial, sy_pass;         // serial of the last scan

    // Feedback loops (vk.h VkCtx.feedback, vk_exec_draw.c feedback_prepare). Learned targets are attached in
    // ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT; they stay in it across consecutive passes and go back to GENERAL at the next
    // full barrier that is not a pass begin.
    VkImage fbl_img[9];                  // images in the feedback layout now
    VkImageAspectFlags fbl_aspect[9];
    int nfbl;
    uint32_t fbl_mask;                   // attachments of the open pass in that layout: bit i colour i, bit 8 depth
    uint32_t fb_dirty;                   // attachments of the open pass written by a draw or clear since the pass began
                                         // or the last in-pass barrier
    const void *fb_fs;                   // feedback_scan's memo: fragment shader, texture bind serial, pass serial
    uint32_t fb_serial, fb_pass, fb_sampled;
    uint32_t tex_serial, pass_serial;    // bumped by every texture or image bind and pass begin
} VkExec;

// What asks for a barrier (vk_exec_barrier's src): a pass begin, a whole-target clear outside a pass, a dispatch,
// a copy or fill (vk_exec_xfer.c), the game's own nvnCommandBufferBarrier, an inline uniform update, texture work at
// the replay's position (vk_exec_inline_begin), a host pass (vk_ao.c, vk_dlaa.c), the end of the submission.
// AC_TRACE=perf counts the barriers by source (vk_exec.c bar_log).
enum { VK_BAR_PASS, VK_BAR_CLEAR, VK_BAR_DISPATCH, VK_BAR_COPY, VK_BAR_GAME, VK_BAR_UPDATE, VK_BAR_INLINE, VK_BAR_HOST,
       VK_BAR_END, VK_BAR_N };
// What ends an open pass (vk_exec_pass_end_why): a barrier of source VK_BAR_*, or one of these.
enum { VK_END_TARGETS = VK_BAR_N,        // OP_SET_RT with other targets
       VK_END_AREA,                      // a pass without attachments whose render area grows
       VK_END_FEEDBACK,                  // a feedback-loop restart (vk_exec_draw.c feedback_prepare)
       VK_END_CHAIN,                     // a restart behind a barrier the pass chain left out (vk_exec_chain_draw)
       VK_END_OTHER, VK_END_N };
// The barrier a pass began behind (VkExec.pass_how): all commands and all memory, attachment writes only, none.
enum { VK_PASS_FULL, VK_PASS_NARROW, VK_PASS_NONE };
// Begins dynamic rendering with the current targets unless a pass is open. A barrier precedes it: a full one, or
// what the pass chain (vk_exec.c) finds enough after passes that only wrote attachments. Call before a draw or an
// attachment clear.
void vk_exec_pass_begin(VkExec *ex);
// The pass chain (vk_exec.c). vk_exec_chain_on: 0 off; bit 0: the storage writes of draws are tracked (AC_BAR_NARROW or
// AC_BAR_ELIDE is not 0); bit 1: the images draws sample too (AC_BAR_ELIDE is not 0).
// vk_exec_chain_draw: a draw of the open pass, before anything of it is recorded. writes: its shaders write a storage
// buffer or image; unsynced: it samples an image a pass wrote since the last barrier. Restarts the pass behind the
// barrier the draw needs and returns 1 (the caller looks at the draw again), else 0.
int vk_exec_chain_on(void);
int vk_exec_chain_draw(VkExec *ex, int writes, int unsynced);
// A draw samples levels [l0, l1) and layers [a0, a1) of img: stamps them read since the last barrier; 1 when a pass
// had them attached, or memory they alias, since then.
int vk_exec_chain_sampled(VkExec *ex, const VkTexImage *img, uint32_t l0, uint32_t l1, uint32_t a0, uint32_t a1);
// Records the whole-target clears that wait for a pass (VkExec.pclear) as image clears behind one full barrier. Every
// vk_exec_barrier does it first; host work that orders itself with its own barriers calls it before reading or writing
// a game image (vk_contact.c).
void vk_exec_clears_flush(VkExec *ex);
// Ends the open pass (no-op without one) and marks its targets GPU-written. Call before copies and dispatches.
// why: VK_BAR_* or VK_END_*, counted under AC_TRACE=perf.
void vk_exec_pass_end_why(VkExec *ex, int why);
static inline void vk_exec_pass_end(VkExec *ex) { vk_exec_pass_end_why(ex, VK_END_OTHER); }
// Full memory barrier (all commands, all memory); ends the open pass first. Copies and dispatches order themselves with
// it. Skipped when nothing was recorded since the last one. src: VK_BAR_*.
void vk_exec_barrier(VkExec *ex, int src);
// vk_texture.c (AC_TEX_ALIAS): commands that must run at the replay's current position. vk_exec_inline_begin returns
// the command buffer being recorded after a full barrier (an open pass is ended; *reopen says so), or VK_NULL_HANDLE
// when this thread is not recording a replay. vk_exec_inline_end begins the pass again when it was open.
// vk_exec_inline_buffer: a device buffer for transfers, destroyed when the submission completes.
VkCommandBuffer vk_exec_inline_begin(int *reopen);
void vk_exec_inline_end(int reopen);
// vk_exec_draw.c: forget what the command buffer being recorded has bound (pipeline, sets, dynamic state, vertex
// buffers): the replay continues in a new one (vk_sync_split).
void vk_exec_draw_new_cb(void);
// vk_exec_draw.c, AC_TRACE=perf: logs the dynamic state calls and commands per submission over the last subs
// submissions (AC_DYN_DELTA), and starts the counts again.
void vk_exec_draw_perf(uint64_t subs);
VkBuffer vk_exec_inline_buffer(VkDeviceSize size);
// vk_exec.c: the stand-in a 3D BC render target is drawn through goes with its image (vk_exec_image_gone).
void vk_exec_bc_gone(VkImage image);
// vk_hdr.c: with AC_HDR=1, notes a draw into a presented window texture (vk_hdr_track: set by the first present).
extern int vk_hdr_track;
void vk_hdr_draw(VkExec *ex, const VkPipeInfo *info);
// Scissor 0 clipped to the render area of the open pass; extent 0: nothing to draw.
VkRect2D vk_exec_scissor(const VkExec *ex);
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h);   // pass size when no attachment is set
// 1: every target of the current pass (any 1: at least one) was rendered in the previous presented frame (per-frame
// targets); 0 otherwise, or when the pass has none.
int vk_exec_targets_recent(VkExec *ex, int any);
// The current pass's targets with their formats, extents and the frames they were last rendered in (diagnostics).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap);

// AC_TRACE=gfx: replay counters of the current frame, logged and reset at each present (vk_exec_frame_log).
// Draws the replay drops (vk_exec_draw.c prepare_draw and the draw ops), by reason; the difference between draws_seen
// and draws is their sum plus the draws skipped for a pending pipeline or shader (PERF_CNT_PIPE_SKIP, _SHADER_SKIP).
enum { VK_DROP_NOPROG, VK_DROP_DISABLED, VK_DROP_SCISSOR, VK_DROP_EMPTY, VK_DROP_OTHER, VK_DROP_N };
typedef struct {
    uint64_t passes, draws_seen, draws, clears, copies, dispatches, submits;
    uint64_t fb_draws, fb_barriers, fb_restarts, fb_layouts;   // feedback loops: draws sampling a written target of
                                                               // their pass, in-pass barriers, pass restarts, layout
                                                               // transitions
    uint64_t dropped[VK_DROP_N];         // VK_DROP_*: no program bound, render disabled, empty scissor, zero count,
                                         // other (no vertex stage, untranslatable shader, AC_SKIP_*, SMAA set aside)
} VkExecFrame;
extern VkExecFrame vk_exec_frame;
void vk_exec_frame_written(const tex_t *t);                    // vk_texture_gpu_written: t rendered or copied into
uint64_t vk_exec_recording_fence(void);                        // fence of the submission this thread records, else 0
void vk_exec_frame_log(const tex_t *presented, uint64_t frame);

// AC_DUMP=<frame> (vk_dump.c): between presents frame-1 and frame, every pass's targets and every dispatch's storage
// images are copied out and written as PNGs to AC_DUMP_DIR (default "dump"), with passes.txt (targets, pipelines,
// textures, samplers) and images.txt (per image value statistics). The hooks run only when vk_dump_armed is set.
extern int vk_dump_armed;
void vk_dump_init(void);                 // reads the environment; vk_exec_record's first call
void vk_dump_pass_end(VkExec *ex);       // after vkCmdEndRendering
void vk_dump_draw(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info);   // after a draw's binds
void vk_dump_dropped(VkExec *ex, const char *why);   // a draw the replay drops (vk_exec_draw.c drop_draw)
void vk_dump_after_draw(VkExec *ex, const char *kind, int64_t count, int64_t instances, int64_t first,
                        int64_t base_vertex, uint64_t ibuf);   // after the draw command (may end the pass)
void vk_dump_dispatch(VkExec *ex, const GfxShader *cs, VkPipeline p, const VkPipeInfo *info, const p_dispatch *d);
void vk_dump_copy(VkExec *ex, uint32_t op, const void *payload);   // before a copy or texture clear is replayed
void vk_dump_present(uint64_t frame);    // vk_be_present: waits for the dump frame's copies and closes the logs

// c0, the NVN driver constant bank of one stage (vk_exec_draw.c; AC_FIX_C0=0 binds zeros instead): bank is
// VK_C0_BYTES, stage an NVN stage (0 vertex, 1 fragment, 5 compute), ssbo and ubo the 16 bound slots of that stage.
#define VK_C0_BYTES 0x800u
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance);

// vk_dlaa.c: once an upscaler scaled the scene this frame, draws in passes wider than vk_dlaa_swap_w + 1 that sample the
// scene's composite vk_dlaa_swap_img read the upscaler's picture vk_dlaa_swap_view in its place, so the game's scaling
// draw scales nothing and copies the upscaled picture (vk_exec_draw.c tex_info). Cleared at each present.
extern VkImage vk_dlaa_swap_img;
extern VkImageView vk_dlaa_swap_view;
extern uint32_t vk_dlaa_swap_w;

// Handler hooks. vk_exec.c defines both WEAK: they stop with rt_fatal naming the op.
// vk_exec_draw.c: OP_DRAW, OP_DRAW_IDX, OP_DISPATCH, OP_BIND_* (program, state, vattr, vstream, vbuf, ubo, ssbo, tex,
// samp, img), OP_UPDATE_UB, OP_DRAW_IDX_IND, OP_DISPATCH_IND. payload: nvn_op_payload of the record.
void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload);
// The aspect a sampled view of img reads (vk_exec_draw.c get_view): stencil for a view in STENCIL depth-stencil mode.
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv);
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks);   // kept for the image's life
// vk_photo.c: OP_PHOTO, the photo mode's copy of the game's output into a host buffer and the album image;
// OP_PHOTO_FILL, the album image into the texture the game's photo passes drew into (photo.c).
void vk_photo_copy(VkExec *ex, const p_photo *p);
void vk_photo_fill(VkExec *ex, const p_photo_fill *p);
// vk_exec_xfer.c: OP_COPY_B2B, OP_COPY_B2T, OP_COPY_T2B, OP_COPY_T2T, OP_CLEAR_TEX, OP_SIGNAL_EVENT, OP_REPORT_COUNTER,
// OP_RESET_COUNTER, OP_RENDER_ENABLE, OP_RENDER_COND, OP_CLEAR_BUF.
void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload);
// vk_contact.c: a pass about to begin (its targets resolved, no pass open), and a dispatch or copy about to be recorded
void vk_contact_pass_begin(VkExec *ex);
void vk_contact_flush(VkExec *ex);
// vk_ao.c, ambient occlusion: a pass about to begin (its targets resolved, no pass open), a dispatch or copy about to be
// recorded, and a draw about to be recorded (the camera block)
void vk_ao_pass(VkExec *ex);
void vk_ao_flush(VkExec *ex);
void vk_ao_draw(VkExec *ex);
// vk_exec_xfer.c, samples-passed counting: a pass about to begin (occlusion query slots the completion thread read back
// go back to the pool here, on the GPU and outside the instance about to open), a draw about to be recorded in the open
// pass (begins an occlusion query when counting and none is active), and the open pass about to end (ends the active
// query).
void vk_exec_oq_resets(VkExec *ex);
void vk_exec_oq_draw(VkExec *ex);
void vk_exec_oq_pass_end(VkExec *ex);
// vk_gpuprof.c, AC_GPU_PASSES: GPU time per pass. kind: 0 start of a command buffer, 1 before a pass begins, 2 after
// it ends, 3 after a dispatch (arg: program serial), 4 after a copy (arg: op), 5 end of a command buffer, 6 after the
// ambient occlusion (vk_ao.c). Outside render passes only.
int vk_gpuprof_on(void);
void vk_gpuprof_mark(VkExec *ex, int kind, uint64_t arg);
void vk_gpuprof_flush(uint64_t fence);
