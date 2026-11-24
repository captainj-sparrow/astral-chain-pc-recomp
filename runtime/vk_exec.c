// The op-stream replay core of the Vulkan back end: walks the command handles of
// one submission into its command buffer, keeps render targets and fixed-function state, opens and closes dynamic
// rendering passes and records clears and barriers. Draws, binds and dispatches go to vk_exec_op_draw
// (vk_exec_draw.c); copies, counters, events and conditional rendering to vk_exec_op_xfer (vk_exec_xfer.c). Images
// stay in VK_IMAGE_LAYOUT_GENERAL (vk_texture.c); memory barriers order passes, clears and whatever those two record
// between them: full ones (all commands, all memory) everywhere but between passes that only wrote attachments,
// where "the pass chain" below records a narrower one or none.
//
// The op loop (END / BEGIN / JUMP / CALL, which ops are pure state, the state defaults, render targets kept until they
// change, clears skipped while rendering is disabled) follows MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:10588-10700, 7409-7460, 5449-5457
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou). Pass begin and end (GENERAL-layout attachments, LOAD/STORE, render area from the smallest attachment, a
// barrier before a pass), the scissor clip and attachment clears follow LAS Recomp
// runtime/nvn/vulkan/exec.c:855-1149, 3553-3778 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "platform/platform.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_dlaa.h"

#define WEAK __attribute__((weak))
#define CALL_DEPTH 8

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    VkImageAspectFlags aspect;
    uint32_t level, base, layers;
    VkImageView view;
} ViewEnt;

static VkExec X;
static int x_ready;
static _Thread_local int x_here;   // this thread is inside vk_exec_record (vk_exec_inline_begin)

// ---- AC_TRACE=gfx frame counters

VkExecFrame vk_exec_frame;
static struct { const uint8_t *mem; uint32_t w, h, fmt, n; } fw[64];   // textures written this frame
static int nfw;

uint64_t vk_exec_recording_fence(void) { return x_here ? X.fence : 0; }

void vk_exec_frame_written(const tex_t *t) {
    if (!(rt_trace & RT_TRACE_GFX) || !t) return;
    for (int i = 0; i < nfw; i++)
        if (fw[i].mem == t->mem) { fw[i].n++; return; }
    if (nfw == 64) return;
    fw[nfw].mem = t->mem, fw[nfw].w = t->d.width, fw[nfw].h = t->d.height, fw[nfw].fmt = (uint32_t)t->d.format;
    fw[nfw++].n = 1;
}

void vk_exec_frame_log(const tex_t *p, uint64_t frame) {
    if (!(rt_trace & RT_TRACE_GFX)) return;
    VkExecFrame *f = &vk_exec_frame;
    int hit = 0;
    char list[512] = "";
    size_t o = 0;
    for (int i = 0; i < nfw; i++) {
        if (p && fw[i].mem == p->mem) hit = 1;
        if (o < sizeof list - 48)
            o += (size_t)snprintf(list + o, sizeof list - o, " %p:%ux%u/%#x*%u", (const void *)fw[i].mem, fw[i].w,
                                  fw[i].h, fw[i].fmt, fw[i].n);
    }
    rt_log("frame %llu: submits %llu passes %llu draws %llu/%llu (dropped: no program %llu, disabled %llu, scissor "
           "%llu, empty %llu, other %llu) clears %llu copies %llu dispatches %llu; present %p "
           "%ux%u/%#x written %d; feedback draws %llu barriers %llu restarts %llu layouts %llu; targets%s",
           (unsigned long long)frame, (unsigned long long)f->submits,
           (unsigned long long)f->passes, (unsigned long long)f->draws, (unsigned long long)f->draws_seen,
           (unsigned long long)f->dropped[VK_DROP_NOPROG], (unsigned long long)f->dropped[VK_DROP_DISABLED],
           (unsigned long long)f->dropped[VK_DROP_SCISSOR], (unsigned long long)f->dropped[VK_DROP_EMPTY],
           (unsigned long long)f->dropped[VK_DROP_OTHER],
           (unsigned long long)f->clears, (unsigned long long)f->copies, (unsigned long long)f->dispatches,
           p ? (const void *)p->mem : NULL, p ? p->d.width : 0, p ? p->d.height : 0, p ? (uint32_t)p->d.format : 0,
           hit, (unsigned long long)f->fb_draws, (unsigned long long)f->fb_barriers, (unsigned long long)f->fb_restarts,
           (unsigned long long)f->fb_layouts, list);
    memset(f, 0, sizeof *f);
    nfw = 0;
}

WEAK void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

WEAK void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

enum { W_UNRENDERABLE, W_LEVEL, W_LAYER, W_VIEW_FORMAT, W_BLOCK_VIEW, W_BLOCK_LAYERS, W_CLEAR_TARGET, W_CLEAR_MASK, W_STENCIL_MASK, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

// NVN defaults of the replay state (nvn_mtl.m).
static void state_reset(VkExec *ex) {
    memset(ex, 0, sizeof *ex);
    ex->st_vmask[0] = ex->st_vmask[1] = ex->st_wmask[0] = ex->st_wmask[1] = 0xff;
    ex->depth_range[1] = 1.0f;
    vk_pipe_state_defaults(&ex->pipe);
    ex->barrier_at = -1;
    ex->bar_prev = VK_BAR_N;
}

// ---- render targets

static uint32_t level_extent(uint32_t v, uint32_t level) {
    v >>= level;
    return v ? v : 1;
}

// A view format other than the texture's is used when both map directly and have the same texel size (the image is
// created MUTABLE_FORMAT for colour formats). blocks: t is a BC texture with block_views, and the view an uncompressed
// format whose texel is one of its blocks.
static int view_format(const tex_t *t, const view_t *v, int blocks, VkFormat *out) {
    TexVkFormat a, b;
    int bw, bh, vbw, vbh;
    if (!tex_vk_format(t->d.format, &a) || a.vk != a.ideal || !tex_vk_format(v->format, &b) || b.vk != b.ideal ||
        b.depth || b.stencil)
        return 0;
    int bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), vbpb = nvn_fmt_bpb(v->format, &vbw, &vbh);
    if (!bpb || bpb != vbpb || (!blocks && (bw != 1 || bh != 1)) || vbw != 1 || vbh != 1) return 0;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, b.vk, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
    *out = b.vk;
    return 1;
}

// ---- 3D BC render targets
// A 3D BC texture has no block views (vk_texture.c create_tex, AC_BLOCK_VIEWS_3D): on RDNA2 cards such an image
// made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE took GPU writes past the end of its memory (issue #5). Draws
// into it through an uncompressed view of its blocks (agl's BC4 encoder fills 32x32x32 and 64x64x64 volumes and their
// levels at the logo) go to a stand-in: a 2D array of the blocks of one level in the view's format, one layer per
// depth plane. It starts with the level's content, so a pass that writes part of a plane keeps the rest, and each pass
// end copies the planes it drew into the level (a copy between a compressed and a size-compatible uncompressed format).
typedef struct { VkImage bc, img; VmaAllocation alloc; VkFormat fmt; uint32_t level, w, h, d; } BcStandIn;
#define BC_STANDINS 64   // one per level drawn: 6 volumes of up to 7 levels at the logo
static BcStandIn bcs[BC_STANDINS];
static int nbcs;
static PlatMutex bcs_lock = PLAT_MUTEX_INIT;   // the replay adds and reads; destroy_tex (any thread) removes

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, uint32_t layers) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
}

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                           VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// The stand-in of level level of a 3D BC image for draws through view format fmt, made on first use; 0 when none can
// be made (logged). Replay thread.
static int bc_standin(const VkTexImage *img, uint32_t level, VkFormat fmt, BcStandIn *out) {
    static int warned;
    plat_mutex_lock(&bcs_lock);
    int full = nbcs == BC_STANDINS;
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == img->image && bcs[i].level == level && bcs[i].fmt == fmt) {
            *out = bcs[i];
            plat_mutex_unlock(&bcs_lock);
            return 1;
        }
    plat_mutex_unlock(&bcs_lock);
    if (full) {
        if (!warned++) rt_log("vk: more than %d 3D BC render targets: the others are left out", BC_STANDINS);
        return 0;
    }
    uint32_t lw = level_extent(img->width, level), lh = level_extent(img->height, level);
    BcStandIn s = {img->image, VK_NULL_HANDLE, NULL, fmt, level, (lw + 3) / 4, (lh + 3) / 4,
                   level_extent(img->depth, level)};
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {s.w, s.h, 1}, .mipLevels = 1, .arrayLayers = s.d, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ci, &ai, &s.img, &s.alloc, NULL) != VK_SUCCESS) {
        if (!warned++) rt_log("vk: stand-in for a 3D BC render target could not be created: left out");
        return 0;
    }
    // In the upload batch, which goes to the queue before this part: GENERAL, then the texture's blocks.
    VkCommandBuffer cb = vk_upload_begin();
    image_barrier(cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, s.d);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s.d},
                      .extent = {lw, lh, s.d}};   // texels of the compressed source
    vkCmdCopyImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vk_upload_end();
    plat_mutex_lock(&bcs_lock);
    bcs[nbcs++] = s;
    plat_mutex_unlock(&bcs_lock);
    rt_log("vk: 3D BC render target %ux%ux%u level %u drawn through a %ux%u x %u stand-in (format %d)", img->width,
           img->height, img->depth, level, s.w, s.h, s.d, (int)fmt);
    *out = s;
    return 1;
}

// vk_exec_draw.c vk_exec_image_gone: the stand-in of a BC image that goes, freed once the GPU is done with it.
void vk_exec_bc_gone(VkImage image) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == image) {
            vk_defer_image(bcs[i].img, bcs[i].alloc);
            bcs[i--] = bcs[--nbcs];
        }
    plat_mutex_unlock(&bcs_lock);
}

// At a pass end: the planes the pass drew into a stand-in go to its texture.
static void bc_copy_back(VkExec *ex, const VkExecTarget *t) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++) {
        if (bcs[i].img != t->image) continue;
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->base_layer, t->layers},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, bcs[i].level, 0, 1},
                          .dstOffset = {0, 0, (int32_t)t->base_layer},
                          .extent = {bcs[i].w, bcs[i].h, t->layers}};   // texels of the uncompressed source
        vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, bcs[i].bc, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        ex->recorded++;
        break;
    }
    plat_mutex_unlock(&bcs_lock);
}

// A draw into a 3D BC image without block views through an uncompressed view of its blocks: the target is the
// image's stand-in (bc_standin). 1 when o is filled (target or left out), 0 for any other target.
static int bc_target(const tex_t *t, const view_t *v, int has_v, const VkTexImage *img, VkExecTarget *o) {
    int bw, bh;
    if (img->type != VK_IMAGE_TYPE_3D || img->block_views || !nvn_fmt_bpb(t->d.format, &bw, &bh) || bw != 4 || bh != 4)
        return 0;
    VkFormat fmt;
    BcStandIn s;
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return 1;
    }
    if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt) || !bc_standin(img, level, fmt, &s)) {
        warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                  t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
        return 1;
    }
    uint32_t base = 0, layers = s.d;
    if (v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < s.d ? s.d - base : 0);
    }
    if (base >= s.d || !layers) return 1;
    if (base + layers > s.d) layers = s.d - base;
    o->image = s.img;   // not img->image: view_of makes this command buffer's view of the stand-in
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = 0, o->base_layer = base, o->layers = layers;
    o->width = s.w, o->height = s.h, o->samples = 1;
    o->nvn_format = v->format;
    return 1;
}

static void resolve_target(const tex_t *t, const view_t *v, int has_v, int depth_slot, VkExecTarget *o) {
    memset(o, 0, sizeof *o);
    o->tex = t;
    if (!t) return;
    const VkTexImage *img = vk_texture_get(t);
    if (img && !depth_slot && bc_target(t, v, has_v, img, o)) return;   // a 3D BC image: its stand-in
    VkImageUsageFlags need = depth_slot ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    int is3d = img && img->type == VK_IMAGE_TYPE_3D && !depth_slot;   // slices drawn through 2D views (2D_ARRAY_COMPATIBLE)
    if (!img || !(img->usage & need) || (img->type != VK_IMAGE_TYPE_2D && !is3d)) {
        warn_once(W_UNRENDERABLE, "render target format %#x target %d (%ux%u) is not renderable here: left out",
                  t->d.format, t->d.target, t->d.width, t->d.height);
        return;
    }
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return;
    }
    // A 3D level's depth planes take the place of layers (Vulkan views them as the array layers of a 2D array view).
    uint32_t nlay = is3d ? level_extent(img->depth, level) : img->layers;
    uint32_t base = 0, layers = nlay;
    if (has_v && v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < nlay ? nlay - base : 0);
    }
    if (base >= nlay || !layers) {
        warn_once(W_LAYER, "render target layer %u of a %u-layer texture: left out", base, nlay);
        return;
    }
    if (base + layers > nlay) layers = nlay - base;
    VkFormat fmt = img->format;
    int32_t nvn_fmt = t->d.format;
    int blocks = !depth_slot && img->block_views;
    if (blocks) {   // a BC texture takes draws only through an uncompressed view of its blocks
        if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt)) {
            warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                      t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
            return;
        }
        nvn_fmt = v->format;
        if (layers > 1) {   // an uncompressed view of a BC image has one layer (no maintenance6 here)
            warn_once(W_BLOCK_LAYERS, "render target format %#x: block view of %u layers, first layer only", t->d.format,
                      layers);
            layers = 1;
        }
    } else if (!depth_slot && has_v && v->set_format && v->format != t->d.format) {
        int ok = view_format(t, v, 0, &fmt);
        // an image made with a format list (vk_texture.c target_compression) takes views of the listed formats only:
        // for another one it is made again without a list, before the pass draws into it
        if (ok && img->format_list && fmt != img->format && fmt != img->view_twin) {
            const VkTexImage *made = vk_texture_unlist(t);
            if (made) img = made;
            else ok = 0;
        }
        if (ok) nvn_fmt = v->format;
        else fmt = img->format, warn_once(W_VIEW_FORMAT, "render target view format %#x of a format %#x texture: "
                                          "texture format used", v->format, t->d.format);
    }
    o->image = img->image;
    o->img = img;
    o->scale = img->scale > 0 ? img->scale : 1.0f;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = depth_slot ? img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                           : VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = level, o->base_layer = base, o->layers = layers;
    o->is3d = (uint8_t)is3d;
    o->blocks = (uint8_t)blocks;
    o->width = level_extent(img->width, level), o->height = level_extent(img->height, level);
    if (blocks) o->width = (o->width + 3) / 4, o->height = (o->height + 3) / 4;   // the view's extent: 4x4 blocks
    o->samples = img->samples ? img->samples : 1;
    o->nvn_format = nvn_fmt;
    if (!o->aspect) o->image = VK_NULL_HANDLE;
}

/* The size of a pass without attachments: the far corner of viewport 0, else of the scissor, 1..16384, as
   MacNativeGames and nxnative size it. Astral Chain draws a fragment shader that only writes a storage buffer into a 1x1
   viewport; a pass sized by the viewport's width alone puts an offset viewport outside the scissor, and the cave rock
   loses its ambient light. */
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h) {
    double x1 = 0, y1 = 0;
    if (ex->has_vp) {
        x1 = ceil(fmax(ex->vp[0][0], ex->vp[0][0] + ex->vp[0][2]));
        y1 = ceil(fmax(ex->vp[0][1], ex->vp[0][1] + ex->vp[0][3]));
    }
    if ((x1 < 1 || y1 < 1) && ex->has_scissor) x1 = (double)ex->scissor.x + ex->scissor.w, y1 = (double)ex->scissor.y + ex->scissor.h;
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1;
    *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}

// Resolves ex->rt to this command buffer's images and derives the pass parameters and the pipeline's target formats.
static void resolve_rt(VkExec *ex) {
    if (ex->rt_resolved) return;
    ex->rt_resolved = 1;
    const p_set_rt *r = &ex->rt;
    int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n;
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 0;
    float scale = 1.0f;
    for (int i = 0; i < 8; i++) {
        VkExecTarget *t = &ex->color[i];
        resolve_target(i < n ? r->color[i] : NULL, &r->cview[i], i < n && r->has_cview[i], 0, t);
        ex->pass_color[i] = t->image ? t->format : VK_FORMAT_UNDEFINED;
        ex->pipe.color_fmt[i] = t->image ? t->nvn_format : 0;
        if (!t->image) continue;
        if (t->width < w) w = t->width;
        if (t->height < h) h = t->height;
        if (t->layers < layers) layers = t->layers;
        if (!samples) samples = t->samples;
        if (t->scale > scale) scale = t->scale;
    }
    VkExecTarget *d = &ex->depth;
    resolve_target(r->depth, &r->dview, r->has_dview, 1, d);
    ex->pass_depth = d->image && (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pass_stencil = d->image && (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pipe.depth_fmt = d->image ? d->nvn_format : 0;
    if (d->image) {
        if (d->width < w) w = d->width;
        if (d->height < h) h = d->height;
        if (d->layers < layers) layers = d->layers;
        if (!samples) samples = d->samples;
        if (d->scale > scale) scale = d->scale;
    }
    ex->pass_noatt = w == UINT32_MAX;
    if (ex->pass_noatt) {
        vk_exec_noatt_dims(ex, &w, &h);
        layers = 1;
        static int logged;
        if (logged < 8 && ex->has_vp && (ex->vp[0][0] != 0 || ex->vp[0][1] != 0 || ex->vp[0][2] != w || ex->vp[0][3] != h))
            logged++, rt_log("vk_exec: pass without attachments %ux%u for viewport %g,%g %gx%g", w, h, ex->vp[0][0],
                             ex->vp[0][1], ex->vp[0][2], ex->vp[0][3]);
    }
    ex->pass_w = w, ex->pass_h = h, ex->pass_layers = layers;
    ex->pass_scale = ex->pass_noatt ? 1.0f : scale;
    if (scale > 1.0f) {   // an enlarged shadow map (vk_texture.c shadow_scale_of) beside a target of the game's size
        static int mixed_logs;
        int mixed = d->image && d->scale < scale;
        for (int i = 0; i < 8; i++) mixed |= ex->color[i].image && ex->color[i].scale < scale;
        if (mixed && mixed_logs < 4)
            mixed_logs++, rt_log("vk_exec: a pass mixes an enlarged shadow map (scale %g) with targets of the game's size "
                                 "(%ux%u): its draws do not fit; AC_SHADOW_SCALE=1 avoids it", scale, w, h);
    }
    // A declared colour target can have no image (the game passes NULL, or resolve_target left it out). The pipeline
    // counts the colours up to the last format it has, and VkRenderingInfo must carry the same count: a pass that
    // declares four but leaves only target 0 renderable used to give the pipeline one attachment and the pass four,
    // which the validation layer rejects (VUID-vkCmdDraw-colorAttachmentCount-06179) and Intel's driver may fault on.
    int ncolor = 0;
    for (int i = 0; i < n; i++)
        if (ex->color[i].image) ncolor = i + 1;
    ex->pass_ncolor = (uint32_t)ncolor;
    ex->pass_samples = samples ? samples : 1;
    ex->pipe.samples = (uint8_t)ex->pass_samples;
}

// The attachment view of a target. A registered texture's view lives as long as its image (vk_exec_target_view); any
// other image's view is made for this command buffer and freed at its fence.
static VkImageView view_of(VkExec *ex, const VkExecTarget *t) {
    if (t->img && t->img->image == t->image)
        return vk_exec_target_view(t->img, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, t->blocks);
    ViewEnt *v = ex->views;
    for (int i = 0; i < ex->nviews; i++)
        if (v[i].image == t->image && v[i].format == t->format && v[i].type == t->view_type && v[i].aspect == t->aspect &&
            v[i].level == t->level && v[i].base == t->base_layer && v[i].layers == t->layers)
            return v[i].view;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = t->view_type, .format = t->format,
                                .subresourceRange = {t->aspect, t->level, 1, t->base_layer, t->layers}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    if (t->blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
        rt_fatal("vk: render target view (format %d, %ux%u) could not be created", (int)t->format, t->width, t->height);
    if (ex->nviews == ex->cap_views) {
        ex->cap_views = ex->cap_views ? ex->cap_views * 2 : 16;
        ex->views = realloc(ex->views, sizeof(ViewEnt) * (size_t)ex->cap_views);
        if (!ex->views) rt_fatal("vk: out of memory");
        v = ex->views;
    }
    v[ex->nviews++] = (ViewEnt){t->image, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, view};
    return view;
}

typedef struct { int n; VkImageView v[]; } ViewList;

static void free_views(void *arg) {   // completion thread, once the submission that used them is done
    ViewList *l = arg;
    VkDevice dev = vk_ctx()->device;
    for (int i = 0; i < l->n; i++) vkDestroyImageView(dev, l->v[i], NULL);
    free(l);
}

// ---- target history
// The presented frame (gfx_counters.frames + 1; 0: never) each target subresource was last rendered into, and the one
// before. prepare_draw lets a draw into a target the previous frame did not render (a one-shot draw: a volume or LUT
// generated once) wait for its shaders and pipeline instead of skipping it. Per subresource, not per image: a volume
// may be generated a few slices per frame. Direct-mapped: a slot another subresource took reads as not rendered, so a
// collision makes a draw wait rather than skip.
typedef struct { VkImage image; uint32_t level, layer; uint64_t last, prev; } TargetFrames;
static TargetFrames tfr[1024];

static TargetFrames *tfr_slot(const VkExecTarget *t) {
    uint64_t h = ((uint64_t)(uintptr_t)t->image ^ (uint64_t)t->level << 56 ^ (uint64_t)t->base_layer << 40) *
                 0x9E3779B97F4A7C15ull;
    return &tfr[h >> 54];
}

static int tfr_match(const TargetFrames *e, const VkExecTarget *t) {
    return e->image == t->image && e->level == t->level && e->layer == t->base_layer;
}

static void tfr_note(const VkExecTarget *t, uint64_t stamp) {
    TargetFrames *e = tfr_slot(t);
    if (!tfr_match(e, t)) *e = (TargetFrames){t->image, t->level, t->base_layer, 0, 0};
    if (e->last != stamp) e->prev = e->last, e->last = stamp;
}

static int tfr_recent(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return tfr_match(e, t) && (e->last == stamp ? e->prev : e->last) == stamp - 1;
}

int vk_exec_targets_recent(VkExec *ex, int any) {
    resolve_rt(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int n = 0, recent = 0;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        n++;
        if (tfr_recent(t, stamp)) recent++;
        else if (!any) return 0;
    }
    return any ? recent > 0 : n > 0;
}

// One line naming each target of the current pass: format, extent, layer and the two frames it was last rendered in
// (prepare_draw's pipeline wait log).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap) {
    resolve_rt(ex);
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < 9 && n < cap; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        const TargetFrames *e = tfr_slot(t);
        int m = tfr_match(e, t);
        int w = snprintf(buf + n, cap - n, "%s%c%d fmt %d %ux%u L%u seen %llu/%llu", n ? "; " : "", i < 8 ? 'c' : 'd',
                         i < 8 ? i : 0, (int)t->format, t->width, t->height, t->base_layer,
                         (unsigned long long)(m ? e->last : 0), (unsigned long long)(m ? e->prev : 0));
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "no targets");
}

// ---- passes and barriers

static int has_image(const VkImage *list, int n, VkImage img) {
    for (int i = 0; i < n; i++)
        if (list[i] == img) return 1;
    return 0;
}

static VkImageMemoryBarrier2 layout_barrier(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
}

// AC_TRACE=perf: the barriers of the replay by what asked for them, summed over the submissions between two logs
// (bar_log). The source of the barrier before names the work recorded between the two: a pass after VK_BAR_PASS, a
// clear after VK_BAR_CLEAR, and so on.
static struct {
    uint64_t rec[VK_BAR_N][VK_BAR_N + 1];   // recorded: [source][source of the one before; VK_BAR_N: the first]
    uint64_t skipped[VK_BAR_N];             // asked for with nothing recorded since the last barrier
    uint64_t ended[VK_END_N];               // open passes ended, by what ended them
    uint64_t passes, same_rt;               // passes begun; those on the targets of the pass before (a restart)
    uint64_t game, game_in_pass;            // OP_BARRIER ops; those inside an open pass
    uint64_t clears_load, clears_image;     // whole-target clears: folded into a pass's load op, recorded as image clears
    uint64_t narrow, elided;                // pass begins behind a narrow barrier (counted in rec too), behind none
    uint64_t need_read, need_hint;          // chain_need: an attachment sampled since the last barrier, a hinted pass
    uint64_t need_write, need_alias;        // an attachment attached to a pass since then, aliasing memory either way
    uint64_t restart_read, restart_write;   // vk_exec_chain_draw: restarts for a sampled image, for a storage write
    uint64_t hint_drop;                     // hints a pass did not confirm
    struct { int32_t bits; uint64_t n; } game_bits[8];   // their NVNbarrierBits values
} BC;

static const char *const bar_name[VK_END_N] = {"pass", "clear", "dispatch", "copy", "game", "update", "inline", "host",
                                               "end", "targets", "area", "feedback", "chain", "other"};

static void bar_log(uint64_t subs) {
    double k = 1.0 / (double)subs;
    char buf[1024];
    size_t o = 0;
    uint64_t all = 0;
    for (int s = 0; s < VK_BAR_N; s++) {
        uint64_t n = 0;
        for (int p = 0; p <= VK_BAR_N; p++) n += BC.rec[s][p];
        all += n;
        if ((!n && !BC.skipped[s]) || o > sizeof buf - 320) continue;
        o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%s %.1f (", o ? "; " : "", bar_name[s], (double)n * k);
        for (int p = 0; p <= VK_BAR_N; p++)
            if (BC.rec[s][p])
                o += (size_t)snprintf(buf + o, sizeof buf - o, "after %s %.1f, ", p < VK_BAR_N ? bar_name[p] : "nothing",
                                      (double)BC.rec[s][p] * k);
        o += (size_t)snprintf(buf + o, sizeof buf - o, "skipped %.1f)", (double)BC.skipped[s] * k);
    }
    rt_log("perf: barriers per submission %.1f by source: %s", (double)all * k, buf);
    if (vk_exec_chain_on())
        rt_log("perf: pass chain per submission: pass begins behind a narrow barrier %.1f, behind none %.1f; barriers kept "
               "for an attachment attached since the last one %.1f, sampled %.1f, aliasing either %.1f, for a hint %.1f; "
               "passes restarted for a sampled image %.1f, for a storage write %.1f; hints dropped %.1f",
               (double)BC.narrow * k, (double)BC.elided * k, (double)BC.need_write * k, (double)BC.need_read * k,
               (double)BC.need_alias * k, (double)BC.need_hint * k,
               (double)BC.restart_read * k, (double)BC.restart_write * k, (double)BC.hint_drop * k);
    o = 0, buf[0] = 0;
    for (int e = 0; e < VK_END_N; e++)
        if (BC.ended[e]) o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%s %.1f", o ? ", " : "", bar_name[e], (double)BC.ended[e] * k);
    char bits[256] = "";
    size_t b = 0;
    for (int i = 0; i < 8 && BC.game_bits[i].n; i++)
        b += (size_t)snprintf(bits + b, sizeof bits - b, " %#x x%.1f", (unsigned)BC.game_bits[i].bits, (double)BC.game_bits[i].n * k);
    rt_log("perf: passes per submission %.1f (%.1f on the targets of the pass before), ended by: %s; game barriers %.1f "
           "(%.1f in a pass), bits:%s; whole-target clears: %.1f as load ops, %.1f as image clears",
           (double)BC.passes * k, (double)BC.same_rt * k, buf, (double)BC.game * k, (double)BC.game_in_pass * k, bits,
           (double)BC.clears_load * k, (double)BC.clears_image * k);
    memset(&BC, 0, sizeof BC);
}

// ---- the pass chain
// Every image stays in GENERAL, so a pass begin needs a barrier only for what it touches of the work recorded since
// the last one. Measured in Harmony Square on an Arc 140T at 1080p: 135 of the 173 full barriers of a frame were pass
// begins behind a pass, and leaving the memory barriers of all pass begins out took 1.13 ms of the 7.7 ms of GPU time
// a frame (a wrong picture: the ceiling). Two steps take most of it back:
// - AC_BAR_NARROW (default 1): a pass begin that follows nothing but passes which wrote attachments only takes a
//   barrier whose source is the attachment writes (and stages) of those passes instead of every write of every
//   command; its destination stays every later command and memory access. Same orderings for what was recorded; the
//   driver waits for and flushes less (0.25 ms of 7.4 a frame there, A/B in one run).
// - AC_BAR_ELIDE (default 1): such a pass begin takes no barrier at all when it touches nothing those passes touched.
//   0.27 ms of 7.2 a frame there on top of the narrow barriers; with them and the clears kept for the next pass
//   (clear_later), 0.55 ms of 7.4 (7 %), and 173 barriers a frame become 64, 49 of them narrow.
// The chain is what was recorded since the last barrier: chain_at says it is passes only, chain_other that one of
// their draws wrote a storage buffer or image (then the next barrier is a full one, as is every barrier of another
// source). Images carry two stamps of sync_epoch, which every recorded barrier bumps: sync_w, set when a pass ends
// with the image attached, and sync_r, set when a draw samples it (vk_exec_draw.c chain_draw), each with the levels
// and layers that covers (their union within the epoch: the shadow cascades are layers of one image drawn pass after
// pass, a mip chain levels of one image). An image bound over pool memory (VkTexImage.bound) may share its bytes with
// another image, so those also go into two lists of memory ranges, attached and sampled since the last barrier (CR
// below), compared by address with the other images.
// What the elision relies on: Maxwell's texture and shader caches do not see ROP writes, so NVN has the game call
// nvnCommandBufferBarrier (INVALIDATE_TEXTURE, INVALIDATE_SHADER) before it samples or loads what it rendered
// (docs/maxwell.md 1.9), and the replay records that call as a full barrier. What Maxwell orders by itself are the
// ROP's accesses: a pass may draw over, load or test against what an earlier pass stored. Vulkan orders attachment
// accesses only inside one render pass instance, so the chain tracks those: a pass begins without a barrier only
// when none of its attachments (nor memory they alias) was attached to a pass or sampled since the last barrier.
// Reads the game does not announce are still looked for: a draw that samples an image written since the last
// barrier, or writes storage in a pass that began without a barrier, restarts the pass behind the barrier it needs
// (vk_exec_chain_draw), as a feedback loop does. A pass
// restarted for a sampled image is remembered (the hints) and takes its barrier at its begin from then on. A pass is
// named by its targets, the targets of the pass before and how many passes of the submission had that pair before
// it: the frame draws several passes on the same pair of targets (blur ping-pong), only some of which read the other.
// A hinted pass none of whose draws sampled an image written before its barrier counts against the hint, which goes
// after HINT_KEEP such passes in a row, so that a pass which needs its barrier one frame in two keeps it.
// The first synchronization scope of a narrow barrier holds every earlier attachment write and, as execution, the
// logically earlier stages of those draws (vertex and fragment shaders), so it also orders their reads before any
// later write. Not seen by the elision: a buffer read (uniform, storage, vertex, index) of an attachment's pool bytes
// that the game does not precede with its barrier; the narrow barrier keeps every later access in its second scope.
static int chain_knob(int elide) {
    static int mode[2] = {-1, -1};
    if (mode[elide] < 0) {
        mode[elide] = rt_knob(elide ? "BAR_ELIDE" : "BAR_NARROW", "1");
        if (mode[elide] != 1)
            rt_log("vk_exec: pass begins %s: %s (AC_BAR_%s)", elide ? "without a barrier where no hazard is known"
                                                                    : "behind a barrier of attachment writes only",
                   mode[elide] == 2 ? "in the B blocks" : mode[elide] ? "on" : "off", elide ? "ELIDE" : "NARROW");
    }
    return mode[elide];
}

int vk_exec_chain_on(void) { return (chain_knob(0) || chain_knob(1) ? 1 : 0) | (chain_knob(1) ? 2 : 0); }

#define HINT_SETS 1024
#define HINT_WAYS 4
#define HINT_KEEP 16
// Passes that took a restart for a sampled image, in sets of HINT_WAYS: a frame has some thirty of them, and two in
// one direct-mapped slot would evict each other every frame (measured: 2 restarts a frame for good).
static struct { uint64_t key; uint32_t left; } hints[HINT_SETS][HINT_WAYS];
static uint8_t pair_n[256];   // passes of this submission per pair of targets, direct-mapped by the pair's hash

// The hint of pass key, or NULL. add: the way to put it in when it has none (an empty one, else the one closest to
// going).
static typeof(hints[0][0]) *hint_find(uint64_t key, int add) {
    typeof(hints[0][0]) *set = hints[key * 0x9E3779B97F4A7C15ull >> 54], *victim = &set[0];
    for (int w = 0; w < HINT_WAYS; w++) {
        if (set[w].key == key) return &set[w];
        if (!set[w].key || (victim->key && set[w].left < victim->left)) victim = &set[w];
    }
    return add ? victim : NULL;
}

// Pool memory ranges of bound images since the last barrier: [0] attached to a pass that ended, [1] sampled. A full
// list counts as covering everything.
#define CR_N 64
static struct { uint64_t lo[CR_N], hi[CR_N]; const VkTexImage *img[CR_N]; int n, full; } CR[2];

static void cr_reset(void) { CR[0].n = CR[0].full = CR[1].n = CR[1].full = 0; }

static void cr_add(int k, const VkTexImage *img) {
    if (CR[k].n == CR_N) { CR[k].full = 1; return; }
    CR[k].lo[CR[k].n] = img->addr, CR[k].hi[CR[k].n] = img->addr + img->mem_size, CR[k].img[CR[k].n++] = img;
}

// Another image of list k over bytes img is bound over (img itself is compared by subresource).
static int cr_hits(int k, const VkTexImage *img) {
    if (!img->bound || !img->mem_size) return 0;
    if (CR[k].full) return 1;
    uint64_t lo = img->addr, hi = img->addr + img->mem_size;
    for (int i = 0; i < CR[k].n; i++)
        if (CR[k].img[i] != img && CR[k].lo[i] < hi && lo < CR[k].hi[i]) return 1;
    return 0;
}

// Levels and layers: s[0..1] levels, s[2..3] layers. A 3D image's attachments name depth planes where its sampled
// views name its one layer: both cover every layer.
static void sub_set(uint32_t *s, const VkTexImage *img, uint32_t l0, uint32_t l1, uint32_t a0, uint32_t a1) {
    if (img->type == VK_IMAGE_TYPE_3D) a0 = 0, a1 = UINT32_MAX;
    s[0] = l0, s[1] = l1, s[2] = a0, s[3] = a1;
}

static int sub_meets(const uint32_t *u, const uint32_t *s) { return u[0] < s[1] && s[0] < u[1] && u[2] < s[3] && s[2] < u[3]; }

// Stamps s into the image's subresources of this epoch (u, at stamp *at): the first one of the epoch replaces them,
// the others widen them. Returns 1 for the first.
static int sub_note(uint32_t *at, uint32_t *u, const uint32_t *s, uint32_t ep) {
    if (*at != ep) {
        *at = ep;
        memcpy(u, s, 4 * sizeof *u);
        return 1;
    }
    if (s[0] < u[0]) u[0] = s[0];
    if (s[1] > u[1]) u[1] = s[1];
    if (s[2] < u[2]) u[2] = s[2];
    if (s[3] > u[3]) u[3] = s[3];
    return 0;
}

static void target_sub(const VkExecTarget *t, uint32_t *s) {
    sub_set(s, t->img, t->level, t->level + 1, t->base_layer, t->base_layer + t->layers);
}

int vk_exec_chain_sampled(VkExec *ex, const VkTexImage *img, uint32_t l0, uint32_t l1, uint32_t a0, uint32_t a1) {
    uint32_t ep = ex->sync_epoch, s[4];
    VkTexImage *m = (VkTexImage *)img;
    sub_set(s, img, l0, l1, a0, a1);
    int hit = (img->sync_w == ep && sub_meets(img->sync_wsub, s)) || (CR[0].n + CR[0].full && cr_hits(0, img));
    if (sub_note(&m->sync_r, m->sync_rsub, s, ep) && img->bound && img->mem_size) cr_add(1, img);
    return hit;
}

// Whether the pass about to begin on the resolved targets needs its barrier although nothing but passes was recorded
// since the last one: an attachment, or memory it aliases, was attached to one of those passes or sampled by their
// draws, or the pass took a restart before (a hint).
static int chain_need(VkExec *ex) {
    uint32_t ep = ex->sync_epoch;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image || !t->img) continue;
        const VkTexImage *img = t->img;
        uint32_t s[4];
        target_sub(t, s);
        if (img->sync_w == ep && sub_meets(img->sync_wsub, s)) return BC.need_write++, 1;
        if (img->sync_r == ep && sub_meets(img->sync_rsub, s)) return BC.need_read++, 1;
        if (cr_hits(0, img) || cr_hits(1, img)) return BC.need_alias++, 1;
    }
    if (ex->pass_key && hint_find(ex->pass_key, 0)) {
        ex->pass_hinted = 1;
        ex->hint_epoch = ex->sync_epoch;
        return BC.need_hint++, 1;
    }
    return 0;
}

// A barrier of kind VK_PASS_* (a pass begin may ask for a narrow one or none; anything else is VK_PASS_FULL). The
// images in the feedback layout become want[] (pass begin) or none (every other barrier): the others go back to
// GENERAL and the new ones leave it, in the same command, and that barrier is a full one. Returns the kind of the
// barrier the caller's commands follow.
static int barrier_to(VkExec *ex, int src, int kind, const VkImage *want, const VkImageAspectFlags *wasp, int nwant) {
    vk_exec_pass_end_why(ex, src);
    if (ex->npclear && src != VK_BAR_PASS) vk_exec_clears_flush(ex);   // a pass begin folds or flushes them itself
    VkImageMemoryBarrier2 ib[18];
    uint32_t n = 0;
    for (int i = 0; i < ex->nfbl; i++)
        if (!has_image(want, nwant, ex->fbl_img[i]))
            ib[n++] = layout_barrier(ex->fbl_img[i], ex->fbl_aspect[i], VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
                                     VK_IMAGE_LAYOUT_GENERAL);
    for (int i = 0; i < nwant; i++)
        if (!has_image(ex->fbl_img, ex->nfbl, want[i]))
            ib[n++] = layout_barrier(want[i], wasp[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT);
    for (int i = 0; i < nwant; i++) ex->fbl_img[i] = want[i], ex->fbl_aspect[i] = wasp[i];
    ex->nfbl = nwant;
    // The end of a submission is never left out: a full barrier just before it reaches no host stage (below).
    if (ex->barrier_at == ex->recorded && !n && src != VK_BAR_END) {
        BC.skipped[src]++;
        return ex->bar_narrow ? VK_PASS_NARROW : VK_PASS_FULL;
    }
    if (n) kind = VK_PASS_FULL;
    if (kind == VK_PASS_NONE) return BC.elided++, VK_PASS_NONE;
    BC.rec[src][ex->bar_prev]++;
    BC.narrow += kind == VK_PASS_NARROW;
    ex->bar_prev = src;
    vk_exec_frame.fb_layouts += n;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    if (kind == VK_PASS_NARROW) {
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    // The game reads what the GPU wrote into its memory once the timeline reaches the submission (shadow storage on
    // Linux, imported pools on integrated GPUs). A fence or semaphore wait makes device writes visible to the device
    // only, and ALL_COMMANDS holds no host stage, so the submission's last barrier also carries the writes into host
    // reads. The VRAM mirror's write-back copies have their own (vk_mem_mirror_end). Most drivers flush at the end of
    // a submission anyway; the specification asks for the dependency (Longshot 905e36a9 made the same change).
    if (src == VK_BAR_END) mb.dstStageMask |= VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask |= VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = n, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->barrier_at = ++ex->recorded;
    ex->chain_at = ex->recorded;
    ex->chain_other = 0;
    ex->bar_narrow = kind == VK_PASS_NARROW;
    if (!++ex->sync_epoch) ex->sync_epoch = 1;   // 0 is an image that was never stamped
    cr_reset();
    return kind;
}

int vk_exec_chain_draw(VkExec *ex, int writes, int unsynced) {
    if (writes) ex->chain_other = 1;
    if (!unsynced && !(writes && ex->pass_how == VK_PASS_NONE)) return 0;
    uint64_t key = ex->pass_key;
    ex->pass_force = 1;
    vk_exec_pass_end_why(ex, VK_END_CHAIN);
    vk_exec_pass_begin(ex);   // behind a narrow barrier for a sampled image, a full one for a storage write (chain_other)
    if (unsynced) {
        BC.restart_read++;
        if (key) *hint_find(key, 1) = (typeof(hints[0][0])){key, HINT_KEEP};
    } else {
        BC.restart_write++;
    }
    return 1;
}

void vk_exec_barrier(VkExec *ex, int src) { barrier_to(ex, src, VK_PASS_FULL, NULL, NULL, 0); }

// ---- whole-target clears kept for the next pass
// The game clears a target and then draws into it: SetRenderTargets, ClearColor, ClearDepthStencil, draws. Recorded as
// they come, each clear was a full barrier and a vkCmdClear*Image, and the pass after it began behind another full
// barrier and loaded the cleared texels (in Harmony Square: 29 of the 173 full barriers of a frame, and a write plus a
// read of every cleared target). A clear of a whole subresource outside a pass waits in VkExec.pclear instead; the
// pass that attaches that subresource over its whole render area takes it as VK_ATTACHMENT_LOAD_OP_CLEAR. Whatever
// else would be recorded first (a barrier of any other source, a pass that does not attach it, the end of the
// submission) records the waiting clears as image clears behind one barrier, so nothing reads or writes the image
// between the game's clear and ours.

void vk_exec_clears_flush(VkExec *ex) {
    int n = ex->npclear;
    if (!n) return;
    ex->npclear = 0;   // first: the barrier below flushes
    barrier_to(ex, VK_BAR_CLEAR, VK_PASS_FULL, NULL, NULL, 0);
    for (int i = 0; i < n; i++) {
        const struct VkExecClear *p = &ex->pclear[i];
        VkImageSubresourceRange r = {p->aspect, p->level, 1, p->base_layer, p->layers};
        if (p->aspect & VK_IMAGE_ASPECT_COLOR_BIT) vkCmdClearColorImage(ex->cb, p->image, VK_IMAGE_LAYOUT_GENERAL, &p->v.color, 1, &r);
        else vkCmdClearDepthStencilImage(ex->cb, p->image, VK_IMAGE_LAYOUT_GENERAL, &p->v.depthStencil, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(p->tex);
        BC.clears_image++;
    }
}

// A whole-target clear waits for the next pass. A later clear of the same subresource replaces the value of the
// aspects it clears.
static void clear_later(VkExec *ex, const VkExecTarget *t, VkImageAspectFlags aspect, const VkClearValue *v) {
    struct VkExecClear *p = NULL;
    for (int i = 0; i < ex->npclear && !p; i++) {
        struct VkExecClear *q = &ex->pclear[i];
        if (q->image != t->image || q->level != t->level) continue;
        if (q->base_layer == t->base_layer && q->layers == t->layers) {
            p = q;
        } else if (q->base_layer < t->base_layer + t->layers && t->base_layer < q->base_layer + q->layers) {
            // other layers of the level that share some with a waiting clear (an array, then one of its layers): the
            // waiting ones are recorded first, so the two clears keep their order behind separate barriers
            vk_exec_clears_flush(ex);
            break;
        }
    }
    if (!p) {
        if (ex->npclear == VK_EXEC_CLEARS) vk_exec_clears_flush(ex);
        p = &ex->pclear[ex->npclear++];
        *p = (struct VkExecClear){.image = t->image, .tex = t->tex, .level = t->level, .base_layer = t->base_layer,
                                  .layers = t->layers};
    }
    if (aspect & VK_IMAGE_ASPECT_COLOR_BIT) p->v.color = v->color;
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) p->v.depthStencil.depth = v->depthStencil.depth;
    if (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) p->v.depthStencil.stencil = v->depthStencil.stencil;
    p->aspect |= aspect;
}

// At a pass begin, targets resolved: a waiting clear becomes a load op when it clears an attachment of this pass over
// the pass's whole area and layers; the others are recorded as image clears first, since the pass may sample them.
// Returns the attachments to clear (bit i colour i, bit 8 depth, bit 9 stencil), their values in v[0..8] (8: depth
// and stencil).
static uint32_t clears_fold(VkExec *ex, VkClearValue v[9]) {
    uint32_t m = 0;
    int keep = 0;
    for (int i = 0; i < ex->npclear; i++) {
        const struct VkExecClear *p = &ex->pclear[i];
        int slot = -1;
        for (int k = 0; k < 9 && slot < 0; k++) {
            const VkExecTarget *t = k < 8 ? &ex->color[k] : &ex->depth;
            if ((k < 8 && (uint32_t)k >= ex->pass_ncolor) || t->image != p->image || t->level != p->level ||
                t->base_layer != p->base_layer || t->layers != p->layers)
                continue;
            if (t->width != ex->pass_w || t->height != ex->pass_h || t->layers != ex->pass_layers) continue;
            if (k < 8 ? !(p->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || (m >> k & 1) : (p->aspect & ~t->aspect) != 0) continue;
            slot = k;
        }
        if (slot < 0) {
            ex->pclear[keep++] = *p;
            continue;
        }
        if (slot < 8) m |= 1u << slot;
        else m |= (p->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? 0x100u : 0) | (p->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? 0x200u : 0);
        v[slot] = p->v;
        BC.clears_load++;
    }
    ex->npclear = keep;
    vk_exec_clears_flush(ex);
    return m;
}

void vk_exec_pass_begin(VkExec *ex) {
    if (ex->in_pass) return;
    resolve_rt(ex);
    vk_contact_pass_begin(ex);
    vk_ao_pass(ex);     // after the opaque lighting, before the transparents (vk_ao.c)
    vk_dlaa_pass(ex);   // DLAA runs before the bloom chain's first pass (vk_dlaa.c)
    // Learned targets (VkTexImage.fb_learned) are attached in the feedback layout
    VkImage want[9];
    VkImageAspectFlags wasp[9];
    int nwant = 0;
    uint32_t fbl = 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || !t->image || !t->img || !t->img->fb_learned) continue;
        fbl |= 1u << i;
        if (has_image(want, nwant, t->image)) continue;
        want[nwant] = t->image, wasp[nwant++] = i < 8 ? VK_IMAGE_ASPECT_COLOR_BIT : t->img->aspect;
    }
    VkClearValue clv[9];
    uint32_t cl = ex->npclear ? clears_fold(ex, clv) : 0;   // before the barrier: clears that do not fold are recorded
    if (ex->pass_rt != ex->rt_serial && (vk_exec_chain_on() & 2)) {   // other targets than the pass before: its name
        uint64_t pair = ex->last_rt_hash * 0x9E3779B97F4A7C15ull ^ ex->rt_hash;
        ex->pass_key = (pair + pair_n[pair >> 56]++ * 0xD6E8FEB86659FD93ull) | 1;
        ex->last_rt_hash = ex->rt_hash;
    }
    // The barrier: a full one, unless nothing but passes that wrote attachments only was recorded since the last
    // barrier (the pass chain)
    int kind = VK_PASS_FULL, chain = vk_exec_chain_on();
    ex->pass_hinted = ex->hint_hit = 0;
    if (chain && !ex->chain_other && ex->recorded == ex->chain_at && !vk_dump_armed) {
        if (RT_KNOB_ON(chain_knob(0))) kind = VK_PASS_NARROW;
        if (RT_KNOB_ON(chain_knob(1)) && !ex->pass_force && !chain_need(ex)) kind = VK_PASS_NONE;
    }
    ex->pass_force = 0;
    ex->pass_how = (uint8_t)barrier_to(ex, VK_BAR_PASS, kind, want, wasp, nwant);
    BC.passes++;
    if (ex->pass_rt == ex->rt_serial) BC.same_rt++;
    ex->pass_rt = ex->rt_serial;
    if (vk_gpuprof_on()) vk_gpuprof_mark(ex, 1, 0);
    ex->fbl_mask = fbl;
    ex->fb_dirty = (cl & 0xff) | (cl & 0x300 ? 0x100u : 0);   // a load-op clear writes its attachment in this pass
    ex->pass_serial++;
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    for (uint32_t i = 0; i < ex->pass_ncolor; i++) {
        const VkExecTarget *t = &ex->color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (!t->image) continue;
        ca[i].imageView = view_of(ex, t);
        ca[i].imageLayout = fbl >> i & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = cl >> i & 1 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (cl >> i & 1) ca[i].clearValue = clv[i];
    }
    const VkExecTarget *d = &ex->depth;
    if (d->image) {
        VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view_of(ex, d),
                                       .imageLayout = fbl >> 8 & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                   : VK_IMAGE_LAYOUT_GENERAL,
                                       .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        if (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) da = a;
        if (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = a;
        if (cl & 0x100) da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, da.clearValue = clv[8];
        if (cl & 0x200) sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, sa.clearValue = clv[8];
    }
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {ex->pass_w, ex->pass_h}},
                          .layerCount = ex->pass_layers, .colorAttachmentCount = ex->pass_ncolor,
                          .pColorAttachments = ca,
                          .pDepthAttachment = ex->pass_depth ? &da : NULL,
                          .pStencilAttachment = ex->pass_stencil ? &sa : NULL};
    vkCmdBeginRendering(ex->cb, &ri);
    vk_exec_frame.passes++;
    ex->recorded++;
    ex->in_pass = 1;
}

void vk_exec_pass_end_why(VkExec *ex, int why) {
    if (!ex->in_pass) return;
    BC.ended[why]++;
    vk_exec_oq_pass_end(ex);
    vkCmdEndRendering(ex->cb);
    ex->recorded++;
    ex->in_pass = 0;
    ex->chain_at = ex->recorded;   // the pass chain: whatever is recorded from here on that is not a pass breaks it
    if (vk_exec_chain_on() & 2) {
        for (int i = 0; i < 9; i++) {
            const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
            if (!t->image || !t->img) continue;
            VkTexImage *m = (VkTexImage *)t->img;
            uint32_t s[4];
            target_sub(t, s);
            if (sub_note(&m->sync_w, m->sync_wsub, s, ex->sync_epoch) && m->bound && m->mem_size) cr_add(0, m);
        }
        typeof(hints[0][0]) *hs = ex->pass_hinted ? hint_find(ex->pass_key, 0) : NULL;
        if (hs) {
            if (ex->hint_hit) hs->left = HINT_KEEP;
            else if (!--hs->left) hs->key = 0, BC.hint_drop++;
        }
        ex->pass_hinted = 0;
    }
    for (int i = 0; i < ex->pass_ncolor && i < 8; i++)   // stand-ins of 3D BC targets go to their textures
        if (ex->color[i].image && ex->color[i].img && ex->color[i].img->image != ex->color[i].image)
            bc_copy_back(ex, &ex->color[i]);
    if (vk_gpuprof_on()) vk_gpuprof_mark(ex, 2, 0);
    if (vk_dump_armed) vk_dump_pass_end(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    for (int i = 0; i < 8; i++)
        if (ex->color[i].image) vk_texture_gpu_written(ex->color[i].tex), tfr_note(&ex->color[i], stamp);
    if (ex->depth.image) vk_texture_gpu_written(ex->depth.tex), tfr_note(&ex->depth, stamp);
}

VkCommandBuffer vk_exec_inline_begin(int *reopen) {
    VkExec *ex = &X;
    *reopen = 0;
    if (!x_here || !ex->cb) return VK_NULL_HANDLE;
    *reopen = ex->in_pass;
    vk_exec_barrier(ex, VK_BAR_INLINE);   // ends the pass; feedback-layout targets go back to GENERAL
    return ex->cb;
}

void vk_exec_inline_end(int reopen) {
    X.recorded++;
    if (reopen) vk_exec_pass_begin(&X);
}

typedef struct { VkBuffer b; VmaAllocation a; } InlineBuf;
static void free_inline_buf(void *arg) {   // completion thread
    InlineBuf *t = arg;
    vmaDestroyBuffer(vk_ctx()->vma, t->b, t->a);
    free(t);
}

VkBuffer vk_exec_inline_buffer(VkDeviceSize size) {
    InlineBuf *t = calloc(1, sizeof *t);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (!t || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &t->b, &t->a, NULL) != VK_SUCCESS) { free(t); return VK_NULL_HANDLE; }
    vk_completion_add(X.fence, free_inline_buf, t);
    return t->b;
}

VkRect2D vk_exec_scissor(const VkExec *ex) {
    int64_t w = ex->pass_w, h = ex->pass_h;
    if (!ex->has_scissor) return (VkRect2D){{0, 0}, {(uint32_t)w, (uint32_t)h}};
    double k = ex->pass_scale > 1.0f ? ex->pass_scale : 1.0;
    int64_t x0 = llround(ex->scissor.x * k), y0 = llround(ex->scissor.y * k);
    int64_t x1 = llround((ex->scissor.x + ex->scissor.w) * k), y1 = llround((ex->scissor.y + ex->scissor.h) * k);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    if (x1 <= x0 || y1 <= y0) return (VkRect2D){{0, 0}, {0, 0}};
    return (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
}

// ---- clears

// Outside a pass, a clear of a whole target (no narrower scissor, the target as large as the render area) clears the
// image directly instead of opening a pass for it: at once (clear_color, a view of another format than the image's:
// an image clear writes the value in the image's format, a load op in the view's) or kept for the next pass's load op
// (clear_later).
static int whole_target(const VkExec *ex, const VkExecTarget *t) {
    // vkCmdClear*Image addresses a 3D level whole, not one depth plane: slices of a 3D target clear in a pass; it
    // cannot clear a compressed image: views of BC blocks clear in a pass too
    if (ex->in_pass || t->is3d || t->blocks || t->width != ex->pass_w || t->height != ex->pass_h) return 0;
    if (!ex->has_scissor) return 1;
    const p_scissor *s = &ex->scissor;
    return s->x <= 0 && s->y <= 0 && (int64_t)s->x + s->w >= t->width && (int64_t)s->y + s->h >= t->height;
}

// AC_CLEAR_LOAD (default 1; 0 records every whole-target clear at once, as before; ab alternates with the AC_AB
// blocks): whole-target clears wait for the next pass's load op (clear_later).
static int clears_later_on(void) {
    static int mode = -1;
    if (mode < 0) mode = rt_knob("CLEAR_LOAD", "1");
    return RT_KNOB_ON(mode);
}

static void clear_attachment(VkExec *ex, const VkClearAttachment *a) {
    vk_exec_pass_begin(ex);
    VkClearRect cr = {vk_exec_scissor(ex), 0, ex->pass_layers};
    if (!cr.rect.extent.width || !cr.rect.extent.height) return;
    vkCmdClearAttachments(ex->cb, 1, a, 1, &cr);
    ex->recorded++;
    ex->fb_dirty |= a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? 1u << a->colorAttachment : 0x100u;
}

static void clear_color(VkExec *ex, const p_clear_color *c) {
    vk_exec_frame.clears++;
    if (ex->render_disabled) return;
    resolve_rt(ex);
    int i = c->index;
    if (i < 0 || i >= (int)ex->pass_ncolor || !ex->color[i].image) {
        warn_once(W_CLEAR_TARGET, "colour clear of target %d, which is not bound or not renderable: skipped", i);
        return;
    }
    uint32_t mask = (uint32_t)c->mask & 0xf;
    if (!mask) return;
    if (mask != 0xf) {   // vkCmdClearAttachments ignores write masks; a masked clear needs a draw (vk_exec_draw.c)
        warn_once(W_CLEAR_MASK, "colour clear with channel mask %#x is not replayed: skipped", mask);
        return;
    }
    const VkExecTarget *t = &ex->color[i];
    VkClearColorValue cv;
    memcpy(cv.float32, c->rgba, sizeof cv.float32);   // integer targets: the payload's bits as the front end stored them
    if (whole_target(ex, t)) {
        if (clears_later_on() && t->img && t->format == t->img->format) {
            clear_later(ex, t, VK_IMAGE_ASPECT_COLOR_BIT, &(VkClearValue){.color = cv});
            return;
        }
        vk_exec_barrier(ex, VK_BAR_CLEAR);
        VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, 1, t->base_layer, t->layers};
        vkCmdClearColorImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(t->tex);
        BC.clears_image++;
        return;
    }
    VkClearAttachment a = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)i};
    a.clearValue.color = cv;
    clear_attachment(ex, &a);
}

static void clear_ds(VkExec *ex, const p_clear_ds *c) {
    if (ex->render_disabled) return;
    resolve_rt(ex);
    const VkExecTarget *t = &ex->depth;
    if (!t->image) return;
    VkImageAspectFlags asp = 0;
    if (c->depth_mask && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
    uint32_t sm = (uint32_t)c->stencil_mask & 0xff;
    if (sm && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) {
        if (sm == 0xff) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        else warn_once(W_STENCIL_MASK, "stencil clear with mask %#x is not replayed: skipped", sm);
    }
    if (!asp) return;
    float d = c->depth < 0.0f ? 0.0f : c->depth > 1.0f ? 1.0f : c->depth;
    VkClearDepthStencilValue v = {d, (uint32_t)c->stencil & 0xff};
    if (whole_target(ex, t)) {
        if (clears_later_on()) {
            clear_later(ex, t, asp, &(VkClearValue){.depthStencil = v});
            return;
        }
        vk_exec_barrier(ex, VK_BAR_CLEAR);
        VkImageSubresourceRange r = {asp, t->level, 1, t->base_layer, t->layers};
        vkCmdClearDepthStencilImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(t->tex);
        BC.clears_image++;
        return;
    }
    VkClearAttachment a = {.aspectMask = asp};
    a.clearValue.depthStencil = v;
    clear_attachment(ex, &a);
}

// ---- the op loop

static void set_rt(VkExec *ex, const p_set_rt *r) {
    if (!memcmp(&ex->rt, r, sizeof *r)) return;   // the same targets again: the pass goes on
    vk_exec_pass_end_why(ex, VK_END_TARGETS);
    ex->rt = *r;
    ex->rt_resolved = 0;
    ex->rt_serial++;
    if (vk_exec_chain_on() & 2) ex->rt_hash = nvn_hm_hash(r, sizeof *r);
}

// ---- command buffer parts
// NVN starts the GPU as soon as commands are submitted; one Vulkan command buffer per submission starts it only after
// the whole replay. Between passes, once AC_SPLIT_DRAWS (64; 0 off) draws and dispatches were recorded since the
// last part, the part goes to the queue (vk_sync_split) and the replay continues in a new command buffer. Only
// command-buffer state is lost there (bindings and dynamic state: vk_exec_draw_new_cb, ex->dirty); barriers, layouts and
// queries are not: a barrier's first scope is everything earlier in submission order, image layouts belong to the image,
// and occlusion queries end with their pass. Fences, completion callbacks and reports stay per submission: only the last
// part signals the value. nxnative commits every 64 draws between passes; MacNativeGames after 30, then every 40.
// A part also spans at least AC_SPLIT_US (1000) of replay time: a split costs about 40 us of the render thread
// (vkQueueSubmit2, a command buffer, the mirror prologue), and a fight frame of 2,000 draws made 15 parts on the 7900X;
// a millisecond of recording is a fraction of a millisecond of GPU work, so the GPU still starts early.
static int split_n = -1, split_draws, part_start;
static uint64_t split_ns, part_t0;
static uint64_t n_parts, n_subs;

static void maybe_split(VkExec *ex) {
    if (split_n < 0) {
        split_n = cfg_int("AC_SPLIT_DRAWS");
        if (split_n < 0) split_n = 0;
        int us = cfg_int("AC_SPLIT_US");
        split_ns = us > 0 ? (uint64_t)us * 1000 : 0;
        rt_log("vk: replay command buffer split every %d draws between passes, parts of at least %d us%s", split_n,
               us > 0 ? us : 0, split_n ? "" : " (off)");
    }
    if (!split_n || ex->in_pass || split_draws < split_n) return;
    if (split_ns && kern_now_ns() - part_t0 < split_ns) return;   // split_draws stays: the next draw asks again
    split_draws = 0;
    if (ex->recorded == part_start) return;
    if (vk_gpuprof_on()) vk_gpuprof_mark(ex, 5, 0);
    ex->cb = vk_sync_split();
    if (vk_gpuprof_on()) vk_gpuprof_mark(ex, 0, 0);
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
}

static void walk(VkExec *ex, const op_hdr *h, int depth) {
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        switch (h->op) {
        case OP_END: return;
        case OP_BEGIN: break;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_CALL:
            if (depth >= CALL_DEPTH) rt_fatal("vk: command buffer calls nested deeper than %d", CALL_DEPTH);
            walk(ex, nvn_op_first(((const p_jump *)a)->target), depth + 1);
            break;
        case OP_SET_RT: set_rt(ex, a); break;
        case OP_CLEAR_COLOR: clear_color(ex, a); break;
        case OP_CLEAR_DS: clear_ds(ex, a); break;
        case OP_VIEWPORTS: {
            const p_viewports *v = a;
            for (int i = 0; i < v->n && v->first + i < 16; i++)
                if (v->first + i >= 0) memcpy(ex->vp[v->first + i], v->v[i], sizeof ex->vp[0]);
            ex->has_vp = 1;
            ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
            break;
        }
        case OP_SCISSOR:
            ex->scissor = *(const p_scissor *)a;
            ex->has_scissor = 1;
            ex->dirty |= VK_EXEC_DIRTY_SCISSOR;
            break;
        case OP_DEPTH_RANGE: {
            const p_depth_range *r = a;
            ex->depth_range[0] = r->n, ex->depth_range[1] = r->f;
            ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
            break;
        }
        case OP_STENCIL: {
            const p_stencil *s = a;
            for (int f = 0; f < 2; f++) {
                if (!(s->face & (1 << f))) continue;
                if (s->which == 0) ex->st_vmask[f] = (uint32_t)s->value_mask;
                else if (s->which == 1) ex->st_wmask[f] = (uint32_t)s->mask;
                else ex->st_ref[f] = (uint32_t)s->ref;
            }
            ex->dirty |= VK_EXEC_DIRTY_STENCIL;
            break;
        }
        case OP_POLY_OFFSET:
            ex->poly_offset = *(const p_poly_offset *)a;
            ex->dirty |= VK_EXEC_DIRTY_DEPTH_BIAS;
            break;
        case OP_BLEND_COLOR:
            memcpy(ex->blend_color, ((const p_blend_color *)a)->rgba, sizeof ex->blend_color);
            ex->dirty |= VK_EXEC_DIRTY_BLEND_CONST;
            break;
        case OP_DEPTH_CLAMP: ex->pipe.depth_clamp = ((const p_flag *)a)->value != 0; break;
        case OP_RASTER_DISCARD: ex->pipe.raster_discard = ((const p_flag *)a)->value != 0; break;
        case OP_PRIM_RESTART: ex->prim_restart = *(const p_prim_restart *)a; break;
        case OP_COPY_STRIDE: ex->copy_stride = ((const p_copy_stride *)a)->stride; break;
        case OP_SET_TEX_POOL: ex->tex_pool = ((const p_set_pool *)a)->pool, ex->tex_serial++; break;
        case OP_SET_SAMP_POOL: ex->samp_pool = ((const p_set_pool *)a)->pool; break;
        case OP_BARRIER: {
            int32_t bits = ((const p_barrier *)a)->bits;
            BC.game++, BC.game_in_pass += ex->in_pass != 0;
            for (int i = 0; i < 8; i++)
                if (!BC.game_bits[i].n || BC.game_bits[i].bits == bits) {
                    BC.game_bits[i].bits = bits, BC.game_bits[i].n++;
                    break;
                }
            vk_exec_barrier(ex, VK_BAR_GAME);
            break;
        }
        case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: case OP_UPDATE_UB: case OP_DRAW_IDX_IND: case OP_DISPATCH_IND:
            if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND) vk_exec_frame.draws_seen++, split_draws++;
            if (h->op == OP_DISPATCH || h->op == OP_DISPATCH_IND) vk_exec_frame.dispatches++, split_draws++;
        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
        case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG:
            vk_exec_op_draw(ex, h->op, a);
            if ((h->op == OP_DISPATCH || h->op == OP_DISPATCH_IND) && !ex->in_pass && vk_gpuprof_on())
                vk_gpuprof_mark(ex, 3, ex->prog ? ex->prog->serial : 0);
            break;
        case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX:
            if (h->op != OP_CLEAR_TEX) vk_exec_frame.copies++;
            if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
            // fall through
        case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
        case OP_CLEAR_BUF:
            vk_exec_op_xfer(ex, h->op, a);
            if ((h->op == OP_COPY_B2B || h->op == OP_COPY_B2T || h->op == OP_COPY_T2B || h->op == OP_COPY_T2T ||
                 h->op == OP_CLEAR_TEX || h->op == OP_CLEAR_BUF) && !ex->in_pass && vk_gpuprof_on())
                vk_gpuprof_mark(ex, 4, h->op);
            break;
        case OP_PHOTO: vk_photo_copy(ex, a); break;
        case OP_PHOTO_FILL: vk_photo_fill(ex, a); break;
        default: rt_fatal("vk: op %s has no replay handler", nvn_op_name(h->op));
        }
        if (split_draws >= split_n) maybe_split(ex);
        h = nvn_op_next(h);
    }
}

int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence) {
    VkExec *ex = &X;
    if (!x_ready) state_reset(ex), vk_dump_init(), x_ready = 1;
    ex->cb = cb;
    ex->fence = fence;
    ex->recorded = 0;
    ex->in_pass = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->rt_resolved = 0;   // images may have changed since the last command buffer
    ex->nviews = 0;
    ex->barrier_at = -1;
    ex->bar_prev = VK_BAR_N;
    ex->chain_at = -1, ex->chain_other = 0, ex->bar_narrow = 0, ex->pass_force = 0;   // the first pass begins behind a full barrier
    memset(pair_n, 0, sizeof pair_n);
    if (!++ex->sync_epoch) ex->sync_epoch = 1;
    cr_reset();
    ex->nfbl = 0;   // the previous command buffer ended with a full barrier: every image is in GENERAL
    vk_exec_frame.submits++;
    split_draws = part_start = 0;
    part_t0 = kern_now_ns();
    x_here = 1;
    vk_texture_unlist_pending();   // images made again without a format list in the last replay, and those to make
    if (n) vk_mem_mirror_begin(), vk_usnap_begin(handles, n, fence);
    if (n && vk_gpuprof_on()) vk_gpuprof_mark(ex, 0, 0);
    for (int i = 0; i < n; i++) {
        if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
        walk(ex, nvn_op_first(handles[i]), 0);
    }
    vk_exec_pass_end_why(ex, VK_BAR_END);
    vk_exec_clears_flush(ex);   // a clear no pass took: the next submission, or the present, sees it
    x_here = 0;
    if (ex->recorded) vk_exec_barrier(ex, VK_BAR_END);   // later submissions see this one's writes
    if (n) ex->recorded += vk_mem_mirror_end(ex->cb, fence), vk_usnap_end(fence);   // VRAM mirror bytes the GPU wrote go back to guest memory
    if (n && vk_gpuprof_on()) vk_gpuprof_mark(ex, 5, 0), vk_gpuprof_flush(fence);
    if (ex->nviews) {
        ViewList *l = malloc(sizeof *l + sizeof(VkImageView) * (size_t)ex->nviews);
        if (!l) rt_fatal("vk: out of memory");
        l->n = ex->nviews;
        for (int i = 0; i < ex->nviews; i++) l->v[i] = ((ViewEnt *)ex->views)[i].view;
        vk_completion_add(fence, free_views, l);
        ex->nviews = 0;
    }
    ex->cb = VK_NULL_HANDLE;
    if (n) n_subs++;
    if ((rt_trace & RT_TRACE_PERF) && n_subs >= 240) {
        rt_log("perf: replay command buffer parts per submission %.2f", (double)(n_subs + n_parts) / (double)n_subs);
        bar_log(n_subs);
        vk_exec_draw_perf(n_subs);
        n_subs = n_parts = 0;
    }
    return ex->recorded - part_start;   // what the last part holds
}
