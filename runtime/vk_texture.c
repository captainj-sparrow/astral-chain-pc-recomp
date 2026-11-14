// Vulkan textures of the graphics layer: a VkImage cache keyed by storage
// GPU address plus descriptor, uploads through staging buffers with block-linear deswizzle and ASTC decode (vk_astc.c),
// re-uploads when page write generations (vk_memory.c) show the storage written, ReadTexels / WriteTexels, the texture
// and sampler descriptor slots, and VkSamplers.
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn @ b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile): which textures mirror their pool memory (nvn_mtl.m tex_mirrored, 3347-3362), the
// "GPU-owned" rule that stops re-uploads once the GPU rendered into a mirrored texture (nvn_mtl.m tex_mtl, 3366-3372),
// the live texture at a storage address that a slot of a finalized texture follows (hle_nvn.c:373-393), and
// ReadTexels / WriteTexels (nvn_mtl.m:10105-10195). Image creation, upload regions and samplers are ported from LAS
// Recomp runtime/nvn/vulkan/textures.c:235-330, 778-844 @ da35b83a (MIT, Copyright (c) 2026 Peiikai).
//
// Every image lives in VK_IMAGE_LAYOUT_GENERAL. Uploads are recorded into the upload batch of vk_memory.c; the replay
// submits it (vk_upload_flush) before a submission that samples the textures.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

typedef struct {
    uint64_t addr;
    nvn_texdesc d;
    uint8_t has_layout, layout[8], pad[7];
} TexKey;

typedef struct Tex {
    VkTexImage pub;               // first: vk_texture_get hands out &pub
    TexKey key;
    tex_t t;                      // the newest texture object seen with this key
    TexVkFormat f;
    VmaAllocation alloc;          // own memory; NULL when bound into pool memory
    VmaAllocation bound_to;       // the pool memory it is bound into
    uint64_t bytes;               // own allocation size, 0 when bound into pool memory (cache budget)
    uint32_t last_use;            // submit_epoch of the last bind or lookup (eviction order)
    uint32_t wgen;                // page generation of the last upload check
    uint32_t checked;             // submit_epoch of the last check from a draw binding
    int uploaded, released;
    uint32_t n_up;                // uploads of this entry (vk_texture_debug)
    int last_ch;                  // last vk_mem_changed result of make_fresh: 1 written, 0 clean, -1 not tracked
    struct Tex *next_at;          // other entries at the same storage address
    uint8_t gpu_dirty;            // mirrored, and the GPU wrote the image after the last write-back to the storage
    uint32_t gpu_gen;             // page generation when the GPU write that set gpu_dirty was recorded
    uint64_t gpu_seq;             // order of that GPU write among all of them (gpu_writes)
    uint64_t gpu_fence;           // fence of the submission that made that write (0: not recorded in one)
    uint64_t wr_fence;            // any entry: newest submission recorded to write its image (render, copy, clear,
                                  // storage binding); ReadTexels waits for it instead of draining the replay queue
    uint8_t listed;               // in owned[]
    uint8_t *snap;                // a texture that fills no page: its storage bytes when the GPU write was recorded
    uint32_t wt_n;                // WriteTexels into the storage since the last upload
    struct Tex *rel_prev, *rel_next;   // while released: the list of released entries, oldest first
    uint8_t bc_flip;                   // AC_ASTC_BC=bc1: the storage's alpha verdict changed; waiting to be
                                       // destroyed (sweep_flip) so the next lookup recreates the other format
    uint32_t bc_gen;                   // AC_ASTC_BC=bc1: page generation of create_tex's alpha scan, until the first
                                       // upload consumes it (bc_verdict_current); 0 none
    uint32_t ahead;                    // AC_TEX_PREFETCH: the submit_epoch pre_step made it at (0: made by a use). Not
                                       // drawn yet while last_use still equals it (cache_before)
    uint32_t nomem_epoch;              // submit_epoch of an ASTC upload that found video memory full (upload): the
                                       // next submit tries again, not every binding of this one
} Tex;

// A slot keeps what vk_texture_by_id resolved for it, so a draw binding skips the key lookup and the address chain: the
// entry (ce), valid while the generation of its address (agen) is still cgen, the view and has_view handed out with it,
// and the submit_epoch of the last freshness check through the slot (cepoch). Hot fields first: a lookup that hits
// touches one or two cache lines.
typedef struct {
    struct Tex *ce;               // NULL: nothing cached (registered since, or never resolved)
    uint32_t cgen;                // agen of the slot's address when ce was resolved
    uint32_t cepoch;
    view_t cview;
    uint8_t chas_view, used, has_view;
    view_t v;
    tex_t t;
} TexSlot;
typedef struct { samp_par p; VkSampler s, sp; uint8_t used; } SampSlot;   // s: vk_sampler_for(&p) once looked up; sp:
                                                                           // its point-sampled form (vk_sampler_point_by_id_locked)

static PlatRMutex tex_lock;
static PlatOnce tex_once = PLAT_ONCE_INIT;
static nvn_hmap by_key = {.klen = sizeof(TexKey)};
static nvn_hmap by_addr = {.klen = 8};
static TexSlot *tslots;
static int ntslots;
static SampSlot *sslots;
static int nsslots;
static nvn_hmap samplers = {.klen = sizeof(samp_par)};
static int nreleased;
static Tex *rel_head, *rel_tail;   // released entries in release order (sweep_released drops from the head)
static uint64_t n_uploads, n_reuploads;
// Own memory of every entry, live and released (tex_lock), and the cache budget's drops since start.
static uint64_t own_bytes, n_cache_drops, n_cache_drop_bytes;
// Re-uploads since the last perf line (tex_lock): all, and those of textures a WriteTexels wrote since their last upload.
static uint64_t ru_n, ru_bytes, ru_ns, ru_wt_n, ru_wt_bytes, ru_wt_ns;
// AC_TEX_ALIAS (default on; =0: a new image over live storage starts empty): a new descriptor over the storage of a
// live image that the GPU wrote gets that image's content (alias_copy). Counts since the last log line.
static int tex_alias = -1;
static uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
// The recompression accounting (AC_ASTC_BC=bc1), since the last perf line: scans of
// recompressed ASTC storages at entry creation — count, bytes, nanoseconds, and the split by the format the
// scan chose (BC1: provably opaque), by where the image ended up (bound into pool memory, own memory against
// the cache budget) and skipped (3D target, no storage). AC_ASTC_BC=1 (BC3 forced) scans nothing.
static uint64_t sc_n, sc_bytes, sc_ns, sc_op_n, sc_op_bytes, sc_al_n, sc_al_bytes;
static uint64_t sc_bound_n, sc_bound_bytes, sc_own_n, sc_own_bytes, sc_skip_n;
// Entries whose storage's alpha verdict turned against their image's format (upload): destroyed at the next
// submit, so the next lookup recreates them in the other recompression. Rare — a texture's content turning
// opaque or alpha-carrying after the image was made.
static Tex **bc_flip_list;
static int nbc_flip, cbc_flip;
static uint32_t submit_epoch = 1;   // bumped per queue submit: draw bindings check a texture's storage once per submit
// Mirrored textures the GPU wrote into, sorted by address (vk_texture_cpu_stale, vk_texture_write_back). Changed under
// tex_lock and owned_lock; read under either.
static PlatMutex owned_lock = PLAT_MUTEX_INIT;
static Tex **owned;
static int nowned, cowned;
static uint64_t max_owned;   // largest storage size in owned[]
static uint64_t gpu_writes;  // GPU writes into mirrored textures so far: Tex.gpu_seq
// Bumped whenever the entry a slot resolves to can change: an entry created or destroyed, released or taken back into
// use (the slot lookup and the follow of a finalized slot read only by_key, by_addr and the released flags).
static uint32_t tex_gen = 1;
// Bumped with tex_gen and whenever a slot is registered: what vk_texture_by_id returns for any slot may have changed.
static uint32_t slots_gen = 1;
static uint32_t samp_gen = 1;   // bumped whenever a sampler slot is registered: vk_sampler_by_id may answer otherwise
static uint64_t n_checks;    // draw-binding storage checks (make_fresh_snap) since the last perf line
// tex_gen per storage address (hashed): a slot resolves through the entries at its own address only (slot_resolve), so
// a slot's resolution stays valid while its address's generation does. Astral Chain finalizes and initializes transient
// textures every frame: about 90 tex_gen and 160 slots_gen bumps per frame in the monsters fight, one every 16 draws,
// which emptied every cache keyed on the global generations.
#define AGEN_N 4096
static uint32_t agen[AGEN_N];
// Per slot id, for vk_texture_slot_token without the lock: the registration serial (bits 16-47; 0: never registered)
// and the address generation index (bits 0-15), one word so a reader gets a consistent pair. Fixed-size (slot ids are
// 20 bits), unlike tslots, which moves when it grows.
static uint64_t *sinfo;
static uint32_t reg_serial;

static uint32_t agen_idx(uint64_t addr) { return (uint32_t)(addr * 0x9E3779B97F4A7C15ull >> 52); }

static void tex_gen_bump(uint64_t addr) {
    if (!++tex_gen) tex_gen = 1;
    slots_gen++;
    uint32_t *g = &agen[agen_idx(addr)];
    __atomic_store_n(g, *g + 1, __ATOMIC_RELEASE);
}

static volatile int32_t tex_ready;
static void tex_init(void *p) {
    (void)p;
    __atomic_store_n(&tex_ready, 1, __ATOMIC_RELEASE);
    return;
}
static void lock(void) {
    if (!__atomic_load_n(&tex_ready, __ATOMIC_ACQUIRE)) plat_once(&tex_once, tex_init, NULL);
    plat_rmutex_lock(&tex_lock);
}
static void unlock(void) { plat_rmutex_unlock(&tex_lock); }

static uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
static uint32_t mip(uint32_t v, unsigned lv) { return umax(v >> lv, 1); }

static void make_key(const tex_t *t, TexKey *k) {
    memset(k, 0, sizeof *k);
    k->addr = (uintptr_t)t->mem;
    k->d = t->d;
    k->has_layout = t->has_layout;
    memcpy(k->layout, t->layout, 8);
}

// Keys equal apart from the descriptor's swizzle selectors: the same storage, format and geometry.
static int same_but_swizzle(const TexKey *a, const TexKey *b) {
    TexKey x = *a, y = *b;
    x.d.swizzle_r = x.d.swizzle_g = x.d.swizzle_b = x.d.swizzle_a = 0;
    y.d.swizzle_r = y.d.swizzle_g = y.d.swizzle_b = y.d.swizzle_a = 0;
    return !memcmp(&x, &y, sizeof x);
}

// MNG nvn_mtl.m tex_mirrored: textures whose content the CPU provides in pool memory. Display, compressible, zcull,
// depth, multisample and GPU-only storage stays GPU-owned.
static int tex_mirrored(const tex_t *t) {
    if (!t->mem || nvn_gpu_only((uintptr_t)t->mem)) return 0;
    if (t->has_layout) return 1;
    const pool_t *p = t->pool;
    if (!p || !(p->flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED))) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL |
                      NVN_TEXTURE_FLAGS_DEPTH16_PREFER_FAST_CLEAR | NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL))
        return 0;
    if (t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->d.format >= 0x31 && t->d.format <= 0x36) return 0;   // stencil / depth formats
    return 1;
}

static void log_limited(volatile int32_t *n, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void log_limited(volatile int32_t *n, const char *fmt, ...) {
    if (plat_inc32(n) > 16) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    rt_log("vk: %s", buf);
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT

// ---------------------------------------------------------------- images
static VkFormatFeatureFlags features_of(VkFormat f) {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &p);
    return p.optimalTilingFeatures;
}

static VkImageAspectFlags aspect_of(const TexVkFormat *f) {
    VkImageAspectFlags a = 0;
    if (f->depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (f->stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Binds img into the pool memory behind its GPU address (physical pools, virtual pools through the page table) when
// the memory type, alignment and contiguous size allow; 0 otherwise.
// A depth / stencil image that needs more bytes than its NVN storage (guest_size) is not bound: AMD's RDNA2 driver
// (26.10.1) puts the depth compression metadata after the depth data, and above 1080p that tail lies over the next
// texture of the game's heap; each depth write then scrambled it (garbage band at the top, pink glow, blocks in the
// grass, a blue checkered scene in the menu). Colour images keep their place: the RX 9070 also sizes some of them
// above their storage, and nothing there showed that the GPU writes the excess.
static int format_compressed(VkFormat f) {   // vkCmdClearColorImage takes no block-compressed format
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
           (f >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK && f <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK);
}

static int bind_pool_memory(VkCtx *c, Tex *e, VkImage img, uint64_t guest_size) {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, img, &req);
    if ((e->f.depth || e->f.stencil) && req.size > guest_size) return 0;
    VkMemSpan sp;
    if (!req.size) return 0;   // an image the device could not lay out (image_size)
    if (!vk_mem_resolve(e->pub.addr, req.size, &sp) || !sp.alloc || sp.alloc_offset % req.alignment) return 0;
    VmaAllocationInfo info;
    vmaGetAllocationInfo(c->vma, sp.alloc, &info);
    if (!(req.memoryTypeBits >> info.memoryType & 1)) return 0;
    if (vmaBindImageMemory2(c->vma, sp.alloc, sp.alloc_offset, img, NULL) != VK_SUCCESS) return 0;
    e->bound_to = sp.alloc;
    return 1;
}

// agl's BC4 encoder draws a BC texture's blocks through an RGBA16UI view, one texel per 8-byte block (refs/nxnative
// docs/agents/04-graphics.md, Breath of the Wild and Tears of the Kingdom sections). Vulkan does it with an uncompressed
// view of a BLOCK_TEXEL_VIEW_COMPATIBLE image; EXTENDED_USAGE lets the BC image carry the COLOR_ATTACHMENT usage that
// only such views support. Whether the device takes that combination for this format and shape:
static int block_views_ok(const VkImageCreateInfo *ci) {
    VkPhysicalDeviceImageFormatInfo2 fi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                                           .format = ci->format, .type = ci->imageType, .tiling = ci->tiling,
                                           .usage = ci->usage, .flags = ci->flags};
    VkImageFormatProperties2 p = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    if (vkGetPhysicalDeviceImageFormatProperties2(vk_ctx()->phys, &fi, &p) != VK_SUCCESS) return 0;
    const VkImageFormatProperties *q = &p.imageFormatProperties;
    return ci->extent.width <= q->maxExtent.width && ci->extent.height <= q->maxExtent.height &&
           ci->extent.depth <= q->maxExtent.depth && ci->mipLevels <= q->maxMipLevels &&
           ci->arrayLayers <= q->maxArrayLayers;
}

// The memory the device wants for an image made from ci: 0 when it cannot lay it out. RADV on GFX9 (Vega, Renoir)
// hands back a 0-byte image for a mipmapped block-compressed 3D image at least as deep as it is wide in blocks, with
// COLOR_ATTACHMENT usage and no STORAGE: that usage asks for a thin 3D layout (Mesa radv_image.c), for which the GFX9
// address library has no swizzle mode left (gfx9addrlib.cpp), and the error is dropped; VMA then refuses the 0-byte
// allocation. The format properties call it supported, so only the size shows it.
static VkDeviceSize image_size(const VkImageCreateInfo *ci) {
    VkDeviceImageMemoryRequirements q = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = ci};
    VkMemoryRequirements2 r = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(vk_ctx()->device, &q, &r);
    return r.memoryRequirements.size;
}

// UNDEFINED -> GENERAL; new depth / stencil images start cleared (LAS textures.c: the game copies depth
// without stencil, and the stencil of those images must not be whatever the memory held).
static void init_image(Tex *e, VkCommandBuffer cb) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image, .subresourceRange = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    if (e->f.depth || e->f.stencil) {
        VkClearDepthStencilValue zero = {0.0f, 0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearDepthStencilImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else if (vk_fix_zero() && !e->bound_to && !e->pub.mirrored && !format_compressed(e->f.vk)) {
        // GPU-only storage that got memory of its own starts at zero, as the pool it stands for (vk_memory.c
        // alloc_physical). Images aliasing pool memory are not cleared: that would erase what other textures rendered
        // there; mirrored images are uploaded from their storage on first use.
        VkClearColorValue zero = {0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearColorImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    }
}

// AC_SHADOW_SCALE (1 to 4; 2 by default, 1 in the Steam Deck profile): the game's shadow maps are made this many times
// larger on each side. The rule for DEPTH16 square arrays comes from the TotK Recomp, whose game renders its sun cascades
// into a 960x960 array of 4 layers and its static shadows into 1440x1440 and 720x720 arrays of 2; Astral Chain's sun
// shadow map is the 2048x2048 one shadow_scale_of describes. Both are made
// larger here: render passes into them scale the game's viewports and scissors (vk_exec.c resolve_rt, pass_scale),
// copies scale their regions (vk_exec_xfer.c sub_resolve), and shaders sample them with normalised coordinates, so the
// game sees nothing of it. A map whose storage the CPU writes (mirrored) or that aliases pool memory keeps its size,
// as its texels must match the guest layout.
static int shadow_scale(void) {
    static int scale = -1;
    if (scale < 0) {
        int s = cfg_int("AC_SHADOW_SCALE");
        s = s < 1 ? 1 : s > 4 ? 4 : s;
        rt_log("vk: shadow map scale %d", s);
        scale = s;
    }
    return scale;
}

// Two kinds of map are made larger. The DEPTH16 square arrays of 2 to 4 layers described above. And Astral Chain's sun
// shadow map on 1.0.1: one 2048x2048 DEPTH24_STENCIL8 2D texture, rendered in a depth-only pass and read by the
// full-screen pass that makes the screen-space sun shadow mask (frame 1266 of build/v101/dump: passes 0006 and 0007).
// That shader reads raw depth with taps placed in UV, so a larger map gives finer shadow edges with the same softness,
// as the former ssaa setting did when it enlarged this map along with the screen. The scene depth has the same format
// but the screen's 16:9 shape, and a render pass mixing an enlarged target with one of the game's size would not fit
// (vk_exec.c resolve_rt logs one), hence the exact size.
static int shadow_scale_of(const nvn_texdesc *d, const TexVkFormat *f, uint32_t samples, int mirrored) {
    int scale = shadow_scale();
    if (scale == 1 || mirrored || samples > 1 || !f->depth) return 1;
    int sun = d->format == NVN_FORMAT_DEPTH24_STENCIL8 && d->target == NVN_TEXTURE_TARGET_2D && d->width == 2048 &&
              d->height == 2048 && d->levels <= 1;
    int cascades = d->format == NVN_FORMAT_DEPTH16 && d->target == NVN_TEXTURE_TARGET_2D_ARRAY && d->width == d->height &&
                   d->width >= 256 && d->depth >= 2 && d->depth <= 4;
    if (!sun && !cascades) return 1;
    uint32_t most = vk_ctx()->props.limits.maxImageDimension2D;
    return d->width * (uint32_t)scale <= most ? scale : 1;
}

// AC_TEX_MIP_SKIP (0 = off, the default; the Steam Deck profile sets 1024): an ASTC texture whose level 0 has that
// larger side or more is made without its level 0, three quarters of its memory. At the Deck's 1280x800 a material texture
// of 1024 texels or more is seldom drawn large enough to sample its level 0, so the picture loses little; in the
// deck profile's measurement those textures held about 370 MiB of the 900 MiB of texture images, and the 512 MiB
// texture cache, which was dropping and decoding again hundreds of them a minute, holds more of them.
// The image's level i is the game's level i + 1 (VkTexImage.skip, vk_tex_level). Sampling with derivatives picks the
// right level by itself, as the image is half the size; views and writes translate the game's levels (vk_exec_draw.c
// get_view, image_texels_at); integer texel fetches and size queries go through the translator's render scale of the
// texture, 0.5 (vk_exec_draw.c support_scales), so the game's shaders still see the game's size. Explicit LODs and level
// fetches above 0 land one level smaller, which is a little softer and the same place in the texture.
// Only ASTC 2D and 2D-array textures in own memory (not aliasing a pool), two levels or more, single-sampled and not
// enlarged: the game never renders into ASTC or copies it (vk_exec_xfer.c skips such copies), so their content only
// comes from storage uploads and WriteTexels.
int vk_tex_mip_skip(void) {
    static int side = -1;
    if (side < 0) {
        int v = cfg_int("AC_TEX_MIP_SKIP");
        side = v < 0 ? 0 : v;
        if (side) rt_log("vk: ASTC textures of %d texels or more on their larger side leave out level 0 (AC_TEX_MIP_SKIP)", side);
    }
    return side;
}

static uint32_t mip_skip_of(const nvn_texdesc *d, uint64_t addr, uint32_t w, uint32_t h, uint32_t levels,
                            uint32_t samples, float scale) {
    int side = vk_tex_mip_skip();
    if (!side || d->format < 0x79 || d->format > 0x94 || levels < 2 || samples > 1 || scale > 1.0f) return 0;
    if (d->target != NVN_TEXTURE_TARGET_2D && d->target != NVN_TEXTURE_TARGET_2D_ARRAY) return 0;
    if (nvn_tex_is_linear(d) || nvn_gpu_only(addr) || (w > h ? w : h) < (uint32_t)side) return 0;
    return 1;
}

// ---------------------------------------------------------------- uploads ahead of the first draw (AC_TEX_PREFETCH)
// A texture's first upload used to happen at the first draw that binds it, all of it in that frame: on the render
// thread the alpha scan of AC_ASTC_BC=bc1 (2 to 3 ms per MiB of an opaque texture, 85% of the first uploads' CPU time),
// the image and the staging copy, and on the GPU the ASTC decode and BC encode, about 1 ms per million texels on an
// Arc 140T. An enemy spawning with 72 new textures (70 M texels) held the render thread 52 ms and the GPU about as
// long: the freeze the "Texture uploads" counter of the overlay points at. The game makes its textures known long
// before it draws them: of 468 ASTC textures of a save load and a walk, every one was registered in its descriptor
// slot 31 frames or more before its first binding, 331 of them over 300 frames, with the storage already written
// (one was written again in between). So the work is done between the two:
//   - vk_texture_register queues every mirrored 2D ASTC texture that has no image yet (pre_add);
//   - a background thread runs the alpha scan (pre_main), below normal priority, and keeps the verdict with the page
//     generation taken before it read the storage;
//   - each submit, the render thread creates and uploads queued textures through the usual path (pre_step: entry_of,
//     make_fresh_snap) up to a budget of texels, so the GPU decodes a little every frame instead of everything in the
//     frame of the first draw. create_tex takes the worker's verdict when no page was written since (pre_take).
// A texture drawn before its turn uploads at the draw as before, with the worker's verdict if it is ready. Storage
// written after the upload is uploaded again at the binding like any rewritten texture. The textures made ahead count
// against the texture cache like the others, and nothing is made ahead once the cache budget is reached or the device
// memory is 80% used (PRE_VRAM_PCT): an upload ahead never pushes out a texture in use. They are the cost: textures
// the game registered and does not draw hold memory (314 MiB more as BC in play on a save, 1161 to 1475 MiB), up
// to the cache budget; from there the cache's own rule decides (sweep_cache drops the least recently used entries,
// the ones made ahead and never drawn first, cache_before), and the queue waits for room with its scans done.
// Dropping the oldest undrawn ones for newer registrations was tried and undone: the textures registered at boot
// are drawn in play minutes later, and went first. BC textures are left to the draw: their upload is a copy (0.06 ms
// each on the render thread in the same measurement) and making all of them ahead held another 154 MiB.
// An entry the cache budget or the VRAM pressure drops leaves its verdict behind (pre_keep), so its return at the
// next binding does not scan again. AC_TEX_PREFETCH=0 turns all of it off.
#define PRE_MIN_TEXELS (3ull << 19)   // per submit: 1.5 M texels, one 1024x1024 texture with its levels
#define PRE_MAX_TEXELS (8ull << 20)   // per submit at most, when the queue is long (a load)
#define PRE_SPREAD 60                 // submits a queue is spread over between the two bounds
#define PRE_MAX_ITEMS 8               // per submit: each costs the render thread an image and a staging copy
#define PRE_QUEUE_MAX 8192            // textures waiting; a register past it is not queued
#define PRE_KEPT_MAX 16384            // verdicts kept of dropped entries
#define PRE_VRAM_PCT 80               // percent of the device-local budget in use above which nothing is made ahead
#define PRE_COPY_SHARE 8              // a texel that is only copied (native ASTC) counts as 1/8 of a decoded one
#define PRE_SLICE (64u << 10)         // bytes the worker scans per hold of pre_mem_lock
#define PRE_RESCANS 3                 // scans of a storage that keeps being written before it is given up
// PRE_KEPT: the verdict of a dropped entry (pre_keep); in pre_map only, on no list
enum { PRE_NEW, PRE_SCAN, PRE_BUSY, PRE_READY, PRE_KEPT };
typedef struct Pre {
    TexKey key;
    tex_t t;
    uint64_t size, texels;    // storage bytes; texels of every level and layer the image will have
    uint64_t cost, bytes;     // pre_step: the texels counted against the submit's budget, the image's estimated bytes
    uint8_t state;
    uint8_t cancel;           // dropped: out of pre_map, and whoever takes it off its list (or the worker, PRE_BUSY) frees it
    uint8_t bw, bh, rescans;
    int8_t verdict;           // the alpha scan: -1 not scanned, 0 opaque, 1 alpha-carrying
    uint32_t gen;             // page generation taken before the scan read the storage
    struct Pre *next;
} Pre;
typedef struct { Pre *head, *tail; } PreList;
static PlatMutex pre_lock = PLAT_MUTEX_INIT;       // the lists, pre_map and the Pre fields; taken after tex_lock
static PlatMutex pre_mem_lock = PLAT_MUTEX_INIT;   // held by the worker while it reads guest storage (one slice)
static PlatCond pre_cv = PLAT_COND_INIT;
static nvn_hmap pre_map = {.klen = sizeof(TexKey)};
static PreList pre_new, pre_scan, pre_ready;
static volatile int32_t pre_count;   // entries of pre_map
static int pre_kept;                 // PRE_KEPT ones among them (pre_lock)
static uint64_t pre_texels;          // texels of the entries of pre_map (pre_lock)
static int pre_on = -1;
static int pre_in_step;              // the render thread is uploading ahead (tex_lock)
// Since the last perf line (tex_lock): uploads ahead, the worker's scans used, ASTC first uploads made at a draw.
static uint64_t pre_n_ahead, pre_tx_ahead, pre_n_scan_used, pre_n_draw, pre_tx_draw;
static uint32_t mip_skip_of(const nvn_texdesc *d, uint64_t addr, uint32_t w, uint32_t h, uint32_t levels,
                            uint32_t samples, float scale);

static void pre_push(PreList *l, Pre *p) {
    p->next = NULL;
    if (l->tail) l->tail->next = p;
    else l->head = p;
    l->tail = p;
}

static Pre *pre_pop(PreList *l) {
    Pre *p = l->head;
    if (p && !(l->head = p->next)) l->tail = NULL;
    return p;
}

// pre_lock held: p leaves pre_map. The caller either holds p (off its list) or marks it cancelled.
static void pre_unmap(Pre *p) {
    if (nvn_hm_get(&pre_map, &p->key) != p) return;
    nvn_hm_del(&pre_map, &p->key);
    pre_texels -= p->texels;
    if (p->state == PRE_KEPT) pre_kept--;
    __atomic_store_n(&pre_count, (int32_t)pre_map.n, __ATOMIC_RELEASE);
}

// The scan thread: one texture at a time, in PRE_SLICE pieces so that vk_texture_range_gone (the storage goes back
// to the guest) waits for one piece at most.
static void pre_main(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    for (;;) {
        plat_mutex_lock(&pre_lock);
        Pre *p;
        while (!(p = pre_pop(&pre_scan))) plat_cond_wait(&pre_cv, &pre_lock);
        if (p->cancel) {
            plat_mutex_unlock(&pre_lock);
            free(p);
            continue;
        }
        p->state = PRE_BUSY;
        plat_mutex_unlock(&pre_lock);
        uint32_t gen = 0;
        int verdict = 0, tracked = 1;
        for (uint64_t off = 0; off < p->size && !verdict && tracked; off += PRE_SLICE) {
            plat_mutex_lock(&pre_mem_lock);
            int gone = __atomic_load_n(&p->cancel, __ATOMIC_ACQUIRE);
            if (!gone) {
                // Stamped before the first byte is read: a later write carries a newer generation (create_tex).
                if (!off && vk_mem_changed(p->key.addr, p->size, 0, &gen) < 0) tracked = 0;
                uint64_t n = p->size - off < PRE_SLICE ? p->size - off : PRE_SLICE;
                if (tracked) verdict = astc_storage_alpha_possible((const uint8_t *)p->t.mem + off, n, p->bw, p->bh);
            }
            plat_mutex_unlock(&pre_mem_lock);
            if (gone) break;
        }
        plat_mutex_lock(&pre_lock);
        if (p->cancel || !tracked || !gen) {   // untracked storage: no way to tell later whether the verdict holds
            if (!p->cancel) pre_unmap(p);
            plat_mutex_unlock(&pre_lock);
            free(p);
            continue;
        }
        p->gen = gen, p->verdict = (int8_t)verdict, p->state = PRE_READY;
        pre_push(&pre_ready, p);
        plat_mutex_unlock(&pre_lock);
    }
}

// vk_texture_register (tex_lock held, the game's thread): t is queued when it is a texture the upload ahead handles.
static void pre_add(const tex_t *t) {
    if (pre_on < 0) {
        pre_on = cfg_bool("AC_TEX_PREFETCH");
        if (pre_on && !plat_thread_spawn(pre_main, NULL, "vk tex scan", 0)) pre_on = 0;
        rt_log("vk: textures are %s (AC_TEX_PREFETCH)",
               pre_on ? "scanned on a background thread and uploaded ahead of their first draw" : "uploaded at their first draw");
    }
    // ASTC only: the game's static assets, which it never renders into, so an image made early holds what the draw
    // would have uploaded. Other mirrored textures (CPU-written pictures, render targets with CPU storage) are left to
    // the draw path and its aliases.
    const nvn_texdesc *d = &t->d;
    if (!pre_on || !t->mem || d->format < 0x79 || d->format > 0x94 || nvn_tex_is_linear(d) || d->samples > 1) return;
    if (d->target != NVN_TEXTURE_TARGET_2D && d->target != NVN_TEXTURE_TARGET_2D_ARRAY &&
        d->target != NVN_TEXTURE_TARGET_CUBEMAP && d->target != NVN_TEXTURE_TARGET_CUBEMAP_ARRAY)
        return;
    if (!tex_mirrored(t)) return;
    TexKey k;
    make_key(t, &k);
    if (nvn_hm_get(&by_key, &k)) return;
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    if (!L.size) return;
    Pre *p = calloc(1, sizeof *p);
    if (!p) return;
    p->key = k, p->t = *t, p->size = L.size, p->verdict = -1, p->state = PRE_NEW;
    uint32_t levels = umax(d->levels, 1);
    for (uint32_t lv = mip_skip_of(d, k.addr, umax(d->width, 1), umax(d->height, 1), levels, 1, 1.0f); lv < levels && lv < 16; lv++)
        p->texels += (uint64_t)mip(d->width, lv) * mip(d->height, lv) * umax(L.layers, 1);
    plat_mutex_lock(&pre_lock);
    if (nvn_hm_get(&pre_map, &k) || (int)pre_map.n - pre_kept >= PRE_QUEUE_MAX) {   // a kept verdict stays as it is
        plat_mutex_unlock(&pre_lock);
        free(p);
        return;
    }
    nvn_hm_put(&pre_map, &k, p);
    pre_texels += p->texels;
    __atomic_store_n(&pre_count, (int32_t)pre_map.n, __ATOMIC_RELEASE);
    pre_push(&pre_new, p);
    plat_mutex_unlock(&pre_lock);
}

// The texture of key k is created, released or gone: its queue entry goes. Returns the worker's verdict when it is
// ready (0 or 1, *gen the generation it holds from), else -1.
static int pre_take(const TexKey *k, uint32_t *gen) {
    if (!__atomic_load_n(&pre_count, __ATOMIC_ACQUIRE)) return -1;
    int v = -1;
    plat_mutex_lock(&pre_lock);
    Pre *p = nvn_hm_get(&pre_map, k);
    if (p) {
        int kept = p->state == PRE_KEPT;
        if (p->state == PRE_READY || kept) v = p->verdict, *gen = p->gen;
        pre_unmap(p);
        if (kept) free(p);   // on no list and not the worker's: nobody else frees it
        else __atomic_store_n(&p->cancel, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&pre_lock);
    return v;
}

// e is about to be dropped by the cache budget or the VRAM pressure while the game still has the texture: the alpha
// verdict its image was made from stays, with the generation of e's last storage check (nothing was written up to
// it that the image did not get, and an upload that found another verdict trades the entry, bc_flip), for the
// create_tex of its return.
static void pre_keep(const Tex *e) {
    if (pre_on <= 0 || !e->f.astc_bc || vk_astc_bc() != 2 || e->bc_flip || !e->uploaded || !e->wgen) return;
    if (e->t.d.target == NVN_TEXTURE_TARGET_3D) return;
    Pre *p = calloc(1, sizeof *p);
    if (!p) return;
    p->key = e->key, p->t = e->t, p->size = e->t.L.size, p->state = PRE_KEPT;
    p->verdict = e->f.astc_bc != 2, p->gen = e->wgen;
    plat_mutex_lock(&pre_lock);
    if (nvn_hm_get(&pre_map, &p->key) || pre_kept >= PRE_KEPT_MAX) {
        plat_mutex_unlock(&pre_lock);
        free(p);
        return;
    }
    nvn_hm_put(&pre_map, &p->key, p);
    pre_kept++;
    __atomic_store_n(&pre_count, (int32_t)pre_map.n, __ATOMIC_RELEASE);
    plat_mutex_unlock(&pre_lock);
}

// Every queue entry whose storage lies in [lo, lo + size) goes, and the worker is out of that storage on return.
// all: every entry (shutdown).
static void pre_range_gone(uint64_t lo, uint64_t size, int all) {
    if (!__atomic_load_n(&pre_count, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&pre_lock);
    for (uint32_t i = 0; i < pre_map.cap;) {
        Pre *p = pre_map.vals[i];
        if (p && (all || (p->key.addr + p->size > lo && p->key.addr < lo + size))) {
            int kept = p->state == PRE_KEPT;
            pre_unmap(p);   // moves later entries of the probe chain into slot i: look at it again
            if (kept) free(p);
            else __atomic_store_n(&p->cancel, 1, __ATOMIC_RELEASE);
        } else {
            i++;
        }
    }
    plat_mutex_unlock(&pre_lock);
    plat_mutex_lock(&pre_mem_lock);   // a slice in progress ends; the next one sees cancel
    plat_mutex_unlock(&pre_mem_lock);
}

// defer_init: the caller records the image's first transition (init_image or alias_copy).
static Tex *create_tex(const tex_t *t, const TexKey *k, int defer_init) {
    static volatile int32_t n_unmapped, n_unsupported, n_failed, n_unbound, n_thick;
    VkCtx *c = vk_ctx();
    const nvn_texdesc *d = &t->d;
    if (!c || d->target == NVN_TEXTURE_TARGET_BUFFER) return NULL;
    TexVkFormat f;
    if (!tex_vk_format(d->format, &f)) {
        log_limited(&n_unmapped, "texture format %#x has no Vulkan format: %s", d->format, f.reason);
        return NULL;
    }
    if (!tex_vk_format_supported(&f)) {
        log_limited(&n_unsupported, "texture format %#x: VkFormat %d lacks features %#x", d->format, (int)f.vk, (unsigned)f.need);
        return NULL;
    }
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    // AC_ASTC_BC=bc1 (the default): the alpha scan of the raw storage picks this texture's recompression
    // before the image exists. A provably opaque one — every 16-byte unit of the storage decodes alpha 255,
    // astc_storage_alpha_possible (tex_astc.c) — takes BC1 at half of BC3's size with the same colour
    // quality (the BC3 colour block is a BC1 block); the rest keep BC3. 3D targets stay BC3: the scan, like
    // the decoder, judges 2D block modes only. A storage not yet written scans as whatever it holds — zeros
    // are error blocks (opaque), garbage usually alpha-carrying — and the re-scan at every upload (upload)
    // trades the entry in for the other format when the verdict changes.
    int bc_verdict = -1;   // -1 not scanned (no BC1 mode, 3D target, no storage), 0 opaque, 1 alpha-carrying
    uint64_t bc_scan_ns = 0;
    uint32_t bc_gen = 0;   // page generation taken just before the scan; 0 when the storage is not tracked
    // The scan thread's verdict for this texture, if it finished one (AC_TEX_PREFETCH); its queue entry goes either way.
    uint32_t pre_gen = 0, pre_now;
    int pre_verdict = pre_take(k, &pre_gen);
    if (f.astc_bc && vk_astc_bc() == 2 && t->mem && d->target != NVN_TEXTURE_TARGET_3D) {
        uint64_t s0 = plat_now_ns();
        if (pre_verdict >= 0 && vk_mem_changed(k->addr, L.size, pre_gen, &pre_now) == 0) {
            // no page written since the worker stamped pre_gen and read the storage: its verdict describes these bytes
            bc_verdict = pre_verdict, bc_gen = pre_gen;
            pre_n_scan_used++;
        } else {
            // Stamped before the scan reads the bytes: a write after this point carries a newer generation, so the
            // first upload can tell whether this verdict still describes the storage (upload, bc_verdict_current).
            if (vk_mem_changed(k->addr, L.size, 0, &bc_gen) < 0) bc_gen = 0;
            bc_verdict = astc_storage_alpha_possible(t->mem, L.size, f.astc_bw, f.astc_bh);
        }
        bc_scan_ns = plat_now_ns() - s0;
        if (!bc_verdict) {
            TexVkFormat g = f;
            g.vk = f.astc_srgb ? VK_FORMAT_BC1_RGB_SRGB_BLOCK : VK_FORMAT_BC1_RGB_UNORM_BLOCK;
            g.astc_bc = 2;
            g.note = "no native ASTC: opaque after the alpha scan (tex_astc.c), encoded to BC1 on the GPU "
                     "(vk_astc.c, AC_ASTC_BC=bc1)";
            if (tex_vk_format_supported(&g)) f = g;   // a device that cannot sample BC1 keeps BC3
        }
    }
    uint32_t w = umax(d->width, 1), h = umax(d->height, 1), depth = 1, layers = 1;
    uint32_t samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    VkImageType type = VK_IMAGE_TYPE_2D;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: type = VK_IMAGE_TYPE_1D; h = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: type = VK_IMAGE_TYPE_1D; h = 1; layers = L.layers; break;
    case NVN_TEXTURE_TARGET_3D: type = VK_IMAGE_TYPE_3D; depth = umax(d->depth, 1); break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
    case NVN_TEXTURE_TARGET_CUBEMAP: case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: layers = L.layers; break;
    default: break;
    }
    int shadow = shadow_scale_of(d, &f, samples, tex_mirrored(t));
    float scale = (float)shadow;
    w =(uint32_t)lroundf((float)w * scale), h = (uint32_t)lroundf((float)h * scale);
    uint32_t levels = umax(d->levels, 1), most = 1;
    for (uint32_t s = umax(umax(w, h), depth); s > 1; s >>= 1) most++;
    if (levels > most) levels = most;
    if (samples > 1) levels = 1;
    uint32_t skip = mip_skip_of(d, k->addr, w, h, levels, samples, scale);
    if (skip) w = umax(w >> skip, 1), h = umax(h >> skip, 1), levels -= skip;
    VkFormatFeatureFlags ff = features_of(f.vk);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    int ds = f.depth || f.stencil;
    if (ds && (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!ds && (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && samples == 1)
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    // contact shadows (vk_contact.c) darken the game's B10G11R11 scene colour in place from a compute shader
    if (f.vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && samples == 1 &&
        (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && w >= 640 && vk_contact_strength())
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateFlags flags = ds ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;   // format-reinterpreting views
    // Cube compatible only for cube targets: the replay makes cube views of nothing else (vk_exec_draw.c get_view checks
    // the image's own view type). The flag on every square array of 6+ layers changed how AMD's Windows driver lays such
    // an image out, and terrain material arrays sampled through it came out in rainbow blocks that moved with the mip
    // level (Savage Wind Recomp issue #59, BotW on an RX 580 and a 7800 XT).
    if (type == VK_IMAGE_TYPE_2D && layers >= 6 && w == h && samples == 1 &&
        (d->target == NVN_TEXTURE_TARGET_CUBEMAP || d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY))
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // NVN renders into a 3D texture slice by slice (a view's min layer is the depth plane, nvn_mtl.m), e.g. the
    // grading LUT of the menu background blur. Vulkan needs a 2D view of the slice for that.
    if (type == VK_IMAGE_TYPE_3D && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = type, .format = f.vk,
                            .extent = {w, h, depth}, .mipLevels = levels, .arrayLayers = layers,
                            .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    int bw, bh, block_views = 0;
    // AC_BLOCK_VIEWS_3D (default 0): a 3D BC image also gets block views. On RDNA2 cards (driver 26.10.1, issue #5)
    // the 32x32x32 BC4 volume the game creates at the logo, made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE,
    // took a GPU write just past the end of its memory: device lost at the first submit after it when the image ended
    // its memory block. Draws into a 3D BC image go through a stand-in instead (vk_exec.c bc_target).
    static int bv3d = -1;
    if (bv3d < 0) {
        bv3d = rt_knob("BLOCK_VIEWS_3D", "0");
        rt_log("vk: 3D BC textures %s (AC_BLOCK_VIEWS_3D=%d)", bv3d ? "get block views (renderable)" : "have no block views",
               bv3d);
    }
    if (!ds && samples == 1 && f.vk == f.ideal && nvn_fmt_bpb(d->format, &bw, &bh) && bw == 4 && bh == 4 &&
        (type != VK_IMAGE_TYPE_3D || bv3d)) {
        VkImageCreateInfo bc = ci;   // BC formats: renderable through views of their blocks (block_views_ok)
        bc.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        bc.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        if (type == VK_IMAGE_TYPE_3D) bc.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        VkImageUsageFlags hidden = 0;   // creation usage the replay never uses (views restrict theirs, vk_exec.c)
        int ok = block_views_ok(&bc);
        if (ok && type == VK_IMAGE_TYPE_3D && !image_size(&bc)) {
            // STORAGE usage keeps RADV on GFX9 from the thin layout (image_size), so agl's BC4 encoder still renders
            // into the volume; else the image is made without block views and its encoder passes are left out
            bc.usage |= VK_IMAGE_USAGE_STORAGE_BIT, hidden = VK_IMAGE_USAGE_STORAGE_BIT;
            ok = block_views_ok(&bc) && image_size(&bc);
            log_limited(&n_thick, "texture fmt %#x %ux%ux%u: the device cannot lay out a 3D image with block views; %s",
                        d->format, w, h, depth, ok ? "made with storage usage (a thick layout)" : "made without them");
        }
        if (ok) ci = bc, usage = bc.usage & ~hidden, flags = bc.flags, block_views = 1;
    }
    // Feedback loops (vk.h VkCtx.feedback): every attachable image may be sampled by a draw of a pass that renders it,
    // so it gets the usage now; adding it when a loop shows up would mean recreating the image (that lost the
    // grass tile maps once).
    int fb_usage = c->feedback == VK_FEEDBACK_LAYOUT && !block_views &&
                   (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));
    if (fb_usage) ci.usage = usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    Tex *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("vk: out of memory for a texture");
    e->key = *k;
    e->t = *t;
    e->f = f;
    e->last_use = submit_epoch;
    e->pub.addr = k->addr;
    e->bc_gen = bc_verdict >= 0 ? bc_gen : 0;
    VkImage img = VK_NULL_HANDLE;
    if (scale <= 1.0f && nvn_gpu_only(k->addr)) {   // storage in a physical or virtual pool: alias the pool memory (section 3.2)
        if (vkCreateImage(c->device, &ci, NULL, &img) == VK_SUCCESS && !bind_pool_memory(c, e, img, L.size)) {
            vkDestroyImage(c->device, img, NULL);
            img = VK_NULL_HANDLE;
            log_limited(&n_unbound, "texture %#llx (%ux%u fmt %#x) cannot alias its pool memory; own memory instead",
                        (unsigned long long)k->addr, w, h, d->format);
        }
    }
    if (!img) {
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkResult r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
        if (r != VK_SUCCESS) {
            log_limited(&n_failed, "image creation failed (%d): nvn fmt %#x %ux%ux%u target %d levels %u layers %u", (int)r,
                        d->format, w, h, depth, d->target, levels, layers);
            free(e);
            return NULL;
        }
    }
    if (e->alloc) {   // own memory: counted against the cache budget (sweep_cache)
        VmaAllocationInfo mi;
        vmaGetAllocationInfo(c->vma, e->alloc, &mi);
        e->bytes = mi.size;
        own_bytes += e->bytes;
        static int scaled_logs;
        if (scale > 1.0f && scaled_logs < 8)   // what an enlarged shadow map costs, beside the pool storage it leaves
            scaled_logs++, rt_log("vk: shadow map %ux%u fmt %#x made %ux%u (AC_SHADOW_SCALE): %.1f MiB of its own",
                                  d->width, d->height, d->format, w, h, (double)mi.size / 1048576.0);
    }
    // The recompression accounting of the perf line (vk_texture_new_submit): the scan's cost, the format
    // each entry took and where the image lives — the numbers that sized the BC1 change and watch it after.
    if (f.astc_bc && vk_astc_bc() == 2) {
        if (bc_verdict >= 0) {
            sc_n++, sc_bytes += L.size, sc_ns += bc_scan_ns;
            if (f.astc_bc == 2) sc_op_n++, sc_op_bytes += L.size;
            else sc_al_n++, sc_al_bytes += L.size;
        } else {
            sc_skip_n++;
        }
        if (e->bound_to) sc_bound_n++, sc_bound_bytes += L.size;
        else if (e->alloc) sc_own_n++, sc_own_bytes += e->bytes;
    }
    VkTexImage *p = &e->pub;
    p->image = img;
    p->format = f.vk;
    p->type = type;
    p->usage = usage;
    p->aspect = aspect_of(&f);
    p->width = w, p->height = h, p->depth = depth, p->levels = levels, p->layers = layers, p->samples = samples;
    p->skip = (uint8_t)skip;
    p->view_type = type == VK_IMAGE_TYPE_1D ? (d->target == NVN_TEXTURE_TARGET_1D_ARRAY ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                 : type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                 : layers > 1 || d->target == NVN_TEXTURE_TARGET_2D_ARRAY || d->target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    p->alpha_one = f.alpha_one;
    memcpy(p->chan, f.chan, sizeof p->chan);
    p->swz[0] = (uint8_t)d->swizzle_r, p->swz[1] = (uint8_t)d->swizzle_g;
    p->swz[2] = (uint8_t)d->swizzle_b, p->swz[3] = (uint8_t)d->swizzle_a;
    if (!(p->swz[0] | p->swz[1] | p->swz[2] | p->swz[3]))   // descriptors built without nvnTextureBuilderSetDefaults
        p->swz[0] = NVN_TEXTURE_SWIZZLE_R, p->swz[1] = NVN_TEXTURE_SWIZZLE_G, p->swz[2] = NVN_TEXTURE_SWIZZLE_B,
        p->swz[3] = NVN_TEXTURE_SWIZZLE_A;
    p->mirrored = (uint8_t)tex_mirrored(t);
    p->bound = e->bound_to != NULL;
    p->block_views = (uint8_t)block_views;
    p->fb_usage = (uint8_t)fb_usage;
    p->scale = scale;

    if (!defer_init) {
        init_image(e, vk_upload_begin());
        vk_upload_end();
    }
    return e;
}

static int owned_lower(uint64_t a) {
    int lo = 0, hi = nowned;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (owned[m]->pub.addr < a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static void own_add(Tex *e) {
    if (e->listed) return;
    plat_mutex_lock(&owned_lock);
    if (nowned == cowned) {
        cowned = cowned ? 2 * cowned : 256;
        if (!(owned = realloc(owned, sizeof *owned * (size_t)cowned))) rt_fatal("vk: out of memory for the owned-texture list");
    }
    int i = owned_lower(e->pub.addr);
    memmove(owned + i + 1, owned + i, sizeof *owned * (size_t)(nowned - i));
    owned[i] = e;
    __atomic_store_n(&nowned, nowned + 1, __ATOMIC_RELEASE);
    if (e->t.L.size > max_owned) max_owned = e->t.L.size;
    e->listed = 1;
    plat_mutex_unlock(&owned_lock);
}

static void own_remove(Tex *e) {
    if (!e->listed) return;
    plat_mutex_lock(&owned_lock);
    for (int i = owned_lower(e->pub.addr); i < nowned && owned[i]->pub.addr == e->pub.addr; i++)
        if (owned[i] == e) {
            memmove(owned + i, owned + i + 1, sizeof *owned * (size_t)(nowned - i - 1));
            __atomic_store_n(&nowned, nowned - 1, __ATOMIC_RELEASE);
            break;
        }
    e->listed = 0;
    plat_mutex_unlock(&owned_lock);
}

static void rel_append(Tex *e) {
    e->rel_next = NULL;
    e->rel_prev = rel_tail;
    if (rel_tail) rel_tail->rel_next = e;
    else rel_head = e;
    rel_tail = e;
}

static void rel_unlink(Tex *e) {
    if (e->rel_prev) e->rel_prev->rel_next = e->rel_next;
    else rel_head = e->rel_next;
    if (e->rel_next) e->rel_next->rel_prev = e->rel_prev;
    else rel_tail = e->rel_prev;
    e->rel_prev = e->rel_next = NULL;
}

static void bc_flip_remove(Tex *e);   // destroy_tex calls it; defined with the trade list below

// tex_reclaim's purge, while it destroys: the timeline value its entries were last used before (vk_defer_image_after),
// so their memory comes back without waiting for the submission being recorded; 0 otherwise.
static uint64_t purge_tl;

static void destroy_tex(Tex *e) {
    own_remove(e);
    if (e->bytes) own_bytes -= e->bytes;
    tex_gen_bump(e->key.addr);
    nvn_hm_del(&by_key, &e->key);
    Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head == e) {
        if (e->next_at) nvn_hm_put(&by_addr, &e->key.addr, e->next_at);
        else nvn_hm_del(&by_addr, &e->key.addr);
    } else {
        for (Tex *p = head; p; p = p->next_at)
            if (p->next_at == e) { p->next_at = e->next_at; break; }
    }
    if (e->released) nreleased--, rel_unlink(e);
    if (e->bc_flip) bc_flip_remove(e);
    vk_exec_image_gone(e->pub.image);
    if (purge_tl) vk_defer_image_after(e->pub.image, e->alloc, purge_tl);
    else vk_defer_image(e->pub.image, e->alloc);
    free(e->snap);
    free(e);
}

// A recompressed entry whose storage's alpha verdict turned against its image's format (upload): destroyed at
// the next submit (sweep_flip), so the next lookup recreates it in the other format from a fresh scan. Until
// then it answers lookups with the image it has — one submit of the old content, like a released entry being
// replaced. upload and make_fresh leave a traded entry alone (bc_flip), so nothing writes into an image whose
// format the content no longer matches.
static void bc_trade(Tex *e) {
    static volatile int32_t n_trade;
    if (e->bc_flip) return;
    if (nbc_flip == cbc_flip) {
        cbc_flip = cbc_flip ? 2 * cbc_flip : 16;
        if (!(bc_flip_list = realloc(bc_flip_list, sizeof *bc_flip_list * (size_t)cbc_flip)))
            rt_fatal("vk: out of memory for the BC1 trade list");
    }
    bc_flip_list[nbc_flip++] = e;
    e->bc_flip = 1;
    log_limited(&n_trade, "texture %#llx fmt %#x %ux%u: its content's alpha verdict changed; the entry is "
                "recreated as %s", (unsigned long long)e->pub.addr, e->t.d.format, e->t.d.width, e->t.d.height,
                e->f.astc_bc == 2 ? "BC3" : "BC1");
}

static void bc_flip_remove(Tex *e) {   // destroy_tex: the traded entry dies, leave the list first
    for (int i = 0; i < nbc_flip; i++)
        if (bc_flip_list[i] == e) {
            memmove(bc_flip_list + i, bc_flip_list + i + 1, sizeof *bc_flip_list * (size_t)(nbc_flip - 1 - i));
            nbc_flip--;
            break;
        }
}

static void sweep_flip(void) {   // tex_lock held, each submit: recreate the traded entries' format
    while (nbc_flip) destroy_tex(bc_flip_list[0]);
}

// Released entries with another descriptor at the same address are replaced; past RELEASED_MAX released entries the
// oldest one goes at each release (it comes back on demand, a mirrored one re-uploaded from its storage). Astral Chain
// finalizes and initializes transient textures every frame, so a released entry is mostly taken back within frames.
// Dropping all of them at once when the count passed 4096 destroyed thousands of images in one frame: 333-422 ms
// frames every 40-50 s on a GTX 1050 Ti, then the textures still in use were created and uploaded again.
#define RELEASED_MAX 2048
static void sweep_released(void) {
    while (nreleased > RELEASED_MAX && rel_head) destroy_tex(rel_head);
}

// AC_TEX_VRAM_PRESSURE (percent of the device-local budget, default 90; 0 = off): above it, each submit drops up to
// PRESSURE_STEP of the oldest released entries, so a 4 GiB card does not keep RELEASED_MAX released images (each ASTC
// one decoded to RGBA8) while the heap spills into system memory. A few per submit, for the reason above. Only entries
// that come back intact are dropped: mirrored ones the GPU never wrote, re-uploaded from their storage. A render target
// lives only in its image; dropping it brought the texture back empty (black flashes in a Depths boss fight, ~170
// "left uninitialised" aliases a minute). Past the whole budget the cache itself gives way and the ASTC textures made
// from then on are BC3 (over_budget).
#define PRESSURE_STEP 32
static int vram_pressure = -1;
static uint64_t n_pressure_drops;

static void over_budget(uint64_t used, uint64_t budget);   // defined with the cache budget below

static void sweep_pressure(void) {   // tex_lock held
    if (vram_pressure < 0) {
        const char *s = cfg_get("AC_TEX_VRAM_PRESSURE");
        vram_pressure = s && *s ? atoi(s) : 90;
        if (vram_pressure < 0 || vram_pressure > 100) vram_pressure = 90;
        rt_log("vk: released textures under VRAM pressure: %s%d%%", vram_pressure ? "dropped above " : "off", vram_pressure);
    }
    if (!vram_pressure || submit_epoch & 7) return;   // the budget query every 8 submits
    uint64_t used, budget;
    int exact;
    if (!vk_mem_vram(&used, &budget, &exact)) return;
    if (used > budget) over_budget(used, budget);
    if (used <= budget / 100 * (uint64_t)vram_pressure) return;
    int dropped = 0;
    for (Tex *e = rel_head, *next; e && dropped < PRESSURE_STEP; e = next) {
        next = e->rel_next;
        if (!e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty) continue;
        pre_keep(e);
        destroy_tex(e);
        dropped++, n_pressure_drops++;
    }
}

// AC_TEX_CACHE_MB: own memory the texture images may take, in MiB, live and released together. Empty or 0: an eighth
// of the device budget, at least 512 MiB and at most 2 GiB; when the decoded ASTC textures stay RGBA8 (AC_ASTC_BC=0,
// the default on a discrete GPU with 6 GB or more), four to eight times BC's bytes, half of the budget, at most 8 GiB,
// so the textures of one area are not dropped and decoded again as the camera turns; negative: no budget. Past it,
// the least recently used entries that can come back intact are dropped, a few per submit, and uploaded again when the
// game binds them. The released sweep above bounds the cache's dead entries by count; this one bounds the memory of
// everything it holds. The VRAM pressure sweep (sweep_pressure) still drops released entries near the device budget.
#define CACHE_STEP 8
#define CACHE_AGE 90   // submit epochs unused before the budget may drop an entry (about 1.5 s at 60 submits a second)
static uint64_t cache_budget;   // tex_lock
static int cache_budget_read;

static void cache_budget_init(void) {   // tex_lock held
    if (cache_budget_read) return;
    cache_budget_read = 1;
    const char *s = cfg_value("AC_TEX_CACHE_MB");
    long mb = s && *s ? strtol(s, NULL, 10) : 0;
    if (mb > 0) {
        cache_budget = (uint64_t)mb << 20;
    } else if (mb < 0) {
        cache_budget = 0;
    } else {
        uint64_t used, budget;
        int exact;
        int rgba8 = vk_astc_mode() != VK_ASTC_NATIVE && !vk_astc_bc();
        cache_budget = vk_mem_vram(&used, &budget, &exact) && budget ? budget / (rgba8 ? 2 : 8) : (1024ull << 20);
        uint64_t most = rgba8 ? 8192ull << 20 : 2048ull << 20;
        if (cache_budget < (512ull << 20)) cache_budget = 512ull << 20;
        if (cache_budget > most) cache_budget = most;
    }
    rt_log("vk: texture cache: %llu MiB of images (AC_TEX_CACHE_MB)", (unsigned long long)(cache_budget >> 20));
}

static int cache_evictable(const Tex *e) {
    if (!e->bytes) return 0;
    // The same rule as sweep_pressure: only an image whose content the storage can give back. A mirrored one the GPU
    // owns or wrote is newer than its bytes, and one with a snapshot has no pages to re-upload from.
    if (!e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty || e->snap || e->wt_n) return 0;
    return e->last_use && submit_epoch - e->last_use > CACHE_AGE;
}

// Whether the cache budget drops a before b: the entries made ahead (AC_TEX_PREFETCH) that nothing has used yet
// first, then the least recently used. An upload ahead filled the cache to its budget on the Steam Deck profile
// (512 MiB); without this order a texture drawn 20 s ago went before one made 10 s ago that may never be drawn.
static int cache_before(const Tex *a, const Tex *b) {
    int ua = a->ahead && a->last_use == a->ahead, ub = b->ahead && b->last_use == b->ahead;
    return ua != ub ? ua : a->last_use < b->last_use;
}

static void sweep_cache(void) {   // tex_lock held
    cache_budget_init();
    if (!cache_budget || own_bytes <= cache_budget) return;
    for (int k = 0; k < CACHE_STEP && own_bytes > cache_budget; k++) {
        Tex *best = NULL;
        for (uint32_t i = 0; i < by_key.cap; i++) {
            Tex *e = by_key.vals[i];
            if (e && cache_evictable(e) && (!best || cache_before(e, best))) best = e;
        }
        if (!best) break;
        n_cache_drops++, n_cache_drop_bytes += best->bytes;
        pre_keep(best);
        destroy_tex(best);
    }
}

// Video memory is short by over bytes: the cache budget becomes what the cache holds less that, and stays there for the
// rest of the session, as the cache would fill the memory again. It never grows. 1 when it was lowered. tex_lock held.
//
// RECLAIM_MIN keeps a cache an area's textures fit in; below the Steam Deck profile's 512 MiB, so the budget can still
// be lowered there.
#define RECLAIM_MIN (256ull << 20)

static int cache_budget_shrink(uint64_t over) {
    cache_budget_init();
    uint64_t target = own_bytes > over ? own_bytes - over : 0;
    if (target < RECLAIM_MIN) target = RECLAIM_MIN;
    if (cache_budget && target >= cache_budget) return 0;
    cache_budget = target;
    return 1;
}

// Over the device's whole budget, seen by sweep_pressure before any allocation failed. Either the game's own textures
// grew past what the card has left (RGBA8 on a 6 GB card: 5490 MiB in use of a 5233 MiB budget in ordinary play, and a
// decode that found no memory at 5626), or another program took video memory after the game started (a recorder, a
// browser) and the system lowered the budget under what the game already holds. Past its budget the system moves the
// game's video memory to system memory, and a few hundred MiB further allocations fail. So before that the ASTC
// textures made from here on are BC3 instead of RGBA8 (vk_astc.c vk_astc_low_memory), and the cache budget becomes
// what the cache holds less the overshoot past the pressure threshold (cache_budget_shrink), so the sweep drops the
// entries nothing used lately and the upload ahead waits. Both stay for the rest of the session. The budget is lowered
// again only once the cache is down to the last one: while the sweep is still dropping entries, the device's overshoot
// is the one already subtracted. tex_lock held, vram_pressure on.
static void over_budget(uint64_t used, uint64_t budget) {
    static uint32_t n_logs;
    int bc3 = vk_astc_low_memory();
    cache_budget_init();
    uint64_t was = cache_budget, held = own_bytes;
    int lowered = (!cache_budget || own_bytes <= cache_budget) &&
                  cache_budget_shrink(used - budget / 100 * (uint64_t)vram_pressure);
    if ((bc3 || lowered) && n_logs++ < 16)
        rt_log("vk: video memory over its budget (%llu of %llu MiB in use): texture cache budget %llu -> %llu MiB (it "
               "holds %llu MiB)%s",
               (unsigned long long)(used >> 20), (unsigned long long)(budget >> 20), (unsigned long long)(was >> 20),
               (unsigned long long)(cache_budget >> 20), (unsigned long long)(held >> 20),
               bc3 ? ", new ASTC textures are BC3" : "");
}

static int reclaim_order(const void *a, const void *b) {   // tex_reclaim's purge: the order the budget drops entries in
    const Tex *x = *(Tex *const *)a, *y = *(Tex *const *)b;
    return cache_before(x, y) ? -1 : cache_before(y, x) ? 1 : 0;
}

// Video memory full: an ASTC decode found no room for its scratch buffer (vk_astc_record, VK_ASTC_NO_SCRATCH). Left
// undecoded, the image kept whatever its memory held and stayed so while the game used it, as the entry counted as
// uploaded: black characters and a floor showing another texture on a 6 GB card with RGBA8 textures (5626 MiB in use
// of a 5233 MiB budget). The caller calls this and records the decode again while it returns 1; at 0 it takes the
// scratch from system memory (vk_astc_record_host). Two steps, each at most once per submit, as each waits for the GPU
// and the textures of an area arriving in the same submit find the room the first one made:
//
// 1. vk_mem_reclaim alone: the deferred objects that are done give their memory back, first of all the scratch buffers
//    of the decodes this submit recorded before, which live until their upload batch completes. Nothing is dropped.
//    1 when something was freed.
// 2. The purge: the cache budget was too large for what the device has left beside the game's pools, so it loses the
//    device's overshoot (cache_budget_shrink): the bytes in use above the VRAM pressure threshold
//    (AC_TEX_VRAM_PRESSURE, 90% or the Steam Deck's 70%), and at least the failed decode's texels. Every entry that can
//    come back intact goes, least recently used first, down to the new budget; then vk_mem_reclaim waits for the GPU
//    to be done with their images and frees them. 1 when entries were dropped.
//
// From the first call on, the ASTC textures made are BC3 instead of RGBA8 (vk_astc.c vk_astc_low_memory). keep is the
// entry being uploaded, which may itself be old enough to qualify. Every other entry the caller may hold was used in
// this submit (last_use) and does not qualify either. tex_lock held, outside vk_upload_begin / vk_upload_end.
static int tex_reclaim(const Tex *keep, VkDeviceSize decode_bytes) {
    static uint32_t n_logs, flush_epoch, purge_epoch;   // submit_epoch is never 0
    vk_astc_low_memory();
    if (flush_epoch == submit_epoch && purge_epoch == submit_epoch) return 0;
    uint64_t used0 = 0, dev = 0, used = 0, budget = 0;
    int exact;
    vk_mem_vram(&used0, &dev, &exact);
    if (flush_epoch != submit_epoch) {
        flush_epoch = submit_epoch;
        int freed = vk_mem_reclaim(vk_sync_issued());
        if (n_logs < 32) {
            n_logs++;
            vk_mem_vram(&used, &budget, &exact);
            rt_log("vk: video memory full at an ASTC decode of %.1f MiB (%llu of %llu MiB in use): %d deferred objects "
                   "freed once the GPU was done with them; %llu MiB in use",
                   decode_bytes / 1048576.0, (unsigned long long)(used0 >> 20), (unsigned long long)(dev >> 20), freed,
                   (unsigned long long)(used >> 20));
        }
        if (freed) return 1;
    }
    if (purge_epoch == submit_epoch) return 0;
    purge_epoch = submit_epoch;
    cache_budget_init();
    uint64_t was = cache_budget, held = own_bytes;
    uint64_t over = 0, pct = vram_pressure > 0 ? (uint64_t)vram_pressure : 90;
    if (used0 > dev / 100 * pct) over = used0 - dev / 100 * pct;
    if (over < decode_bytes) over = decode_bytes;
    cache_budget_shrink(over);
    uint32_t n = 0, cap = 0, dropped = 0;
    uint64_t dropped_bytes = 0;
    Tex **list = NULL;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (!e || e == keep || !cache_evictable(e)) continue;
        if (n == cap && !(list = realloc(list, sizeof *list * (cap = cap ? 2 * cap : 256))))
            rt_fatal("vk: out of memory for the texture purge list");
        list[n++] = e;
    }
    if (n) qsort(list, n, sizeof *list, reclaim_order);
    // Unused for CACHE_AGE submits (cache_evictable): no submission after the newest submitted value uses them.
    uint64_t tl = vk_sync_issued();
    purge_tl = tl;
    for (uint32_t i = 0; i < n && own_bytes > cache_budget; i++) {
        n_cache_drops++, n_cache_drop_bytes += list[i]->bytes;
        dropped++, dropped_bytes += list[i]->bytes;
        pre_keep(list[i]);
        destroy_tex(list[i]);
    }
    purge_tl = 0;
    free(list);
    int freed = dropped ? vk_mem_reclaim(tl) : 0;   // nothing dropped: nothing to wait for
    if (n_logs < 32) {
        n_logs++;
        vk_mem_vram(&used, &budget, &exact);
        rt_log("vk: video memory full at an ASTC decode of %.1f MiB (%llu of %llu MiB in use): texture cache budget "
               "%llu -> %llu MiB (it held %llu MiB), %u entries dropped (%llu MiB), %d objects freed; %llu MiB in use",
               decode_bytes / 1048576.0, (unsigned long long)(used0 >> 20), (unsigned long long)(dev >> 20),
               (unsigned long long)(was >> 20), (unsigned long long)(cache_budget >> 20), (unsigned long long)(held >> 20),
               dropped, (unsigned long long)(dropped_bytes >> 20), freed, (unsigned long long)(used >> 20));
    }
    return dropped != 0;
}

// ---------------------------------------------------------------- uploads
static uint32_t texel_alpha_one(int format, uint32_t *elem) {
    switch (format) {
    case 0x19: *elem = 1; return 0xFF;
    case 0x1A: *elem = 1; return 0x7F;
    case 0x1B: case 0x1C: *elem = 1; return 1;
    case 0x1D: *elem = 2; return 0x3C00;
    case 0x1E: *elem = 2; return 0xFFFF;
    case 0x1F: *elem = 2; return 0x7FFF;
    case 0x20: case 0x21: *elem = 2; return 1;
    case 0x22: *elem = 4; return 0x3F800000;
    default: *elem = 4; return 1;   // 0x23, 0x24
    }
}

// 3-component texels to 4 (n texels).
static void expand_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e, one = texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 3 * e, dst += 4 * e) {
        memcpy(dst, src, 3 * e);
        memcpy(dst + 3 * e, &one, e);
    }
}

static void shrink_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e;
    texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 4 * e, dst += 3 * e) memcpy(dst, src, 3 * e);
}

// AC_ASTC=check: the compute decoder's texels of n regions (gpu: the regions one after the other) against
// astc_decode_ldr of the same raw blocks (raw: the staging bytes the regions point into).
static void astc_check(const Tex *e, const uint8_t *raw, const VkAstcRegion *ar, int n, const uint8_t *gpu) {
    static volatile int32_t n_bad;
    static uint64_t n_tex, n_texels, n_diff_tex;
    uint32_t bw = e->f.astc_bw, bh = e->f.astc_bh;
    uint64_t diff = 0, first = UINT64_MAX;
    uint32_t want = 0, got = 0;
    int first_k = 0;
    for (int k = 0; k < n; k++) {
        const VkAstcRegion *r = &ar[k];
        size_t sb = (size_t)((r->w + bw - 1) / bw) * ((r->h + bh - 1) / bh) * 16, st = (size_t)r->w * r->h * 4;
        uint8_t *cpu = malloc(st);
        for (uint32_t z = 0; z < r->slices; z++, gpu += st) {
            astc_decode_ldr(raw + r->src + z * sb, r->w, r->h, bw, bh, e->f.astc_srgb, cpu);
            if (!memcmp(cpu, gpu, st)) continue;
            for (size_t i = 0; i < st / 4; i++)
                if (memcmp(cpu + 4 * i, gpu + 4 * i, 4)) {
                    if (first == UINT64_MAX) {
                        first = i, first_k = k;
                        memcpy(&want, cpu + 4 * i, 4), memcpy(&got, gpu + 4 * i, 4);
                    }
                    diff++;
                }
        }
        free(cpu);
        n_texels += (uint64_t)r->w * r->h * r->slices;
    }
    n_tex++;
    if (diff) {
        n_diff_tex++;
        log_limited(&n_bad, "ASTC check: texture %#llx fmt %#x %ux%u: %llu texels differ, first region %d texel %llu "
                    "cpu %08x gpu %08x", (unsigned long long)e->pub.addr, e->t.d.format, bw, bh,
                    (unsigned long long)diff, first_k, (unsigned long long)first, want, got);
    }
    if (n_tex == 1 || !(n_tex & 63) || diff)
        rt_log("vk: ASTC check: %llu textures, %llu texels compared, %llu textures differ", (unsigned long long)n_tex,
               (unsigned long long)n_texels, (unsigned long long)n_diff_tex);
}

// Staging bytes of one part of a texture upload decoded or copied on the CPU (see upload). With AC_ASTC=cpu a
// large ASTC array decodes to RGBA8 in one go: a staging buffer of 676682336 bytes failed (-2) on an Intel UHD with
// 8 GB of shared memory and the game closed before the title screen.
#define UPLOAD_PART (32ull << 20)   // fits the staging pool (vk_staging takes up to half of its 64 MiB block)

// Whether create_tex's alpha verdict still describes the storage: no page written since its scan. Consumed by
// the first upload that asks, so a later upload (the storage was rewritten) scans again.
static int bc_verdict_current(Tex *e) {
    uint32_t since = e->bc_gen, now;
    if (!since) return 0;
    e->bc_gen = 0;
    return vk_mem_changed(e->pub.addr, e->t.L.size, since, &now) == 0;
}

// Copies the whole storage into the image (every level and layer the image has).
static void upload(Tex *e) {
    static volatile int32_t n_bad, n_undecoded, n_split;
    const tex_t *t = &e->t;
    const nvn_texdesc *d = &t->d;
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || !t->mem || e->bc_flip) return;
    if (e->nomem_epoch && e->nomem_epoch == submit_epoch) return;   // video memory full in this submit: the next one
    uint64_t p0 = perf_begin(), u0 = kern_now_ns();
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D, one_d = e->pub.type == VK_IMAGE_TYPE_1D;
    // AC_ASTC_BC=bc1: an upload is the moment the storage's content is read, so the alpha scan runs again.
    // A texture whose content turned opaque or alpha-carrying since its image was made — between the
    // creation scan and a first upload of unwritten storage, or by a CPU rewrite — trades the entry in for
    // the other format (bc_trade) instead of uploading into an image that cannot hold it. Nothing is scanned
    // on a BC3-forced device or for 3D targets, which never take BC1. The first upload skips the scan when no
    // page of the storage was written since create_tex scanned it (bc_verdict_current): the bytes are the ones
    // that verdict was taken from, and the scan is the costly part of a new texture.
    if (e->f.astc_bc && vk_astc_bc() == 2 && !is3d && !bc_verdict_current(e)) {
        int alpha = astc_storage_alpha_possible(t->mem, e->t.L.size, e->f.astc_bw, e->f.astc_bh);
        if ((alpha != 0) != (e->f.astc_bc != 2)) {
            bc_trade(e);
            perf_end(PERF_TEX, p0);
            return;
        }
    }
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear ? 1 : e->pub.layers;
    uint32_t skip = e->pub.skip;   // AC_TEX_MIP_SKIP: image level lv holds the game's level lv + skip
    int astc = e->f.astc_bw != 0, rgb = e->f.expand_rgb;
    int agpu = astc && vk_astc_gpu(), acheck = agpu && vk_astc_mode() == VK_ASTC_CHECK;   // raw blocks to the GPU
    uint32_t n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    VkAstcRegion *ar = agpu ? calloc(n, sizeof *ar) : NULL;
    VkDeviceSize total = 0, decoded = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            uint32_t gl = lv + skip;
            uint32_t w = mip(d->width, gl), h = one_d ? 1 : mip(d->height, gl), sl = is3d ? mip(d->depth, gl) : 1;
            uint64_t texels = (uint64_t)w * h * sl, raw = (uint64_t)(w + bw - 1) / bw * bpb * ((h + bh - 1) / bh) * sl;
            uint64_t out = agpu ? raw : astc ? texels * 4 : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
            rg[k++] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {e->pub.aspect, lv, ly, 1},
                                          .imageExtent = {w, h, sl}};
            total += (out + 15) & ~15ull;
            decoded += texels * 4;
        }
    VkBuffer sb, rbuf = VK_NULL_HANDLE;
    VmaAllocation ralloc = NULL;
    uint8_t *rmap = NULL, *raw_copy = NULL;
    int checked = 0, parts = 1, host = 0;   // host: the decoder's scratch comes from system memory
    // A CPU decode or copy stages at most UPLOAD_PART bytes per staging buffer (a region larger than that alone) and
    // records one copy per buffer; every part stays in the same upload batch, submitted as before. The GPU decoder
    // takes the raw blocks of every region from one buffer.
    for (uint32_t k0 = 0, k1; k0 < n; k0 = k1) {
        VkDeviceSize base = rg[k0].bufferOffset, size;
        for (k1 = k0 + 1; k1 < n && (agpu || (k1 + 1 < n ? rg[k1 + 1].bufferOffset : total) - base <= UPLOAD_PART);) k1++;
        size = (k1 < n ? rg[k1].bufferOffset : total) - base;
        if (k0) parts++;
        VkCommandBuffer cb = vk_upload_begin();
        uint8_t *stg = vk_staging(size, &sb, NULL);
        if (acheck) rmap = vk_staging(decoded, &rbuf, &ralloc);
        for (k = k0; k < k1; k++) {
            uint32_t lv = rg[k].imageSubresource.mipLevel, ly = rg[k].imageSubresource.baseArrayLayer;
            uint32_t w = rg[k].imageExtent.width, h = rg[k].imageExtent.height, sl = rg[k].imageExtent.depth;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            size_t raw = (size_t)rb * rows * sl;
            rg[k].bufferOffset -= base;
            uint8_t *dst = stg + rg[k].bufferOffset, *tmp = (astc && !agpu) || rgb ? malloc(raw) : NULL;
            if (nvn_deswizzle(d, lv + skip, ly, t->mem, tmp ? tmp : dst, rb)) {
                log_limited(&n_bad, "texture %#llx fmt %#x target %d: level %u layer %u has no CPU layout; uploaded as zeros",
                            (unsigned long long)e->pub.addr, d->format, d->target, lv + skip, ly);
                memset(tmp ? tmp : dst, 0, raw);
            }
            if (agpu) {
                ar[k] = (VkAstcRegion){.src = rg[k].bufferOffset, .w = w, .h = h, .slices = sl, .level = lv, .layer = ly};
            } else if (astc) {
                uint64_t a0 = perf_begin();
                for (uint32_t z = 0; z < sl; z++)
                    astc_decode_ldr_mt(tmp + (size_t)z * rb * rows, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb,
                                       dst + (size_t)z * w * h * 4);
                perf_end(PERF_ASTC, a0);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else if (rgb) {
                expand_rgb(d->format, tmp, dst, (size_t)w * h * sl);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else {
                rg[k].bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw, rg[k].bufferImageHeight = rows * (uint32_t)bh;
            }
            free(tmp);
        }
        if (acheck) {
            raw_copy = malloc(size);
            memcpy(raw_copy, stg, size);
        }
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (agpu) {
            uint64_t a0 = perf_begin();
            checked = (host ? vk_astc_record_host : vk_astc_record)(cb, sb, size, ar, (int)n, e->f.astc_bw, e->f.astc_bh,
                                                                    e->f.astc_srgb, e->f.astc_bc, e->pub.image,
                                                                    e->pub.aspect, rbuf);
            perf_end(PERF_ASTC, a0);
            if (checked == VK_ASTC_NO_SCRATCH && !host) {
                // Video memory full, nothing recorded. The decode is recorded again while freeing memory helps
                // (tex_reclaim: at most twice a submit), then once with its scratch in system memory. Freeing submits
                // this batch, and its staging buffer goes with it: the part (the only one, the GPU decoder takes
                // every region from one buffer, so base is 0) is staged again.
                vk_upload_end();
                if (!tex_reclaim(e, decoded)) host = 1;
                if (acheck) {
                    vmaDestroyBuffer(vk_ctx()->vma, rbuf, ralloc);
                    rbuf = VK_NULL_HANDLE, ralloc = NULL;
                    free(raw_copy), raw_copy = NULL;
                }
                k1 = k0;
                continue;
            }
            if (checked <= 0)
                log_limited(&n_undecoded, "ASTC texture %#llx (fmt %#x %ux%u) not decoded on the GPU: %s",
                            (unsigned long long)e->pub.addr, d->format, d->width, d->height,
                            checked ? "decoded again at its next use" : "left as it was");
        } else {
            vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, k1 - k0, rg + k0);
        }
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        vk_upload_end();
    }
    if (parts > 1)
        log_limited(&n_split, "texture %#llx fmt %#x %ux%u, %u levels, %u layers: %.1f MiB staged in %d parts",
                    (unsigned long long)e->pub.addr, d->format, d->width, d->height, levels, layers, total / 1048576.0,
                    parts);
    if (acheck) {   // waits for the GPU
        vk_upload_flush(1);
        if (checked > 0) {
            vmaInvalidateAllocation(vk_ctx()->vma, ralloc, 0, VK_WHOLE_SIZE);
            astc_check(e, raw_copy, ar, (int)n, rmap);
        }
        vmaDestroyBuffer(vk_ctx()->vma, rbuf, ralloc);
    }
    free(raw_copy);
    free(ar);
    free(rg);
    uint64_t dt = kern_now_ns() - u0;
    if (e->uploaded) {
        n_reuploads++, perf_count(PERF_CNT_TEX_REUP);
        ru_n++, ru_bytes += t->L.size, ru_ns += dt;
        if (e->wt_n) {
            ru_wt_n++, ru_wt_bytes += t->L.size, ru_wt_ns += dt;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("perf: frame %llu: re-upload after %u WriteTexels: texture %#llx fmt %#x %ux%u, %.1f KiB, %.3f ms",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), e->wt_n,
                       (unsigned long long)e->pub.addr, d->format, d->width, d->height, t->L.size / 1024.0, dt / 1e6);
        }
    } else {
        n_uploads++, perf_count(PERF_CNT_TEX_UP);
        if (astc && !pre_in_step) pre_n_draw++, pre_tx_draw += decoded / 4;
    }
    e->wt_n = 0;
    // Neither video memory, after what this submit could free, nor system memory had room for the decoder's scratch:
    // the image did not get the content, and an entry counted as uploaded is only uploaded again when its storage
    // changes. Not counted, the next use (make_fresh) tries again.
    e->uploaded = checked != VK_ASTC_NO_SCRATCH;
    e->nomem_epoch = e->uploaded ? 0 : submit_epoch ? submit_epoch : 1;
    e->n_up++;
    perf_end(PERF_TEX, p0);
}

// Whether e's storage fills no 4 KiB page of its own, so the write tracking can only see writes to pages it shares.
static int fills_no_page(const Tex *e) {
    return ((e->pub.addr + e->t.L.size) & ~0xfffull) <= ((e->pub.addr + 0xfff) & ~0xfffull);
}

// A texture that fills no page keeps a copy of its storage bytes from when a GPU write to it was recorded
// (vk_texture_gpu_written), up to SNAP_MAX bytes; wb_exact (AC_WB_EXACT, default on) enables it.
enum { SNAP_MAX = 4096 };
static int wb_exact = -1;
static volatile int32_t n_kept;

// vk_mem_changed over the pages wholly inside e's storage (all of its pages when it fills none): a neighbour that shares
// the first or last page does not count as a write to e. A texture that fills no page shares all of its pages, so
// there a page write counts only when e's own bytes differ from the copy taken at the GPU write. Astral Chain's Purah Pad
// camera renders the ID, screen rect and depth of what is under the reticle into 4x4 targets of 512 bytes, two
// slots used in turn, and the CPU reads them a frame later. One slot's rect target shares its page with data the CPU
// writes every frame: without the copy its renders were never written back, the game read a zero rect every other
// frame and the target's name label blinked. AC_WB_EXACT=0 restores the page-only decision.
static int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now) {
    uint64_t lo = (e->pub.addr + 0xfff) & ~0xfffull, hi = (e->pub.addr + e->t.L.size) & ~0xfffull;
    if (hi <= lo) lo = e->pub.addr, hi = e->pub.addr + e->t.L.size;
    int ch = vk_mem_changed(lo, hi - lo, since, now);
    if (ch != 1 || !since || !e->snap || memcmp(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size)) return ch == 1;
    log_limited(&n_kept, "write-back of %#llx kept: fmt %#x %ux%u, %llu bytes; a neighbour wrote its page",
                (unsigned long long)e->pub.addr, e->t.d.format, e->t.d.width, e->t.d.height,
                (unsigned long long)e->t.L.size);
    return 0;
}

// A mirrored texture is uploaded on first use and again when its storage pages carry a newer write generation.
static void make_fresh(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint64_t size = e->t.L.size;
    if (!e->uploaded) {
        vk_mem_changed(e->pub.addr, size, 0, &e->wgen);   // resets the write bits before the storage is read
        upload(e);
        return;
    }
    uint32_t now;
    int ch = vk_mem_changed(e->pub.addr, size, e->wgen, &now);
    e->wgen = now;
    e->last_ch = ch;
    if (ch) upload(e);   // 1 written, -1 not tracked: both upload again
}

// make_fresh for draw bindings: the page check answers from this submission's write-watch snapshot (vk_memory.c
// vk_mem_changed_snap) instead of a system call per texture. A write the snapshot of the texture's region missed (made
// after it, during this replay) keeps its write-watch bit and is caught by the next submission's snapshot.
static void make_fresh_snap(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint32_t now;
    n_checks++;
    int ch = vk_mem_changed_snap(e->pub.addr, e->t.L.size, e->uploaded ? e->wgen : 0, submit_epoch, &now);
    e->wgen = now;
    if (!e->uploaded) { upload(e); return; }
    e->last_ch = ch;
    if (ch) upload(e);
}

// ---------------------------------------------------------------- aliases (AC_TEX_ALIAS)
// Bytes per block and block size of the image's VkFormat (ASTC decodes to RGBA8, RGB expands to RGBA); 0 when the
// VkFormat substitutes another one of unknown size.
static int img_block(const Tex *e, int *bw, int *bh) {
    if (e->f.astc_bc) { *bw = *bh = 4; return e->f.astc_bc == 2 ? 8 : 16; }
    if (e->f.astc_bw) { *bw = *bh = 1; return 4; }
    int b = nvn_fmt_bpb(e->t.d.format, bw, bh);
    if (e->f.expand_rgb) { *bw = *bh = 1; return b / 3 * 4; }
    return e->f.vk == e->f.ideal ? b : 0;
}

// The new entry e can take src's content texel for texel: colour, one sample, the same shape and storage layout, and
// image formats of one block size.
static int alias_compatible(const Tex *e, const Tex *src) {
    const VkTexImage *p = &e->pub, *q = &src->pub;
    if (e->f.depth || e->f.stencil || src->f.depth || src->f.stencil || p->samples > 1 || q->samples > 1) return 0;
    if (p->type != q->type || p->width != q->width || p->height != q->height || p->depth != q->depth) return 0;
    if (e->t.L.size != src->t.L.size || nvn_tex_is_linear(&e->t.d) != nvn_tex_is_linear(&src->t.d)) return 0;
    int ew, eh, sw, sh, eb = img_block(e, &ew, &eh), sb = img_block(src, &sw, &sh);
    return eb && eb == sb && ew == sw && eh == sh;
}

// src holds content the storage bytes do not: the GPU wrote it and, for a mirrored texture, it was not written back
// and the CPU has not rewritten the storage since (the new entry's upload then reads the newer bytes).
static int alias_has_gpu_content(const Tex *src) {
    if (!src->pub.gpu_owned) return 0;
    if (!src->pub.mirrored) return 1;
    uint32_t now;
    return src->gpu_dirty && !storage_rewritten(src, src->gpu_gen, &now);
}

// Maxwell reads the same bytes through both descriptors: src's content goes into the new image e at the replay's
// current position, before any later command of the submission uses e. Through a buffer, since bound images share
// their memory and image copies may not overlap; e leaves UNDEFINED between the two halves, so its first transition
// cannot clobber src's bytes before they are read. 0 when no replay is recording (e is then not initialised yet).
static int alias_copy(Tex *e, const Tex *src) {
    int reopen;
    VkCommandBuffer cb = vk_exec_inline_begin(&reopen);
    if (!cb) return 0;
    int bw, bh, bpb = img_block(e, &bw, &bh);
    uint32_t levels = e->pub.levels < src->pub.levels ? e->pub.levels : src->pub.levels;
    uint32_t layers = e->pub.layers < src->pub.layers ? e->pub.layers : src->pub.layers, n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    if (!rg) rt_fatal("vk: out of memory for an alias copy");
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = mip(e->pub.width, lv), h = mip(e->pub.height, lv), z = mip(e->pub.depth, lv);
            rg[k] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1},
                                        .imageExtent = {w, h, z}};
            total += ((VkDeviceSize)(w + bw - 1) / bw * ((h + bh - 1) / bh) * z * bpb + 15) & ~15ull;
        }
    VkBuffer buf = vk_exec_inline_buffer(total);
    if (!buf) {
        free(rg);
        vk_exec_inline_end(reopen);
        return 0;
    }
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, src->pub.image, VK_IMAGE_LAYOUT_GENERAL, buf, n, rg);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdCopyBufferToImage(cb, buf, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_exec_inline_end(reopen);
    free(rg);
    return 1;
}

static Tex *entry_of(const tex_t *t) {
    TexKey k;
    make_key(t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e) {
        if (tex_alias < 0) {
            tex_alias = cfg_bool("AC_TEX_ALIAS");
            rt_log("vk: texture aliases: %s", tex_alias ? "a new image takes the live image's content" : "a new image starts empty");
        }
        if (!(e = create_tex(t, &k, tex_alias))) return NULL;
        perf_count(PERF_CNT_TEX_NEW);
        // released entries of another descriptor at this address are replaced by the new one; a live one holding
        // GPU-written content is the alias source, preferring the one that differs only in swizzle (same_but_swizzle)
        int alias = 0;
        Tex *src = NULL;
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr), *nx; p; p = nx) {
            nx = p->next_at;
            if (p->released) { destroy_tex(p); continue; }
            alias = 1;
            if (tex_alias && (!src || (same_but_swizzle(&p->key, &k) && !same_but_swizzle(&src->key, &k))) &&
                alias_compatible(e, p) && alias_has_gpu_content(p))
                src = p;
        }
        if (alias) perf_count(PERF_CNT_TEX_ALIAS);
        if (tex_alias) {
            if (src && alias_copy(e, src)) {
                n_alias_copied++;
                if (e->pub.mirrored) {   // the content is src's GPU render, newer than the storage bytes
                    vk_mem_changed(e->pub.addr, e->t.L.size, 0, &e->wgen);
                    e->uploaded = 1;
                    e->pub.gpu_owned = 1;
                }
            } else {
                init_image(e, vk_upload_begin());
                vk_upload_end();
                if (alias) {
                    if (e->pub.mirrored) n_alias_upload++;
                    else n_alias_uninit++;
                }
            }
        }
        e->next_at = nvn_hm_get(&by_addr, &k.addr);
        nvn_hm_put(&by_addr, &k.addr, e);
        nvn_hm_put(&by_key, &k, e);
        tex_gen_bump(k.addr);
    }
    if (e->released) e->released = 0, nreleased--, rel_unlink(e), tex_gen_bump(k.addr);
    e->t = *t;
    return e;
}

// vk_dump: upload bookkeeping of an image vk_texture_get / vk_texture_by_id returned. pages_written: whether its
// storage pages were written since the last check (1, 0, -1 not tracked); the query leaves that check's state alone.
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written) {
    lock();
    const Tex *e = (const Tex *)img;   // pub is the first member
    uint32_t now;
    *uploads = e->n_up;
    *last_change = e->last_ch;
    *pages_written = e->pub.mirrored && e->uploaded ? vk_mem_changed(e->pub.addr, e->t.L.size, e->wgen, &now) : 0;
    unlock();
}

const VkTexImage *vk_texture_get(const tex_t *t) {
    if (!t || !t->mem) return NULL;
    lock();
    Tex *e = entry_of(t);
    if (e) make_fresh(e), e->last_use = submit_epoch;
    unlock();
    return e ? &e->pub : NULL;
}

// Small render targets the CPU reads without NVN's invalidate (vk_texture_write_back_small); small_pending: one may be
// dirty. wb_limit is the largest target (bytes) that goes back when the CPU waits for a fence: AC_WB_FENCE_LIMIT,
// read once. wb_trace: AC_TRACE=gfx, read together with it.
enum { WB_SMALL_DEFAULT = 64 << 10 };
static int small_pending;
static uint64_t wb_limit = WB_SMALL_DEFAULT;
static int wb_trace;
static int wb_cfg_read;

static void wb_cfg_init(void) {
    if (wb_cfg_read) return;
    int v = cfg_int("AC_WB_FENCE_LIMIT");
    wb_limit = v < 0 ? 0 : (uint64_t)v;
    wb_trace = (rt_trace & RT_TRACE_GFX) != 0;
    wb_cfg_read = 1;
    if (wb_limit != WB_SMALL_DEFAULT) rt_log("vk: fence write-back limit %llu bytes", (unsigned long long)wb_limit);
}

void vk_texture_gpu_written(const tex_t *t) {
    if (!t || !t->mem) return;
    vk_exec_frame_written(t);
    lock();
    Tex *e = entry_of(t);
    if (e) {
        make_fresh(e);   // a partial render keeps what the storage held before
        e->pub.gpu_owned = 1;
        uint64_t f = vk_exec_recording_fence();
        if (f > e->wr_fence) e->wr_fence = f;
        if (e->pub.mirrored) {
            e->gpu_dirty = 1;
            storage_rewritten(e, 0, &e->gpu_gen);   // CPU writes up to here are older than this render
            if (wb_exact < 0) wb_exact = cfg_bool("AC_WB_EXACT");
            if (wb_exact && !e->snap && e->t.L.size <= SNAP_MAX && fills_no_page(e) && !(e->snap = malloc(SNAP_MAX)))
                rt_fatal("vk: out of memory for a texture's storage copy");
            if (e->snap) memcpy(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size);
            e->gpu_seq = ++gpu_writes;
            e->gpu_fence = vk_exec_recording_fence();
            wb_cfg_init();
            if (e->t.L.size <= wb_limit) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
            own_add(e);
        }
    }
    unlock();
}

void vk_texture_release(const tex_t *t) {
    if (!t || !t->mem) return;
    TexKey k;
    make_key(t, &k);
    lock();
    uint32_t pg;
    pre_take(&k, &pg);   // finalized before its upload ahead: nothing to make
    Tex *e = nvn_hm_get(&by_key, &k);
    if (e && !e->released) {
        e->released = 1;
        nreleased++;
        rel_append(e);
        tex_gen_bump(k.addr);
        sweep_released();
    }
    unlock();
}

void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc) {
    lock();
    pre_range_gone(lo, size, 0);
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && ((e->key.addr >= lo && e->key.addr - lo < size) || (alloc && e->bound_to == alloc))) list[n++] = e;
    }
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    unlock();
}

// ---------------------------------------------------------------- descriptor slots
void vk_texture_register(int id, const tex_t *t, const view_t *v) {
    if (id < 0 || id >= (1 << 20)) rt_fatal("vk: texture slot %d out of range", id);
    lock();
    if (id >= ntslots) {
        int n = ntslots ? ntslots : 1024;
        while (n <= id) n *= 2;
        TexSlot *s = realloc(tslots, sizeof *s * (size_t)n);
        if (!s) rt_fatal("vk: out of memory for texture slots");
        memset(s + ntslots, 0, sizeof *s * (size_t)(n - ntslots));
        tslots = s, ntslots = n;
    }
    if (!sinfo) {
        uint64_t *p = calloc(1 << 20, sizeof *p);
        if (!p) rt_fatal("vk: out of memory for texture slot tokens");
        __atomic_store_n(&sinfo, p, __ATOMIC_RELEASE);
    }
    TexSlot *s = &tslots[id];
    s->t = *t;
    s->used = 1;
    s->ce = NULL;
    slots_gen++;
    if (!++reg_serial) reg_serial = 1;
    __atomic_store_n(&sinfo[id], (uint64_t)reg_serial << 16 | agen_idx((uintptr_t)t->mem), __ATOMIC_RELEASE);
    s->has_view = v != NULL;
    if (v) s->v = *v;
    else memset(&s->v, 0, sizeof s->v);
    pre_add(t);
    unlock();
}

// The entry slot s resolves to, and the view and has_view a caller gets with it (the slot's, with the descriptor's
// swizzle when the entry is another descriptor at the same storage). NULL when the slot names no texture.
static Tex *slot_resolve(TexSlot *s, view_t *view, int *has_view) {
    *view = s->v;
    *has_view = s->has_view;
    TexKey k;
    make_key(&s->t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e || e->released) {   // MNG hle_nvn.c: a finalized slot follows the live texture at its storage
        // Prefer the live entry that differs from the slot descriptor only in its swizzle: transient render targets
        // share storage, and the newest one at the address is usually another size. The map screen samples its
        // 2250x1080 map target through an (R,G,B,1) descriptor and followed a 960x60 target at the same address.
        Tex *own = e, *any = NULL;
        e = NULL;
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr); p; p = p->next_at) {
            if (p->released) continue;
            if (!any) any = p;
            if (same_but_swizzle(&p->key, &k)) { e = p; break; }
        }
        if (!e) e = any ? any : own;
    }
    if (!e) e = s->t.mem ? entry_of(&s->t) : NULL;
    // A followed texture is another descriptor at the same storage: its image holds the content, but sampling takes the
    // slot descriptor's swizzle. The UI renders text into RGBA8 and samples it through a (A,A,A,A) descriptor; with the
    // render target's RGBA swizzle the text blur saw R = 1 everywhere and drew solid boxes behind tips and labels.
    if (e && !(s->has_view && s->v.set_swizzle)) {
        const nvn_texdesc *d = &s->t.d;
        uint8_t sw[4] = {(uint8_t)d->swizzle_r, (uint8_t)d->swizzle_g, (uint8_t)d->swizzle_b, (uint8_t)d->swizzle_a};
        if (!(sw[0] | sw[1] | sw[2] | sw[3]))   // descriptors built without nvnTextureBuilderSetDefaults
            sw[0] = NVN_TEXTURE_SWIZZLE_R, sw[1] = NVN_TEXTURE_SWIZZLE_G, sw[2] = NVN_TEXTURE_SWIZZLE_B,
            sw[3] = NVN_TEXTURE_SWIZZLE_A;
        if (memcmp(sw, e->pub.swz, 4)) {
            if (!s->has_view) memset(view, 0, sizeof *view);
            memcpy(view->swz, sw, 4);
            view->set_swizzle = 1;
            *has_view = 1;
        }
    }
    return e;
}

void vk_texture_lock(void) { lock(); }
void vk_texture_unlock(void) { unlock(); }
uint32_t vk_texture_slots_gen(void) { return slots_gen; }

uint64_t vk_texture_slot_token(int id) {
    const uint64_t *si = __atomic_load_n(&sinfo, __ATOMIC_ACQUIRE);
    if (!si || id < 0 || id >= (1 << 20)) return 0;
    uint64_t v = __atomic_load_n(&si[id], __ATOMIC_ACQUIRE);
    return v ? (v >> 16) << 32 | __atomic_load_n(&agen[v & 0xffff], __ATOMIC_ACQUIRE) : 0;
}
uint32_t vk_sampler_gen(void) { return __atomic_load_n(&samp_gen, __ATOMIC_RELAXED); }

const VkTexImage *vk_texture_by_id(int id, view_t *view, int *has_view) {
    lock();
    const VkTexImage *r = vk_texture_by_id_locked(id, view, has_view);
    unlock();
    return r;
}

const VkTexImage *vk_texture_by_id_locked(int id, view_t *view, int *has_view) {
    if (id < 0 || id >= ntslots || !tslots[id].used) return NULL;
    TexSlot *s = &tslots[id];
    uint32_t g = agen[sinfo[id] & 0xffff];
    if (!s->ce || s->cgen != g) {   // resolved again: an entry at its address was created, dropped or released, or the
        view_t v;                   // slot registered (s->ce cleared); an empty result is not kept: image creation may
        int hv;                     // succeed later
        s->ce = slot_resolve(s, &v, &hv);   // entry_of may create an entry and bump g: g read before it
        s->cview = v, s->chas_view = (uint8_t)hv;
        s->cgen = g;
        s->cepoch = 0;
    }
    Tex *e = s->ce;
    if (e) e->last_use = submit_epoch;
    // Bindings of one submit share one check, answered from the submit's write-watch snapshot (guest writes during a
    // replay race the GPU read on hardware too). After one check an entry is either uploaded or never uploads (not
    // mirrored, GPU-owned or no CPU layout), so a slot that checked its entry in this submit has nothing left to do.
    if (e && s->cepoch != submit_epoch) {
        if (!e->uploaded || e->checked != submit_epoch) {
            make_fresh_snap(e);
            e->checked = submit_epoch;
        }
        s->cepoch = submit_epoch;
    }
    if (view) *view = has_view ? s->cview : s->v;
    if (has_view) *has_view = view ? s->chas_view : s->has_view;
    return e ? &e->pub : NULL;
}

// vk_dump: the swizzle selectors of the descriptor registered in slot id; 0 when the slot is empty.
int vk_texture_slot_swz(int id, uint8_t swz[4]) {
    lock();
    int ok = id >= 0 && id < ntslots && tslots[id].used;
    if (ok) {
        const nvn_texdesc *d = &tslots[id].t.d;
        swz[0] = (uint8_t)d->swizzle_r, swz[1] = (uint8_t)d->swizzle_g, swz[2] = (uint8_t)d->swizzle_b;
        swz[3] = (uint8_t)d->swizzle_a;
    }
    unlock();
    return ok;
}

uint32_t vk_texture_epoch(void) { return __atomic_load_n(&submit_epoch, __ATOMIC_RELAXED); }

// Each submit (tex_lock held, the render thread): newly registered textures go to the scan thread or, when their format
// needs no scan, straight to the ready list; then ready ones are created and uploaded up to the submit's budget.
static void pre_step(void) {
    if (pre_on <= 0 || !__atomic_load_n(&pre_count, __ATOMIC_ACQUIRE)) return;
    int bc = vk_astc_bc(), scan_mode = bc == 2, woke = 0;
    static uint8_t vram_full;
    if (!(submit_epoch & 7)) {   // the budget query every 8 submits, like sweep_pressure
        uint64_t used, budget;
        int exact, pct = vram_pressure > 0 && vram_pressure < PRE_VRAM_PCT ? vram_pressure : PRE_VRAM_PCT;
        vram_full = vk_mem_vram(&used, &budget, &exact) && used > budget / 100 * (uint64_t)pct;
    }
    cache_budget_init();
    plat_mutex_lock(&pre_lock);
    for (Pre *p; (p = pre_pop(&pre_new));) {
        if (p->cancel) { free(p); continue; }
        TexVkFormat f;
        if (!tex_vk_format(p->key.d.format, &f)) {
            pre_unmap(p);
            free(p);
            continue;
        }
        // The image's size is not known before it is made: a decoded texel is one byte as BC3 (BC1 takes half),
        // four as RGBA8; native ASTC keeps the storage's size and costs the GPU a copy.
        if (f.astc_bw) p->cost = p->texels, p->bytes = p->texels * (bc ? 1 : 4);
        else p->cost = p->texels / PRE_COPY_SHARE, p->bytes = p->size;
        if (f.astc_bc && scan_mode) {
            p->bw = f.astc_bw, p->bh = f.astc_bh, p->state = PRE_SCAN;
            pre_push(&pre_scan, p), woke = 1;
        } else {
            p->state = PRE_READY;
            pre_push(&pre_ready, p);
        }
    }
    uint64_t budget = pre_texels / PRE_SPREAD;
    plat_mutex_unlock(&pre_lock);
    if (budget < PRE_MIN_TEXELS) budget = PRE_MIN_TEXELS;
    if (budget > PRE_MAX_TEXELS) budget = PRE_MAX_TEXELS;
    uint64_t done = 0;
    pre_in_step = 1;
    for (int items = 0; done < budget && items < PRE_MAX_ITEMS && !vram_full; items++) {
        plat_mutex_lock(&pre_lock);
        Pre *p = pre_ready.head;
        if (!p || (!p->cancel && cache_budget && own_bytes + p->bytes > cache_budget)) {   // no room: the queue waits
            plat_mutex_unlock(&pre_lock);
            break;
        }
        pre_pop(&pre_ready);
        if (p->cancel) {
            plat_mutex_unlock(&pre_lock);
            free(p);
            continue;
        }
        // An entry at the storage already (this descriptor or another one): the draw path owns it, with its aliases.
        int skip = nvn_hm_get(&by_key, &p->key) || nvn_hm_get(&by_addr, &p->key.addr);
        uint32_t now;
        if (!skip && p->verdict >= 0 && vk_mem_changed(p->key.addr, p->size, p->gen, &now) != 0) {
            // written since the scan (the game was still filling it): scanned again, a few times at most
            if (p->rescans++ < PRE_RESCANS) {
                p->state = PRE_SCAN;
                pre_push(&pre_scan, p), woke = 1;
                plat_mutex_unlock(&pre_lock);
                continue;
            }
            skip = 1;
        }
        if (skip) pre_unmap(p);
        plat_mutex_unlock(&pre_lock);
        if (!skip) {
            Tex *e = entry_of(&p->t);   // create_tex takes p's verdict and unmaps p (pre_take)
            if (e && !e->uploaded) {
                make_fresh_snap(e);
                e->checked = submit_epoch;
                e->ahead = e->last_use = submit_epoch;
                done += p->cost;
                pre_n_ahead++, pre_tx_ahead += p->texels;
            }
            plat_mutex_lock(&pre_lock);
            pre_unmap(p);   // when create_tex failed before pre_take
            plat_mutex_unlock(&pre_lock);
        }
        free(p);
    }
    pre_in_step = 0;
    if (woke) plat_cond_signal(&pre_cv);
}

void vk_texture_new_submit(void) {
    lock();
    if (!++submit_epoch) submit_epoch = 1;
    sweep_flip();
    sweep_pressure();
    if (!(submit_epoch & 3)) sweep_cache();
    pre_step();
    static uint64_t log_at;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) / 60;
    if ((rt_trace & RT_TRACE_PERF) && f != log_at && tex_alias > 0) {
        log_at = f;
        rt_log("perf: frame %llu: texture aliases: content copied %llu, filled from storage %llu, left uninitialised %llu",
               (unsigned long long)f * 60, (unsigned long long)n_alias_copied, (unsigned long long)n_alias_upload,
               (unsigned long long)n_alias_uninit);
        n_alias_copied = n_alias_upload = n_alias_uninit = 0;
        rt_log("perf: frame %llu: texture re-uploads %llu (%.1f MiB, %.2f ms), after WriteTexels %llu (%.1f MiB, %.2f ms)",
               (unsigned long long)f * 60, (unsigned long long)ru_n, ru_bytes / 1048576.0, ru_ns / 1e6,
               (unsigned long long)ru_wt_n, ru_wt_bytes / 1048576.0, ru_wt_ns / 1e6);
        ru_n = ru_bytes = ru_ns = ru_wt_n = ru_wt_bytes = ru_wt_ns = 0;
        if (sc_n)
            rt_log("perf: frame %llu: BC1: %llu recompressed ASTC textures scanned (%.1f MiB, %.2f ms, "
                   "%.1f us per MiB): BC1 %llu (%.1f MiB), BC3 %llu (%.1f MiB); pool-bound %llu (%.1f MiB), "
                   "own memory %llu (%.1f MiB), skipped %llu",
                   (unsigned long long)f * 60, (unsigned long long)sc_n, sc_bytes / 1048576.0, sc_ns / 1e6,
                   sc_bytes ? sc_ns / 1e3 / (sc_bytes / 1048576.0) : 0.0,
                   (unsigned long long)sc_op_n, sc_op_bytes / 1048576.0,
                   (unsigned long long)sc_al_n, sc_al_bytes / 1048576.0,
                   (unsigned long long)sc_bound_n, sc_bound_bytes / 1048576.0,
                   (unsigned long long)sc_own_n, sc_own_bytes / 1048576.0, (unsigned long long)sc_skip_n);
        sc_n = sc_bytes = sc_ns = sc_op_n = sc_op_bytes = sc_al_n = sc_al_bytes = 0;
        sc_bound_n = sc_bound_bytes = sc_own_n = sc_own_bytes = sc_skip_n = 0;
        uint64_t ck, calls, untr;
        vk_mem_snap_stats(&ck, &calls, &untr);
        rt_log("perf: frame %llu: binding storage checks %llu (texture %llu), write-watch snapshot queries %llu, untracked %llu",
               (unsigned long long)f * 60, (unsigned long long)ck, (unsigned long long)n_checks, (unsigned long long)calls,
               (unsigned long long)untr);
        n_checks = 0;
        {
            plat_mutex_lock(&pre_lock);
            uint32_t waiting = pre_map.n - (uint32_t)pre_kept;
            int kept = pre_kept;
            uint64_t wtx = pre_texels;
            plat_mutex_unlock(&pre_lock);
            rt_log("perf: frame %llu: texture prefetch: %llu uploaded ahead (%.1f M texels), %llu background scans used; "
                   "ASTC first uploads at a draw %llu (%.1f M texels); waiting %u (%.1f M texels); verdicts kept %d",
                   (unsigned long long)f * 60, (unsigned long long)pre_n_ahead, pre_tx_ahead / 1e6,
                   (unsigned long long)pre_n_scan_used, (unsigned long long)pre_n_draw, pre_tx_draw / 1e6, waiting, wtx / 1e6,
                   kept);
            pre_n_ahead = pre_tx_ahead = pre_n_scan_used = pre_n_draw = pre_tx_draw = 0;
        }
        rt_log("perf: frame %llu: released textures kept %d, dropped under VRAM pressure %llu", (unsigned long long)f * 60,
               nreleased, (unsigned long long)n_pressure_drops);
        n_pressure_drops = 0;
    }
    unlock();
}

// ---------------------------------------------------------------- texel reads and writes
typedef struct { unsigned level, layer0; int nz, is3d, bw, bh, bpb; size_t wb, hb, row, img, x0, y0; } Region;

static int region_of(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img, Region *g) {
    g->bpb = nvn_fmt_bpb(t->d.format, &g->bw, &g->bh);
    if (!g->bpb || r->width <= 0 || r->height <= 0 || r->depth < 0 || r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 ||
        r->xoffset % g->bw || r->yoffset % g->bh)
        return 0;
    g->level = v && v->set_levels ? v->base_level : 0;
    g->layer0 = v && v->set_layers ? v->min_layer : 0;
    g->nz = r->depth ? r->depth : 1;
    g->is3d = t->d.target == NVN_TEXTURE_TARGET_3D;
    g->wb = (size_t)((r->width + g->bw - 1) / g->bw) * (size_t)g->bpb;
    g->hb = (size_t)(r->height + g->bh - 1) / (size_t)g->bh;
    g->row = row ? row : g->wb;
    g->img = img ? img : g->row * g->hb;
    g->x0 = (size_t)(r->xoffset / g->bw) * (size_t)g->bpb;
    g->y0 = (size_t)(r->yoffset / g->bh);
    return 1;
}

// Pool bytes <-> caller rows of the region alone (nvn_texels_box). MacNativeGames' nvn_mtl.m goes through a
// linear copy of the whole level; for a small region of a large texture that cost tens of milliseconds while the guest
// thread held its virtual core and the caller the texture lock (monsters2 replay, frames 765-790).
static int storage_texels(const tex_t *t, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    for (int z = 0; z < g->nz; z++) {
        unsigned layer = g->is3d ? 0 : g->layer0 + (unsigned)r->zoffset + (unsigned)z;
        uint32_t slice = g->is3d ? (uint32_t)(r->zoffset + z) : 0;
        if (nvn_texels_box(&t->d, g->level, layer, (uint32_t)g->x0, (uint32_t)g->y0, slice, (uint32_t)g->wb,
                           (uint32_t)g->hb, t->mem, mem + (size_t)z * g->img, g->row, !write))
            return 0;
    }
    return 1;
}

static int read_aspect(const Tex *e, VkImageAspectFlags *a) {
    if (!e->f.depth && !e->f.stencil) { *a = VK_IMAGE_ASPECT_COLOR_BIT; return 1; }
    switch (e->t.d.format) {   // packed depth-stencil storage has no single-aspect copy
    case 0x31: *a = VK_IMAGE_ASPECT_STENCIL_BIT; return 1;
    case 0x32: case 0x34: *a = VK_IMAGE_ASPECT_DEPTH_BIT; return 1;
    default: return 0;
    }
}

// Image <-> caller rows through a staging buffer; the GPU copy is waited for. The region is in the image's texels.
static int image_texels_at(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    static volatile int32_t n_astc, n_ds;
    VkCtx *c = vk_ctx();
    VkImageAspectFlags aspect;
    if (e->f.astc_bw && !write) {
        log_limited(&n_astc, "ReadTexels of ASTC texture %#llx: the image holds decoded RGBA8; zeros returned",
                    (unsigned long long)e->pub.addr);
        return 0;
    }
    if (!read_aspect(e, &aspect)) {
        log_limited(&n_ds, "texels of packed depth-stencil format %#x are not copied", e->t.d.format);
        return 0;
    }
    // AC_TEX_MIP_SKIP: the game's level g->level is image level il; a write into a level left out of the image has
    // nowhere to go and nothing samples it (reads of ASTC were refused above).
    int il = vk_tex_level(&e->pub, g->level);
    if (il < 0) return write;
    if ((uint32_t)il >= e->pub.levels) return 0;
    uint32_t w = (uint32_t)r->width, h = (uint32_t)r->height;
    int rgb = e->f.expand_rgb, astc = e->f.astc_bw != 0, agpu = astc && write && vk_astc_gpu();
    size_t texels = (size_t)w * h, ib = agpu ? 0 : astc ? 4 : rgb ? (size_t)g->bpb / 3 * 4 : 0;
    size_t slice = ib ? texels * ib : g->wb * g->hb;   // agpu: the raw blocks, decoded by vk_astc_record
    VkBuffer sb;
    VmaAllocation sa;
    uint8_t *stg = vk_staging(slice * (size_t)g->nz, &sb, &sa);
    VkBufferImageCopy *rg = calloc((size_t)g->nz, sizeof *rg);
    for (int z = 0; z < g->nz; z++) {
        uint8_t *s = stg + slice * (size_t)z;
        const uint8_t *m = mem + (size_t)z * g->img;
        if (write) {
            if (agpu) {
                for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
            } else if (astc) {
                uint8_t *tight = malloc(g->wb * g->hb);
                for (size_t y = 0; y < g->hb; y++) memcpy(tight + y * g->wb, m + y * g->row, g->wb);
                uint64_t a0 = perf_begin();
                astc_decode_ldr_mt(tight, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, s);
                perf_end(PERF_ASTC, a0);
                free(tight);
            } else if (rgb) {
                for (size_t y = 0; y < g->hb; y++) expand_rgb(e->t.d.format, m + y * g->row, s + y * w * ib, w);
            } else {
                for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
            }
        }
        rg[z] = (VkBufferImageCopy){.bufferOffset = slice * (size_t)z,
                                    .imageSubresource = {aspect, (uint32_t)il, g->is3d ? 0 : g->layer0 + (uint32_t)r->zoffset + (uint32_t)z, 1},
                                    .imageOffset = {r->xoffset, r->yoffset, g->is3d ? r->zoffset + z : 0},
                                    .imageExtent = {w, h, 1}};
    }
    VkCommandBuffer cb = vk_upload_begin();
    if (write) {
        vmaFlushAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        int done = 0;
        // A BC3 image takes whole 4x4 blocks, or a part that ends on the mip's edge; a region of a 5x5 or 6x6 footprint
        // may end inside a block (vk_astc_record checks the start corner).
        uint32_t mw = e->t.d.width >> g->level, mh = e->t.d.height >> g->level;
        int bc_cut = e->f.astc_bc && (((w & 3) && (uint32_t)r->xoffset + w != (mw ? mw : 1)) ||
                                      ((h & 3) && (uint32_t)r->yoffset + h != (mh ? mh : 1)));
        // A BC1 image holds no alpha: a region the game writes with alpha content through this path (a
        // GPU-owned or non-mirrored ASTC texture, which the re-scan of upload never sees) is encoded with
        // its alpha dropped. Logged when it happens, so the corner stays visible; the mirrored path writes
        // the storage instead and trades the entry at its next upload.
        if (agpu && e->f.astc_bc == 2 &&
            astc_storage_alpha_possible(stg, (uint64_t)slice * g->nz, e->f.astc_bw, e->f.astc_bh))
            log_limited(&n_astc, "WriteTexels of alpha content into the BC1 image of texture %#llx: alpha not "
                        "stored", (unsigned long long)e->pub.addr);
        if (agpu && !bc_cut) {   // bc_cut: nothing recorded, logged below as not decoded
            VkAstcRegion *ar = calloc((size_t)g->nz, sizeof *ar);
            for (int z = 0; z < g->nz; z++)
                ar[z] = (VkAstcRegion){.src = slice * (size_t)z, .w = w, .h = h, .slices = 1, .level = rg[z].imageSubresource.mipLevel,
                                       .layer = rg[z].imageSubresource.baseArrayLayer, .offset = rg[z].imageOffset};
            uint64_t a0 = perf_begin();
            done = vk_astc_record(cb, sb, slice * (size_t)g->nz, ar, g->nz, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc,
                                  e->pub.image, aspect, VK_NULL_HANDLE);
            // Video memory full, nothing recorded. These texels exist nowhere else when the texture is not mirrored or
            // the GPU owns it, so a write left out stayed missing for as long as the game used the texture. The decode
            // is recorded again while freeing memory helps (tex_reclaim: at most twice a submit), then once with its
            // scratch in system memory. The staging buffer is this function's, so it outlives the batch that freeing
            // submits; that batch took the barrier, which the next one needs again.
            for (int again = 1; done == VK_ASTC_NO_SCRATCH && again;) {
                vk_upload_end();
                again = tex_reclaim(e, (VkDeviceSize)texels * 4 * (VkDeviceSize)g->nz);
                cb = vk_upload_begin();
                barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                done = (again ? vk_astc_record : vk_astc_record_host)(cb, sb, slice * (size_t)g->nz, ar, g->nz,
                                                                      e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb,
                                                                      e->f.astc_bc, e->pub.image, aspect, VK_NULL_HANDLE);
            }
            perf_end(PERF_ASTC, a0);
            free(ar);
        }
        if (!agpu) vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)g->nz, rg);
        else if (done <= 0) log_limited(&n_astc, "WriteTexels of ASTC texture %#llx not decoded on the GPU: left as it was",
                                    (unsigned long long)e->pub.addr);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else {
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkCmdCopyImageToBuffer(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, sb, (uint32_t)g->nz, rg);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                VK_ACCESS_2_HOST_READ_BIT);
    }
    vk_upload_end();
    vk_upload_flush(1);
    if (!write) {
        vmaInvalidateAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
        for (int z = 0; z < g->nz; z++) {
            const uint8_t *s = stg + slice * (size_t)z;
            uint8_t *m = mem + (size_t)z * g->img;
            for (size_t y = 0; y < g->hb; y++) {
                if (rgb) shrink_rgb(e->t.d.format, s + y * w * ib, m + y * g->row, w);
                else memcpy(m + y * g->row, s + y * g->wb, g->wb);
            }
        }
    }
    free(rg);
    vmaDestroyBuffer(c->vma, sb, sa);
    return 1;
}

// Image <-> caller rows, the region in the game's texels. A texture made larger (a shadow map by AC_SHADOW_SCALE)
// holds about k texels for each of the game's on a side: the game's region maps to the enlarged one, a read takes the
// enlarged texel nearest each of the game's centres and a write repeats each of the game's texels over those it covers.
// Otherwise ReadTexels of such a texture got its top-left corner.
static int image_texels(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    float k = e->pub.scale;
    if (!(k > 1.0f) || g->bw != 1 || g->bh != 1 || e->f.expand_rgb || e->f.astc_bw) return image_texels_at(e, g, r, mem, write);
    uint32_t lw = umax(e->pub.width >> g->level, 1), lh = umax(e->pub.height >> g->level, 1);
    int32_t x0 = (int32_t)lroundf((float)r->xoffset * k), y0 = (int32_t)lroundf((float)r->yoffset * k);
    int32_t x1 = (int32_t)lroundf((float)(r->xoffset + r->width) * k), y1 = (int32_t)lroundf((float)(r->yoffset + r->height) * k);
    if (x1 > (int32_t)lw) x1 = (int32_t)lw;
    if (y1 > (int32_t)lh) y1 = (int32_t)lh;
    if (x1 <= x0 || y1 <= y0) return 0;
    NVNcopyRegion r2 = *r;
    r2.xoffset = x0, r2.yoffset = y0, r2.width = x1 - x0, r2.height = y1 - y0;
    Region g2 = *g;
    size_t bpb = (size_t)g->bpb, w2 = (size_t)r2.width, h2 = (size_t)r2.height;
    g2.wb = w2 * bpb, g2.hb = h2, g2.row = g2.wb, g2.img = g2.wb * h2;
    uint8_t *tmp = malloc(g2.img * (size_t)g->nz);
    if (!tmp) rt_fatal("vk: out of memory for the texels of an enlarged texture");
    uint32_t *xs = malloc(sizeof *xs * ((size_t)r->width > w2 ? (size_t)r->width : w2));
    if (!xs) rt_fatal("vk: out of memory for the texels of an enlarged texture");
    int ok = 1;
    if (write) {   // each enlarged texel takes the game's texel its centre falls in
        for (size_t x = 0; x < w2; x++) {
            int32_t gx = (int32_t)(((float)(x0 + (int32_t)x) + 0.5f) / k) - r->xoffset;
            xs[x] = (uint32_t)(gx < 0 ? 0 : gx >= r->width ? r->width - 1 : gx);
        }
        for (int z = 0; z < g->nz; z++)
            for (size_t y = 0; y < h2; y++) {
                int32_t gy = (int32_t)(((float)(y0 + (int32_t)y) + 0.5f) / k) - r->yoffset;
                gy = gy < 0 ? 0 : gy >= r->height ? r->height - 1 : gy;
                const uint8_t *src = mem + (size_t)z * g->img + (size_t)gy * g->row;
                uint8_t *dst = tmp + (size_t)z * g2.img + y * g2.row;
                for (size_t x = 0; x < w2; x++) memcpy(dst + x * bpb, src + xs[x] * bpb, bpb);
            }
        ok = image_texels_at(e, &g2, &r2, tmp, 1);
    } else if ((ok = image_texels_at(e, &g2, &r2, tmp, 0))) {   // each of the game's texels takes the enlarged one at its centre
        for (int32_t x = 0; x < r->width; x++) {
            int32_t ex = (int32_t)(((float)(r->xoffset + x) + 0.5f) * k) - x0;
            xs[x] = (uint32_t)(ex < 0 ? 0 : ex >= (int32_t)w2 ? (int32_t)w2 - 1 : ex);
        }
        for (int z = 0; z < g->nz; z++)
            for (int32_t y = 0; y < r->height; y++) {
                int32_t ey = (int32_t)(((float)(r->yoffset + y) + 0.5f) * k) - y0;
                ey = ey < 0 ? 0 : ey >= (int32_t)h2 ? (int32_t)h2 - 1 : ey;
                const uint8_t *src = tmp + (size_t)z * g2.img + (size_t)ey * g2.row;
                uint8_t *dst = mem + (size_t)z * g->img + (size_t)y * g->row;
                for (int32_t x = 0; x < r->width; x++) memcpy(dst + (size_t)x * bpb, src + xs[x] * bpb, bpb);
            }
    }
    free(xs);
    free(tmp);
    return ok;
}

static void zero_rows(const Region *g, uint8_t *dst) {
    for (int z = 0; z < g->nz; z++)
        for (size_t y = 0; y < g->hb; y++) memset(dst + (size_t)z * g->img + y * g->row, 0, g->wb);
}

int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    Region g;
    if (!t || !region_of(t, v, r, row, img, &g)) return 0;
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
    return ok;
}

// ReadTexels: the pool bytes are the content of a mirrored texture the GPU has not rendered into; every other texture
// is read back from its image.
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g)) {
        log_limited(&n_bad, "ReadTexels: invalid region %d,%d,%d %dx%dx%d (format %#x)", r->xoffset, r->yoffset,
                    r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e && e->pub.mirrored && !e->pub.gpu_owned) ok = storage_texels(t, &g, r, dst, 0);
    else if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
}

// WriteTexels: a mirrored texture gets the texels in its pool bytes (and in its image once the GPU owns it); every
// other texture in its image.
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g) || (v && v->set_format && v->format != t->d.format)) {
        log_limited(&n_bad, "WriteTexels: invalid region %d,%d,%d %dx%dx%d or view format (format %#x)", r->xoffset,
                    r->yoffset, r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    if (e && e->pub.mirrored) {
        if (!storage_texels(t, &g, r, (uint8_t *)src, 1))
            log_limited(&n_bad, "WriteTexels: region outside level %u of texture %#llx", g.level, (unsigned long long)e->pub.addr);
        vk_guest_written((uintptr_t)t->mem, t->L.size);
        if (e->pub.gpu_owned) image_texels(e, &g, r, (uint8_t *)src, 1);
        else if (e->uploaded) e->wt_n++;
    } else if (e) {
        image_texels(e, &g, r, (uint8_t *)src, 1);
    }
    unlock();
}

// ---------------------------------------------------------------- CPU reads of GPU-written storage
// A mirrored texture the GPU rendered into holds its texels in its image only; its storage keeps what the CPU last put
// there. The game may still read that storage with the CPU: Astral Chain moves its generated grass maps between pools with
// memcpy once the GPU is done, and the copy then carried zeros. Such a read writes the image back first.
static void write_back(Tex *e) {
    const nvn_texdesc *d = &e->t.d;
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
    for (uint32_t lv = e->pub.skip; lv < e->pub.skip + levels; lv++)   // the game's levels the image holds
        for (uint32_t ly = 0; ly < layers; ly++) {
            view_t v = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)lv, .min_layer = (uint16_t)ly};
            NVNcopyRegion r = {.width = (int)mip(d->width, lv),
                               .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, lv),
                               .depth = is3d ? (int)mip(d->depth, lv) : 1};
            Region g;
            if (!region_of(&e->t, &v, &r, 0, 0, &g)) continue;
            uint8_t *buf = malloc(g.img * (size_t)g.nz);
            if (!buf) rt_fatal("vk: out of memory for a texture write-back");
            if (image_texels(e, &g, &r, buf, 0)) storage_texels(&e->t, &g, &r, buf, 1);
            free(buf);
        }
    e->gpu_dirty = 0;
}

// Mirrored textures overlapping [addr, addr+size) that the GPU wrote since their last write-back.
static int owned_dirty(uint64_t addr, uint64_t size, Tex **hit) {
    int n = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        Tex *e = owned[i];
        if (__atomic_load_n(&e->gpu_dirty, __ATOMIC_RELAXED) && e->pub.addr + e->t.L.size > addr) {
            if (!hit) return 1;
            hit[n++] = e;
        }
    }
    return n;
}

// A storage image may be written by the shaders of the submission being recorded (texture lock held, vk_exec_draw.c
// tex_info). Its storage is not marked GPU-owned: only the write fence for ReadTexels.
void vk_texture_storage_bound_locked(const VkTexImage *img) {
    Tex *e = (Tex *)img;   // pub is the first member
    uint64_t f = vk_exec_recording_fence();
    if (f > e->wr_fence) e->wr_fence = f;
}

uint64_t vk_texture_last_write(const tex_t *t) {
    lock();
    Tex *e = entry_of(t);
    uint64_t f = e ? e->wr_fence : 0;
    unlock();
    return f;
}

size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img) {
    Region g;
    if (!region_of(t, v, r, row, img, &g)) return 0;
    return (size_t)(g.nz - 1) * g.img + (g.hb - 1) * g.row + g.wb;
}

int vk_texture_cpu_stale(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    plat_mutex_lock_shared(&owned_lock);
    int dirty = owned_dirty(addr, size, NULL);
    plat_mutex_unlock_shared(&owned_lock);
    return dirty;
}

// The newest submission recorded to write a GPU-written mirrored texture overlapping [addr, addr+size); 0: none.
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    lock();
    uint64_t f = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        const Tex *e = owned[i];
        if (!e->gpu_dirty || e->pub.addr + e->t.L.size <= addr) continue;
        if (e->gpu_fence > f) f = e->gpu_fence;
        if (e->wr_fence > f) f = e->wr_fence;
    }
    unlock();
    return f;
}

// The image is the newest content only while nothing wrote the storage since the GPU render: Astral Chain finalizes grass maps,
// reuses their storage for new data and memcpys that data on, and writing the old image back first handed the copy
// a stale map or stale resource bytes (wrong textures in the distance). Every hit is checked before any is written
// back, since a write-back is itself a CPU write to storage another hit may share; the hits then go back oldest render
// first, so where two descriptors share storage the newer render wins.
void vk_texture_write_back(uint64_t addr, uint64_t size) {
    lock();
    Tex **hit = malloc(sizeof *hit * (size_t)(nowned + 1));
    if (!hit) rt_fatal("vk: out of memory for a texture write-back");
    int n = owned_dirty(addr, size, hit), k = 0;   // owned[] cannot change while tex_lock is held
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// Astral Chain's grass system renders a 128x128 RGBA8 map of cut and trampled grass into a CPU-cached pool every frame and its
// CPU code reads the pool bytes without nvnMemoryPoolInvalidateMappedRange (cut reactions: blades, insects, the cut
// itself). On the Switch the bytes are there once the frame's fence has passed. So when the
// CPU waits for a fence, mirrored targets up to wb_limit bytes rendered by work up to that fence go back to their
// storage, unless the CPU rewrote it since. Larger targets still go back on an invalidate, memcpy or ReadTexels.
int vk_texture_small_dirty(void) { return __atomic_load_n(&small_pending, __ATOMIC_ACQUIRE); }

// AC_TRACE=gfx: a dirty target the fence write-back leaves alone because it is over the limit (what a CPU read
// after the wait would miss), or sends back although it is over the default limit (what a raised limit changed). One
// line per target address and outcome; the table is small and a full one stops logging.
static void wb_note(const Tex *e, int skipped) {
    static uint64_t seen[2][64];
    static int n_seen[2];
    for (int i = 0; i < n_seen[skipped]; i++)
        if (seen[skipped][i] == e->pub.addr) return;
    if (n_seen[skipped] == 64) return;
    seen[skipped][n_seen[skipped]++] = e->pub.addr;
    const pool_t *p = e->t.pool;
    unsigned fl = p ? p->flags : 0;
    rt_log("vk: fence write-back %s %#llx %ux%u fmt %#x size %llu pool %s (limit %llu)",
           skipped ? "skipped-for-size" : "over-default", (unsigned long long)e->pub.addr, e->t.d.width, e->t.d.height,
           e->t.d.format, (unsigned long long)e->t.L.size,
           !p ? "none" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) ? "cpu-cached" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) ? "cpu-uncached" : "other",
           (unsigned long long)wb_limit);
}

void vk_texture_write_back_small(uint64_t fence) {
    lock();
    wb_cfg_init();
    __atomic_store_n(&small_pending, 0, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&owned_lock);
    int n = 0, later = 0;
    for (int i = 0; i < nowned; i++) n += owned[i]->gpu_dirty && owned[i]->t.L.size <= wb_limit;
    Tex **hit = n ? malloc(sizeof *hit * (size_t)n) : NULL;
    if (n && !hit) rt_fatal("vk: out of memory for a texture write-back");
    n = 0;
    for (int i = 0; i < nowned; i++) {
        Tex *e = owned[i];
        if (!e->gpu_dirty) continue;
        if (e->t.L.size > wb_limit) {
            if (wb_trace && e->gpu_fence <= fence) {
                wb_note(e, 1);
                perf_count(PERF_CNT_WB_SKIP);
            }
            continue;
        }
        if (e->gpu_fence > fence) later = 1;
        else {
            if (wb_trace && e->t.L.size > WB_SMALL_DEFAULT) wb_note(e, 0);
            hit[n++] = e;
        }
    }
    plat_mutex_unlock_shared(&owned_lock);
    if (later) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
    int k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// ---------------------------------------------------------------- samplers (LAS textures.c)
static VkCompareOp compare_op(uint32_t f) {
    switch (f) {
    case NVN_COMPARE_FUNC_NEVER: return VK_COMPARE_OP_NEVER;
    case NVN_COMPARE_FUNC_LESS: return VK_COMPARE_OP_LESS;
    case NVN_COMPARE_FUNC_EQUAL: return VK_COMPARE_OP_EQUAL;
    case NVN_COMPARE_FUNC_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case NVN_COMPARE_FUNC_GREATER: return VK_COMPARE_OP_GREATER;
    case NVN_COMPARE_FUNC_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
    case NVN_COMPARE_FUNC_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

// aniso: AC_ANISO as a level, 0 = the game's; aniso_add: AC_ANISO=auto, the levels added to the game's (doublings)
static struct { int init, mirror_clamp_edge, minmax, border_no_format, aniso_add; float aniso; } sf;

static void sampler_features(VkCtx *c) {
    if (sf.init) return;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                            .pNext = c->has_custom_border_color ? &cbf : NULL};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    sf.mirror_clamp_edge = f12.samplerMirrorClampToEdge;
    sf.minmax = f12.samplerFilterMinmax;
    sf.border_no_format = c->has_custom_border_color && cbf.customBorderColors && cbf.customBorderColorWithoutFormat;
    // AC_ANISO=auto (the default) follows the game resolution's factor over 1080p (resolution.c res_game_size, output_res).
    // At 2160p it follows the emulators' automatic mode (Eglantine, video_core/textures/texture.cpp,
    // TSCEntry::MaxAnisotropy): one doubling of the game's level per whole step of the factor above 1, so twice the
    // game's level. At 1440p that rule adds nothing (the factor is 4/3), and the game asks for no anisotropy on its
    // surfaces (level 1 on all 312 repeating trilinear samplers of a gameplay frame dump), while 4/3 more pixels do
    // little for ground seen at a shallow angle: four doublings, 16x (the device's limit applies). 2160p keeps the
    // doubling: its pixels already sharpen those surfaces, and it is the setting where the GPU has the least time left.
    // At 1080p and below it is the same as game.
    const char *an = cfg_value("AC_ANISO");
    int is_auto = !strcmp(an, "auto");
    sf.aniso = is_auto || !strcmp(an, "game") ? 0 : (float)atoi(an);
    unsigned g[2];
    res_game_size(g);
    float rs = (float)g[1] / 1080.0f;
    sf.aniso_add = !is_auto || rs <= 1.0f ? 0 : rs < 2.0f ? 4 : (int)floorf(rs) - 1;
    if (sf.aniso_add < 0) sf.aniso_add = 0;
    if (sf.aniso > 1) rt_log("vk: anisotropic filtering: %s", an);
    else if (sf.aniso_add) rt_log("vk: anisotropic filtering: the game's level times %d (AC_ANISO=auto, output_res)", 1 << sf.aniso_add);
    else rt_log("vk: anisotropic filtering: the game's");
    sf.init = 1;
}

static VkSampler make_sampler(VkCtx *c, const samp_par *d) {
    sampler_features(c);
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    uint32_t mf = d->minf;
    si.minFilter = mf == 1 || mf == 3 || mf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = mf == 4 || mf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VkSamplerAddressMode modes[3];
    const uint8_t wrap[3] = {d->ws, d->wt, d->wr};
    int border = 0;
    for (int i = 0; i < 3; i++) {
        switch (wrap[i]) {
        case NVN_WRAP_MODE_CLAMP: case NVN_WRAP_MODE_CLAMP_TO_EDGE: modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE:
            modes[i] = sf.mirror_clamp_edge ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            break;
        case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: case NVN_WRAP_MODE_CLAMP_TO_BORDER:
            modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            border = 1;
            break;
        case NVN_WRAP_MODE_MIRRORED_REPEAT: modes[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        default: modes[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }
    si.addressModeU = modes[0], si.addressModeV = modes[1], si.addressModeW = modes[2];
    si.minLod = d->lod_min;
    si.maxLod = d->lod_max > 1000 ? VK_LOD_CLAMP_NONE : d->lod_max;
    if (mf <= NVN_MIN_FILTER_LINEAR) si.maxLod = si.minLod + 0.25f;   // not mipmapped: stay on the base level
    if (si.maxLod < si.minLod) si.maxLod = si.minLod;
    float lim = c->props.limits.maxSamplerLodBias, bias = d->lod_bias;   // with the upscaling bias (sampler_for)
    si.mipLodBias = fmaxf(-lim, fminf(lim, bias));
    // AC_ANISO raises the level for filtered, mipmapped colour samplers that repeat on U and V: the surface textures,
    // which tile, so ground and walls seen at a shallow angle stay sharp into the distance. Depth-compare (shadow)
    // samplers, nearest or unmipmapped ones (data maps, LUTs) and clamped ones keep the game's level: clamped samplers
    // read screen-sized buffers, whose coordinates can stretch steeply at an object's edge, where anisotropic taps reach
    // across it and leave a speckled fringe (the inventory's character preview).
    float aniso = d->aniso, want = sf.aniso_add ? fmaxf(aniso, 1.0f) * (float)(1 << sf.aniso_add) : sf.aniso;
    // The game's own wrap modes: a mirror-clamp one maps to MIRRORED_REPEAT where the device lacks mirror-clamp-to-edge.
    int tiled = (d->ws == NVN_WRAP_MODE_REPEAT || d->ws == NVN_WRAP_MODE_MIRRORED_REPEAT) &&
                (d->wt == NVN_WRAP_MODE_REPEAT || d->wt == NVN_WRAP_MODE_MIRRORED_REPEAT);
    if (want > aniso && tiled && !d->cmp_mode && si.magFilter == VK_FILTER_LINEAR && si.minFilter == VK_FILTER_LINEAR &&
        (mf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST || mf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR))
        aniso = want;
    if (aniso > 1) {
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = fminf(aniso, c->props.limits.maxSamplerAnisotropy);
    }
    if (d->cmp_mode) {
        si.compareEnable = VK_TRUE;
        si.compareOp = compare_op(d->cmp_func);
        // A depth-compare sampler set to nearest returns 0 or 1 per tap, and the game's shaders smooth shadow edges by
        // a few taps spaced for its own shadow map texels. With AC_SHADOW_SCALE those taps cover a fraction of the
        // area and the edges turn hard and stepped; linear filtering makes every tap a 2x2 PCF instead.
        if (shadow_scale() > 1) si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    }
    VkSamplerCustomBorderColorCreateInfoEXT cb = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rm = {.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
    const void **tail = &si.pNext;
    if (border) {
        const float *b = d->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (sf.border_no_format) {
            memcpy(cb.customBorderColor.float32, b, 16);
            si.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            *tail = &cb;
            tail = &cb.pNext;
        } else si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    if (d->reduction && sf.minmax && !si.compareEnable) {
        rm.reductionMode = d->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;
        *tail = &rm;
    }
    VkSampler s;
    VkResult r = vkCreateSampler(c->device, &si, NULL, &s);
    if (r != VK_SUCCESS) rt_fatal("vk: vkCreateSampler failed (%d)", (int)r);
    return s;
}

void vk_sampler_register(int id, const samp_par *s) {
    if (id < 0 || id >= (1 << 12)) rt_fatal("vk: sampler slot %d out of range", id);
    lock();
    if (id >= nsslots) {
        int n = nsslots ? nsslots : 256;
        while (n <= id) n *= 2;
        SampSlot *p = realloc(sslots, sizeof *p * (size_t)n);
        if (!p) rt_fatal("vk: out of memory for sampler slots");
        memset(p + nsslots, 0, sizeof *p * (size_t)(n - nsslots));
        sslots = p, nsslots = n;
    }
    sslots[id].p = *s;
    sslots[id].s = sslots[id].sp = VK_NULL_HANDLE;
    sslots[id].used = 1;
    samp_gen++;
    unlock();
}

// Caller holds the lock. A sampler keeps the game's LOD bias while an upscaler scales the scene too: the bias of
// log2(scene / output) the upscalers' guides ask for (-0.80 at 1472x828 for 2560x1440) made the road's fine dot
// pattern in Harmony Square shimmer in rings as the camera moved, which the temporal upscalers did not settle.
static VkSampler sampler_for(VkCtx *c, const samp_par *s) {
    samp_par k;
    memset(&k, 0, sizeof k);
    k = *s;
    VkSampler v = (VkSampler)nvn_hm_get(&samplers, &k);
    if (!v) {
        v = make_sampler(c, &k);
        nvn_hm_put(&samplers, &k, (void *)v);
    }
    return v;
}

VkSampler vk_sampler_for(const samp_par *s) {
    VkCtx *c = vk_ctx();
    if (!c || !s) return VK_NULL_HANDLE;
    lock();
    VkSampler v = sampler_for(c, s);
    unlock();
    return v;
}

// Samplers live until shutdown, so a slot keeps the one its parameters map to until it is registered again.
VkSampler vk_sampler_by_id_locked(int id) {
    VkCtx *c = vk_ctx();
    if (!c) return VK_NULL_HANDLE;
    VkSampler v = VK_NULL_HANDLE;
    if (id >= 0 && id < nsslots && sslots[id].used) {
        SampSlot *s = &sslots[id];
        if (!s->s) s->s = sampler_for(c, &s->p);
        v = s->s;
    }
    return v;
}

VkSampler vk_sampler_point_by_id_locked(int id) {
    VkCtx *c = vk_ctx();
    if (!c || id < 0 || id >= nsslots || !sslots[id].used) return VK_NULL_HANDLE;
    SampSlot *s = &sslots[id];
    if (!s->sp) {
        samp_par k = s->p;
        k.magf = NVN_MAG_FILTER_NEAREST;
        k.minf = k.minf == NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST :
                 k.minf > NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : k.minf;
        k.aniso = 1;
        s->sp = sampler_for(c, &k);
    }
    return s->sp;
}

VkSampler vk_sampler_by_id(int id) {
    lock();
    VkSampler v = vk_sampler_by_id_locked(id);
    unlock();
    return v;
}

int vk_sampler_params(int id, samp_par *out) {
    lock();
    int ok = id >= 0 && id < nsslots && sslots[id].used;
    if (ok) *out = sslots[id].p;
    unlock();
    return ok;
}

// ---------------------------------------------------------------- statistics and shutdown
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads) {
    lock();
    if (images) *images = by_key.n;
    if (uploads) *uploads = n_uploads;
    if (reuploads) *reuploads = n_reuploads;
    unlock();
}

void vk_texture_census(VkTexCensus *s) {
    VkCtx *c = vk_ctx();
    memset(s, 0, sizeof *s);
    lock();
    cache_budget_init();
    s->cache_budget = cache_budget;
    s->cache_dropped = n_cache_drop_bytes;
    s->cache_drops = (uint32_t)n_cache_drops;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e) continue;
        uint64_t bytes = 0;
        if (e->alloc && c) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(c->vma, e->alloc, &ai);
            bytes = ai.size;
        }
        if (e->released) s->released++, s->own_released += e->alloc != NULL, s->bytes_released += bytes;
        else s->live++, s->own_live += e->alloc != NULL, s->bytes_live += bytes;
        const nvn_texdesc *d = &e->t.d;
        uint32_t side = d->width > d->height ? d->width : d->height;
        if (!e->released && bytes && d->format >= 0x79 && d->format <= 0x94 && e->pub.levels >= 2 &&
            (d->target == NVN_TEXTURE_TARGET_2D || d->target == NVN_TEXTURE_TARGET_2D_ARRAY) && side >= 512) {
            int k = side >= 2048 ? 0 : side >= 1024 ? 1 : 2;
            s->astc_n[k]++, s->astc_bytes[k] += bytes;
        }
    }
    unlock();
}

void vk_texture_shutdown(void) {
    VkCtx *c = vk_ctx();
    lock();
    pre_range_gone(0, 0, 1);
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++)
        if (by_key.vals[i]) list[n++] = by_key.vals[i];
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    for (uint32_t i = 0; i < samplers.cap; i++)
        if (samplers.vals[i] && c) vkDestroySampler(c->device, (VkSampler)samplers.vals[i], NULL);
    free(by_key.keys), free(by_key.vals), free(by_addr.keys), free(by_addr.vals), free(samplers.keys), free(samplers.vals);
    by_key = (nvn_hmap){.klen = sizeof(TexKey)};
    by_addr = (nvn_hmap){.klen = 8};
    samplers = (nvn_hmap){.klen = sizeof(samp_par)};
    free(tslots), free(sslots);
    tslots = NULL, sslots = NULL, ntslots = nsslots = 0, nreleased = 0;
    rel_head = rel_tail = NULL;
    if (sinfo) memset(sinfo, 0, sizeof *sinfo << 20);   // no slot is registered: tokens 0
    unlock();
}
