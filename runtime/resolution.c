// Game resolution: the size the game composes its frame in (its output: canvas, UI and swapchain) and the size of its
// 3D scene, above the shipped 1920x1080 and 1600x900. The game does the rendering itself, as with the community
// resolution mods (UltraCam, Visual Fixes): the patch sets output and scene (recomp/patches-v121.txt,
// patches-v143.txt; MacNativeGames games/totk/runtime/game_patches.c and game_patches_v143.c, MIT, Copyright (c) 2026
// MacNativeGames contributors) make the game read both sizes from the patch page, and the texture heap that holds the
// output- and scene-sized render targets grows with them.
//
// Settings, read at start: AC_OUTPUT_RES = <w>x<h> (default 1920x1080, the game's own), AC_RENDER_SCALE = the
// scene's size relative to the output: "game" (default, 5/6, the game's 1600x900 at 1080p) or a percentage; above 100
// the game supersamples its scene, which smooths edges (anti-aliasing) at the cost of GPU time. The game's own final
// pass (FSR 1 EASU) scales the scene to the output; the present scales the output to the window (vk_fsr.c).
//
// Patch page: +16 the scene size {w, h} as floats, +32 the output size and +48 the swapchain size as {w, h} in one
// 64-bit word, +40 the renderer (1.2.1) or display configuration (1.4.3) the patched code stores there.
// +56 the camera's aspect ratio as a float when the output is not 16:9 (set aspect), +60 the field-of-view factor
// (set fov, res_fov_live).
// 1.4.3 scales the scene it reads by output width / 1920 (a 2560x1440 output renders its 1600x900 scene at 2128x1200;
// MacNativeGames measured the same), so its page holds the wanted size divided by that factor and the game's 5/6 needs
// no scene patch. 1.2.1 renders the size it reads (1600x900 on a 2560x1440 output): its page holds the wanted size, and
// the scene set is on whenever the output differs from 1920x1080.
//
// Heap: agl's DynamicTextureAllocator is one fixed heap, "DynamicTextureAllocatorMem" in
// System/HeapSize/Product.Nin_NX_NVN.HeapDef.byml.zs (113664 KiB, carved from "Graphics", 201728 KiB). At 2560x1440
// the shipped size runs out at the title screen. Both heaps grow by the same amount: the shipped size times the output's
// pixel ratio above 1920x1080 plus the scene's above 1600x900 (MacNativeGames' measurement: a 3440x1440 canvas with a
// 4620x1935 scene crashed at +258 MiB and loaded at +400 MiB; this formula gives +750 MiB there). romfs.c serves the
// rewritten table (res_heapdef_build) in place of the original; memory.c adds the amount to the application pool, so
// no other heap shrinks. Without the rewritten table (a RomFS that lacks it, a format this code does not know) the
// patch sets stay off and the game keeps its own sizes.
//
// Astral Chain 1.0.1 has no output set: there the res set reads AC_OUTPUT_RES from the patch page at the game's render
// size sites and grows the game's render-target budget with it (res_pick below); none of the HeapDef growth above
// applies.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#include "cfg.h"
#include "cpu.h"
#include "gfx.h"
#include "host.h"
#include "layout.h"
#include "ov.h"
#include "platform/platform.h"
#include "simd.h"

int gfx_display_height(void);   // vk.h (which needs Vulkan): the screen the game shows on, 0 when unknown

#define BASE_W 1600
#define BASE_H 900
#define SCENE_W 1600
#define SCENE_H 900
#define DTA_KIB 113664u      // DynamicTextureAllocatorMem, shipped
#define SCENE_MAX 4096       // per side, as MacNativeGames' AC_RENDER_RES
#define SHADOW_BASE 960      // the sun shadow map's base size (its cascades: a D16 array of 4 layers)

static struct {
    int decided;
    unsigned ow, oh, sw, sh;   // output and scene
    int scene_set;             // the scene differs from the game's 5/6 of the output
    int heapdef_ok;            // romfs.c serves the grown HeapDef
    int on;                    // res_init switched the patch sets on
    unsigned shadow;           // fixed sun shadow map size (AC_SHADOW_RES), 0: the game's own
} R;

static void decide(void) {
    if (R.decided) return;
    R.decided = 1;
    R.ow = BASE_W;
    R.oh = BASE_H;
    const char *o = cfg_value("AC_OUTPUT_RES");
    unsigned w = 0, h = 0;
    char tail;
    if (sscanf(o, "%ux%u%c", &w, &h, &tail) == 2 && w >= 1280 && h >= 720 && w <= 7680 && h <= 4320) {
        R.ow = w & ~1u;
        R.oh = h & ~1u;
    } else {
        rt_log("resolution: AC_OUTPUT_RES=%s is not <width>x<height> in 1280x720..7680x4320; 1920x1080", o);
    }
    const char *s = cfg_value("AC_RENDER_SCALE");
    double k = 5.0 / 6.0;
    if (strcmp(s, "game")) {
        char *end;
        double pct = strtod(s, &end);
        if (end != s && !*end && pct >= 25 && pct <= 400) {
            k = pct / 100.0;
            R.scene_set = 1;
        } else {
            rt_log("resolution: AC_RENDER_SCALE=%s is not game or a percentage 25..400; game", s);
        }
    }
    double sw = R.ow * k, sh = R.oh * k;
    double fit = fmin(1.0, fmin(SCENE_MAX / sw, SCENE_MAX / sh));
    if (fit < 1.0) R.scene_set = 1;
    R.sw = (unsigned)(sw * fit + 0.5);
    R.sh = (unsigned)(sh * fit + 0.5);
    const char *sr = cfg_value("AC_SHADOW_RES");
    if (strcmp(sr, "game")) {
        unsigned n = (unsigned)atoi(sr);
        if (n >= 256 && n <= 4096 && !(n & 7)) R.shadow = n;
        else rt_log("resolution: AC_SHADOW_RES=%s is not game or a size 256..4096 (a multiple of 8); game", sr);
    }
}

// Texture heap for a sun shadow map above the base size: the four layers of 2 bytes, three maps in flight (the engine
// makes the array anew each frame).
static uint64_t shadow_kib(void) {
    return R.shadow > SHADOW_BASE ? ((uint64_t)R.shadow * R.shadow - SHADOW_BASE * SHADOW_BASE) * 8 * 3 / 1024 : 0;
}

static int bigger(void) {
    decide();
    return R.ow != BASE_W || R.oh != BASE_H || R.scene_set || shadow_kib();
}

static int output_set(void);   // below: TotK's output patch set is in the generated C

// TotK's HeapDef growth for its output, scene and shadow sizes; 0 without the output set (1.0.1), whose size comes from
// the res set and grows the pool through res_output_extra_bytes instead.
uint64_t res_heap_extra_kib(void) {
    if (!output_set() || !bigger()) return 0;
    double grow = fmax(0.0, (double)R.ow * R.oh / (BASE_W * BASE_H) - 1.0) +
                  fmax(0.0, (double)R.sw * R.sh / (SCENE_W * SCENE_H) - 1.0);
    uint64_t kib = grow > 0 ? (uint64_t)ceil(DTA_KIB * (1.0 + grow) / 1024.0) * 1024 - DTA_KIB : 0;
    return kib + ((shadow_kib() + 1023) & ~1023ull);   // whole MiB
}

uint64_t res_output_extra_bytes(void);   // below: the res set's render-target budget growth

uint64_t res_heap_extra_bytes(void) {
    return ((res_heap_extra_kib() << 10) + res_output_extra_bytes() + 0x1fffff) & ~0x1fffffull;
}

// ---- the HeapDef table. BYML version 2..7, little endian ("YB"): a hash node is {u8 0xC1, u24 count} and count
// entries {u24 key index, u8 type, u32 value}; an int (type 0xD1) is stored in the entry, so a size changes in place.
typedef struct { uint8_t *p; size_t n; uint32_t keys; } Byml;

static int u32_at(const Byml *b, size_t off, uint32_t *v) {
    if (off + 4 > b->n) return 0;
    memcpy(v, b->p + off, 4);
    return 1;
}

static const char *key_name(const Byml *b, uint32_t k) {
    uint32_t head, at;
    if (!u32_at(b, b->keys, &head) || (head & 0xff) != 0xc2 || k >= head >> 8 || !u32_at(b, b->keys + 4 + 4 * k, &at) ||
        b->keys + at >= b->n || !memchr(b->p + b->keys + at, 0, b->n - b->keys - at))
        return NULL;
    return (const char *)b->p + b->keys + at;
}

// The offset of the value of hash node `node`'s entry `name`, which must have type `type`; 0 when there is none.
static size_t field(const Byml *b, uint32_t node, const char *name, uint8_t type) {
    uint32_t head;
    if (!u32_at(b, node, &head) || (head & 0xff) != 0xc1) return 0;
    for (uint32_t i = 0; i < head >> 8; i++) {
        uint32_t e;
        if (!u32_at(b, node + 4 + 8 * (size_t)i, &e)) return 0;
        const char *k = key_name(b, e & 0xffffff);
        if (k && !strcmp(k, name)) return e >> 24 == type ? node + 8 + 8 * (size_t)i : 0;
    }
    return 0;
}

static size_t heap_size_field(const Byml *b, uint32_t heaps, const char *heap) {
    size_t at = field(b, heaps, heap, 0xc1);
    uint32_t node;
    return at && u32_at(b, at, &node) ? field(b, node, "Size", 0xd1) : 0;
}

int res_heapdef_build(const void *in, size_t n, void **out, size_t *out_n) {
    uint64_t extra = res_heap_extra_kib();
    *out = NULL;
    if (!extra) return 0;
    if (ZSTD_getDictID_fromFrame(in, n)) {
        rt_log("resolution: the HeapDef table needs a zstd dictionary; the game keeps its own resolution");
        return 0;
    }
    unsigned long long raw = ZSTD_getFrameContentSize(in, n);
    if (raw == ZSTD_CONTENTSIZE_UNKNOWN || raw == ZSTD_CONTENTSIZE_ERROR || raw > (64u << 20)) {
        rt_log("resolution: the HeapDef table is not a zstd frame with a size; the game keeps its own resolution");
        return 0;
    }
    Byml b = {malloc((size_t)raw), (size_t)raw, 0};
    if (!b.p) return 0;
    size_t got = ZSTD_decompress(b.p, b.n, in, n);
    uint32_t root, heaps = 0;
    uint16_t ver = 0;
    if (!ZSTD_isError(got) && got == b.n && b.n >= 16) memcpy(&ver, b.p + 2, 2);
    size_t heaps_at = 0, dta_at = 0, gfx_at = 0;
    if (ver >= 2 && ver <= 7 && !memcmp(b.p, "YB", 2) && u32_at(&b, 4, &b.keys) && u32_at(&b, 12, &root) &&
        (heaps_at = field(&b, root, "Heaps", 0xc1)) && u32_at(&b, heaps_at, &heaps)) {
        dta_at = heap_size_field(&b, heaps, "DynamicTextureAllocatorMem");
        gfx_at = heap_size_field(&b, heaps, "Graphics");
    }
    if (!dta_at || !gfx_at) {
        rt_log("resolution: the HeapDef table has no DynamicTextureAllocatorMem / Graphics sizes; the game keeps its own "
               "resolution");
        free(b.p);
        return 0;
    }
    int32_t dta, gfx;
    memcpy(&dta, b.p + dta_at, 4);
    memcpy(&gfx, b.p + gfx_at, 4);
    int32_t dta2 = dta + (int32_t)extra, gfx2 = gfx + (int32_t)extra;
    memcpy(b.p + dta_at, &dta2, 4);
    memcpy(b.p + gfx_at, &gfx2, 4);
    size_t cap = ZSTD_compressBound(b.n);
    void *z = malloc(cap);
    size_t zn = z ? ZSTD_compress(z, cap, b.p, b.n, 9) : 0;
    free(b.p);
    if (!z || ZSTD_isError(zn)) {
        free(z);
        return 0;
    }
    *out = z;
    *out_n = zn;
    rt_log("resolution: HeapDef DynamicTextureAllocatorMem %d -> %d KiB, Graphics %d -> %d KiB", dta, dta2, gfx, gfx2);
    return 1;
}

void res_heapdef_served(void) { R.heapdef_ok = 1; }

// ---- sun shadow map size: each frame the engine sizes its shadow cascades to the area they cover, down to about 400 near
// the ground, which draws foliage shadows blocky, and each change of size moves the shadow texels, so leaf shadows jump.
// A fixed size (the shadow set, recomp/patches-v121.txt) keeps them still and sharper. Page +92.
static void shadow_init(uint8_t *page) {
    if (!R.shadow) {
        rt_log("resolution: sun shadow map sized by the game (AC_SHADOW_RES=game)");
        return;
    }
    int set = rt_patch_set("shadow");
    if (set < 0) {
        rt_log("resolution: the generated C has no shadow patch set: the game sizes its shadow map");
        return;
    }
    int32_t n = (int32_t)R.shadow;
    memcpy(page + 92, &n, 4);
    rt_patch_switch(set, 1);
    rt_log("resolution: sun shadow map %ux%u (AC_SHADOW_RES)", R.shadow, R.shadow);
}

// ---- start: the patch page and the sets
static uint64_t pack(unsigned w, unsigned h) { return (uint64_t)h << 32 | w; }

// AC_SHADOW_RANGE (percent, 50 to 400; 100 = the game's): the sun's shadow cascades end at 20, 40, 80 and 160 m.
// Other values multiply those distances through the cascade patch set (recomp/patches-v143.txt), so the sharp first
// cascade reaches further and the visible steps between cascades move away from the player, at the cost of shadow
// texels per metre (AC_SHADOW_SCALE pays that back).
static void res_shadow_range(void) {
    int pct = cfg_int("AC_SHADOW_RANGE"), set = rt_patch_set("cascade");
    if (pct == 100) return;
    if (set < 0) {
        rt_log("resolution: the generated C has no cascade patch set; shadow_range stays 100");
        return;
    }
    float k = (float)(pct < 50 ? 50 : pct > 400 ? 400 : pct) / 100.0f, d[4] = {20 * k, 40 * k, 80 * k, 160 * k};
    memcpy((uint8_t *)rt_patch_page() + 64, d, sizeof d);
    rt_patch_switch(set, 1);
    rt_log("resolution: shadow cascades end at %g, %g, %g and %g m", d[0], d[1], d[2], d[3]);
}

// ---- start: the patch page and the sets
// Game versions without TotK's output patch set (1.0.1): the res set (recomp/patches-v101.txt) reads the output size
// from the patch page (+32 width, +36 height, u32; rt_res_* below) at the sites that hold the game's render size, so
// output_res takes a size as in the TotK Recomp, here any 16:9 one of 32n x 18n: 1920x1080 by default, 1280x720 for
// GPUs that cannot hold 1080p, up to 3840x2160 (res_size). Above 1080p's pixel count the game's render-target budget
// (main+0x40e0b0, 310 MiB, which 1080p fills to 304) grows with the pixels (res_budget), so does the GRAPHIC BUFFER
// heap (res_gbuf_extra), and the process heap grows by both amounts, as they are carved from it
// (res_output_extra_bytes). AC_RES1062=1 still asks for the community 1062p set instead.
#define RES_PAGE 32
#define RES_BUDGET_GAME (310ull << 20)   // the game's own render-target budget, 0x13600000
#define RES_PIXELS_MAX (3840ull * 2160)  // the most the process heap's 4 GiB (memory.c HEAP_CAP) leaves room for

// TotK's output and scene sets (patches-v121.txt, patches-v143.txt) make the game read any output size from the patch
// page, with the grown HeapDef table (res_init below); without them the res set does.
static int output_set(void) { return rt_patch_set("output") >= 0; }

// The output size output_res asks for on 1.0.1, decided once (mem_init sizes the pool before the sets are switched):
// 32n x 18n from 1280x720 (n = 40) to 3840x2160 (n = 120); else 1920x1080. Other shapes and sizes off that grid fail:
// 1280x800 lost the device in the first gameplay frames (a device fault in a shader, as scene sizes off the grid did
// in the temporal-upscaling work), and in handheld, where the scene targets are 1280x720, a 1280-wide output makes the
// ladder's cmp take the docked entries for them, so a level's target is never made and loading a save reads through
// its null texture in main+0x9d0df0 (as issues #25 and #38). On the grid the only 1280-wide size is 1280x720 itself.
// res_output_grid is that rule alone, also for cfg.c's Steam Deck migration (cfg_migrate_deck_res), which has to know
// which stored values end up as 1920x1080.
int res_output_grid(const char *o, unsigned *w, unsigned *h) {
    unsigned a = 0, b = 0;
    char tail;
    if (!o || sscanf(o, "%ux%u%c", &a, &b, &tail) != 2 || a % 32 || b % 18 || a / 32 != b / 18 || a < 1280 ||
        (uint64_t)a * b > RES_PIXELS_MAX)
        return 0;
    *w = a, *h = b;
    return 1;
}

// ---- the upscaler at start, before the device exists (resolution.c cannot ask NGX or XeSS yet): what upscaling
// (cfg.c cfg_upscaling) will run, judged from the settings and the files next to the executable. vk_dlaa.c answers for
// real once the device is up (res_settle). DLSS needs nvngx_dlss.dll from NVIDIA beside astralchain.exe, XeSS
// libxess.dll, both Windows libraries; FSR 3 is built in, so auto (DLSS where it can run, else FSR 3) always has one.
static int beside_exe(const char *name) {
    char exe[1024], path[1100];
    if (!plat_exe_path(exe, sizeof exe)) return 0;
    char *last = plat_last_sep(exe);
    if (!last) return 0;
    snprintf(path, sizeof path, "%.*s/%s", (int)(last - exe), exe, name);
    return plat_file_exists(path);
}

static int upscaler_at_start(void) {
    const char *up = cfg_upscaling();
    if (!strcmp(up, "fsr3") || !strcmp(up, "auto")) return 1;
#ifdef _WIN32
    if (!strcmp(up, "dlss")) return beside_exe("nvngx_dlss.dll");
    if (!strcmp(up, "xess")) return beside_exe("libxess.dll");
#endif
    return 0;
}


// output_res = auto: the largest game resolution the screen (or the window, when not fullscreen) holds of 1280x720 and
// 1920x1080, and while an upscaler runs from the start also 2560x1440 and 3840x2160: DLAA or the upscaler then writes
// the screen's size, the HUD is drawn at it, and the window copies the picture 1:1 instead of scaling it a second time.
// Without one, more than 1080p would be plain supersampling at several times the cost. Only those sizes: a 1600-line
// screen gets 1440p, which the window scales the rest of the way. Not the default: on a 4K screen it renders four times
// 1080p's pixels, which only the player can choose for their GPU.
static const unsigned AUTO_SIZES[][2] = {{1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}};

static void auto_pick(int temporal, int screen, unsigned out[2]) {
    out[0] = 1920, out[1] = 1080;
    if (screen <= 0) return;   // unknown: the default
    out[0] = 1280, out[1] = 720;
    for (int i = 1; i < (temporal ? 4 : 2); i++)
        if ((int)AUTO_SIZES[i][1] <= screen) out[0] = AUTO_SIZES[i][0], out[1] = AUTO_SIZES[i][1];
}

// The height auto fits: the screen in fullscreen, else the window's client height (window_size).
static int auto_height(void) {
    if (cfg_bool("AC_FULLSCREEN")) return gfx_display_height();
    unsigned w = 0, h = 0;
    char tail;
    if (sscanf(cfg_value("AC_WINDOW_SIZE"), "%ux%u%c", &w, &h, &tail) == 2 && h >= 720) return (int)h;
    return gfx_display_height();
}

static char asked[16];           // output_res as res_size read it ("" before)
static unsigned size_w, size_h;  // the size res_size decided (0 until then)
static int auto_raised;          // output_res = auto took more than it would without an upscaler

const char *res_output_asked(void) { return asked; }

// For the menu, each frame: the size auto would take at the next start, read again only when a setting changed (the
// file check behind upscaler_at_start looks at the file system).
void res_auto_next(unsigned out[2]) {
    static unsigned seen = ~0u, next[2];
    if (cfg_version() != seen) {
        seen = cfg_version();
        auto_pick(upscaler_at_start(), auto_height(), next);
    }
    out[0] = next[0], out[1] = next[1];
}

// Decided once, from mem_init (which sizes the pool for it) before the device exists; res_settle may lower an auto size
// before the game runs.
static void res_size(unsigned *w, unsigned *h) {
    if (!size_w) {
        const char *o = cfg_value("AC_OUTPUT_RES");
        snprintf(asked, sizeof asked, "%s", o);
        unsigned sw = 1920, sh = 1080;
        if (!strcmp(o, "auto")) {
            int temporal = upscaler_at_start(), screen = auto_height();
            unsigned plain[2], got[2];
            auto_pick(0, screen, plain);
            auto_pick(temporal, screen, got);
            sw = got[0], sh = got[1];
            auto_raised = got[0] != plain[0];
            rt_log("resolution: output_res auto: %ux%u for a %d-line %s%s", sw, sh, screen,
                   cfg_bool("AC_FULLSCREEN") ? "screen" : "window", temporal ? " with an upscaler from the start" : "");
        } else if (!res_output_grid(o, &sw, &sh) && strcmp(o, "1920x1080")) {
            rt_log("resolution: output_res = %s is not auto or a 16:9 size of 32n x 18n from 1280x720 to 3840x2160; "
                   "1920x1080", o);
        }
        size_h = sh, size_w = sw;
    }
    *w = size_w, *h = size_h;
}

// The render-target budget for w x h: the game's own up to 1080p's pixel count, which it fits; above, the measured use
// (87 MiB and 110 bytes a pixel: 304 MiB at 1920x1080, 482 at 2560x1440, 955 at 3840x2160, the same 81 allocations over
// 75 s into play) with 15 % to spare, in whole MiB: 545 MiB at 2560x1440, 1100 at 3840x2160.
static uint64_t res_budget(unsigned w, unsigned h) {
    uint64_t px = (uint64_t)w * h;
    if (px <= 1920ull * 1080) return RES_BUDGET_GAME;
    uint64_t use = (87ull << 20) + px * 110;
    return (use + use * 15 / 100 + 0xfffff) & ~0xfffffull;
}

// The res set is in the generated C and nothing turned it off: AC_RES=0, or AC_RES1062=1, whose sites are the same.
static int res_set_wanted(void) {
    if (output_set() || rt_patch_set("res") < 0) return 0;
    const char *r = cfg_get("AC_RES"), *r1062 = cfg_get("AC_RES1062");
    return !(r && !atoi(r)) && !(r1062 && atoi(r1062));
}

// What the GRAPHIC BUFFER heap gets on top of its 143 MiB for w x h. The game makes it at main+0x575c94 inside BUFFER
// ROOT (main+0x575a30, 173.375 MiB) and keeps its constant buffers there, next to one storage buffer sized by the
// render size: 6.97 bytes a pixel (13.8 MiB at 1920x1080, 24.5 at 2560x1440, 55.2 at 3840x2160). Measured in File 02
// at the footprint tutorial, the heap held 95.5 MiB at 1920x1080, 106.2 at 2560x1440 and 136.7 at 3840x2160, where its
// largest free block was under 1 KiB while the game makes about ten constant buffers a frame there: uniform data then
// landed on the headers of other blocks, and the next free of a neighbour crashed in the allocator (main+0x9bb9f0
// writing 0x3f800018, or main+0x9bafa0 reading -1), with or without an upscaler and in every AC_MEMORY mode. None of
// the game's other 37 heaps changes with the size. So above 1080p's pixel count the heap and its root grow by 8 bytes
// a pixel, in whole MiB (13 MiB at 2560x1440, 48 at 3840x2160), which leaves more free space than 1920x1080 has.
static uint64_t res_gbuf_extra(unsigned w, unsigned h) {
    uint64_t px = (uint64_t)w * h;
    if (px <= 1920ull * 1080) return 0;
    return ((px - 1920ull * 1080) * 8 + 0xfffff) & ~0xfffffull;
}

uint64_t res_output_extra_bytes(void) {
    if (!res_set_wanted()) return 0;
    unsigned w, h;
    res_size(&w, &h);
    return res_budget(w, h) - RES_BUDGET_GAME + res_gbuf_extra(w, h);
}

static unsigned res_page(int i) {
    uint32_t v;
    memcpy(&v, (const uint8_t *)rt_patch_page() + RES_PAGE + 4 * i, sizeof v);
    return v;
}

// The res set's sites: a mov of the game's render width or height, the ladder's cmp of the scene target's width
// against it (flags as cmp w0, width sets them), and the budget's mov.
void rt_res_w8(Cpu *c) { c->x[8] = res_page(0); }
void rt_res_h8(Cpu *c) { c->x[8] = res_page(1); }
void rt_res_w9(Cpu *c) { c->x[9] = res_page(0); }
void rt_res_h9(Cpu *c) { c->x[9] = res_page(1); }
void rt_res_h22(Cpu *c) { c->x[22] = res_page(1); }
uint64_t rt_res_cmp_w0(Cpu *c) {
    uint32_t a = (uint32_t)c->x[0], b = res_page(0), r = a - b;
    return MKFLAGS(r >> 31, r == 0, a >= b, ((a ^ b) & (a ^ r)) >> 31);
}
void rt_res_budget_w8(Cpu *c) { c->x[8] = res_budget(res_page(0), res_page(1)); }
// The sizes of BUFFER ROOT and of the GRAPHIC BUFFER heap inside it (res_gbuf_extra).
void rt_res_bufroot_w1(Cpu *c) { c->x[1] = 0xad60000ull + res_gbuf_extra(res_page(0), res_page(1)); }
void rt_res_gbuf_w1(Cpu *c) { c->x[1] = 0x8f00000ull + res_gbuf_extra(res_page(0), res_page(1)); }

static void res_pick(void) {
    int res = rt_patch_set("res"), r1062 = rt_patch_set("res1062");
    if (res < 0) return;
    if (rt_patch_enabled(r1062) && rt_patch_enabled(res)) {
        rt_patch_switch(res, 0);
        rt_log("resolution: AC_RES1062 asks for 1888x1062: the res set is off");
    }
    if (rt_patch_enabled(res)) {
        unsigned w, h;
        res_size(&w, &h);
        uint32_t page[2] = {w, h};
        memcpy((uint8_t *)rt_patch_page() + RES_PAGE, page, sizeof page);
        rt_log("resolution: the game renders %ux%u (render-target budget %llu MiB, graphic buffer heap %llu MiB)", w, h,
               (unsigned long long)(res_budget(w, h) >> 20),
               (unsigned long long)((0x8f00000ull + res_gbuf_extra(w, h)) >> 20));
    } else if (rt_patch_enabled(r1062)) {
        rt_log("resolution: the game renders 1888x1062");
    }
}

void res_game_size(unsigned out[2]) {
    if (rt_patch_enabled(rt_patch_set("res"))) {
        res_size(&out[0], &out[1]);
    } else if (rt_patch_enabled(rt_patch_set("res1062"))) {
        out[0] = 1888, out[1] = 1062;
    } else {
        unsigned scene[2];
        res_sizes(out, scene);
    }
}
// ---- 1.0.1: the scene size for upscaling (the scene set, recomp/patches-v101.txt). The game resolution (res, res1062
// or the game's own) stays the output, where the HUD is drawn; only the scene shrinks, to the upscaling mode's share of
// it: the game's dynamic-resolution levels all take the size on page +16 ({w, h} as floats, read by rt_res_scene_w1 /
// rt_res_scene_h2 below), and the game's own scaling pass (SMAA's blending) draws the scene to the output, where
// vk_dlaa.c runs the upscaler in its place. The scene is never larger than the targets the game already makes, so the
// render-target budget needs nothing more. Decided once at start: the game makes its scene targets then.
static struct {
    unsigned w, h;     // the scene the game renders (the game resolution unless reduced)
    unsigned gw, gh;   // the game resolution
    int reduced;       // the scene set is on: the scene is smaller than the game resolution
    int set;           // the scene set (rt_patch_set), -1 when the generated C lacks it
    char mode[24];     // the upscaling mode that sized it ("native" when not reduced)
} S = {.set = -1};

// The scene's share of the game resolution per mode, DLSS's and AMD's (XeSS's run close: vk_xess.c picks its setting by
// the size). The scene is a size on the 32 x 18 grid the game takes (resolution.c res_output_grid; off it the first 3D
// frame lost the device on an RTX 5070 Ti: 1114x626, 1120x626, 1136x640 and 1104x621 did, 1120x630 played), so n =
// round(N x share) for a game resolution of 32N x 18N, at most N - 1. At least 640x360 (n = 20): the smallest scene
// measured in play, and the scene depth vk_dlaa.c recognises is at least 640 wide. In handheld at most 1280x720
// (n = 40): the game made its handheld scene targets at that size, and a level larger than them is never made, which
// crashed loading a save (main+0x9d0df0, issues #25 and #38). At 1920x1080: 1280x720, 1120x630, 960x540, 640x360.
static double mode_share(const char *m) {
    static const struct { const char *name; double k; } modes[] = {
        {"quality", 2.0 / 3.0}, {"balanced", 0.58}, {"performance", 0.5}, {"ultra_performance", 1.0 / 3.0}};
    for (size_t i = 0; i < sizeof modes / sizeof *modes; i++)
        if (!strcmp(m, modes[i].name)) return modes[i].k;
    return 0;
}

static void scene_for(const char *mode, const unsigned g[2], unsigned out[2]) {
    out[0] = g[0], out[1] = g[1];
    double k = mode_share(mode);
    unsigned big = g[0] / 32;
    if (!k || g[0] % 32 || g[1] % 18 || g[1] / 18 != big) return;   // native, or a game size off the grid (res1062)
    int n = (int)lround(big * k), most = (int)big - 1;
    if (ov_handheld() && most > 40) most = 40;
    if (n > most) n = most;
    if (n < 20) n = 20;
    if (n > most) return;   // no smaller size on the grid
    out[0] = 32u * (unsigned)n, out[1] = 18u * (unsigned)n;
}

void res_scene_for(const char *mode, unsigned out[2]) {
    unsigned g[2];
    res_game_size(g);
    scene_for(S.set >= 0 ? mode : "native", g, out);
}

void res_scene_size(unsigned out[2]) {
    if (S.gw) out[0] = S.w, out[1] = S.h;
    else res_game_size(out);
}

int res_scene_reduced(void) { return S.reduced; }

static void scene_apply(unsigned w, unsigned h) {
    S.w = w, S.h = h, S.reduced = w < S.gw;
    float sc[2] = {(float)w, (float)h};
    memcpy((uint8_t *)rt_patch_page() + 16, sc, sizeof sc);   // before the switch: a site that sees it on reads the size
    rt_patch_switch(S.set, S.reduced);
}

static void scene_init(void) {
    unsigned g[2];
    res_game_size(g);
    S.gw = S.w = g[0], S.gh = S.h = g[1];
    S.set = rt_patch_set("scene");
    const char *q = cfg_upscaling_quality();
    snprintf(S.mode, sizeof S.mode, "native");
    if (!mode_share(q)) return;
    const char *force = cfg_get("AC_PATCH_SCENE");   // AC_PATCH_SCENE=0 (test): the game's own levels
    if (S.set < 0 || (force && !strcmp(force, "0"))) {
        rt_log("resolution: upscaling_quality = %s needs the scene patch set, which %s: the scene keeps the game's size",
               q, S.set < 0 ? "the generated C lacks" : "AC_PATCH_SCENE=0 keeps off");
        return;
    }
    if (!upscaler_at_start()) {
        const char *up = cfg_upscaling();
        if (strcmp(up, "off"))
            rt_log("resolution: upscaling = %s cannot run (%s): the scene keeps the game's size", up,
                   !strcmp(up, "dlss") ? "nvngx_dlss.dll from NVIDIA is not next to astralchain.exe"
                   : !strcmp(up, "xess") ? "libxess.dll is not next to astralchain.exe"
                                         : "not on this system");
        return;
    }
    unsigned s[2];
    scene_for(q, g, s);
    if (s[0] >= S.gw) return;
    snprintf(S.mode, sizeof S.mode, "%s", q);
    scene_apply(s[0], s[1]);
    rt_log("resolution: upscaling %s: the scene at %ux%u, %.0f %% of the game's %ux%u", q, s[0], s[1],
           100.0 * s[0] / S.gw, S.gw, S.gh);
}

// The scene set's sites (each a csel of a level's width into w1 or height into w2).
static unsigned page_size(int i) {
    float v;
    memcpy(&v, (const uint8_t *)rt_patch_page() + 16 + 4 * i, sizeof v);
    return v >= 64.0f && v <= SCENE_MAX ? (unsigned)v : 64u;
}
void rt_res_scene_w1(Cpu *c) { c->x[1] = page_size(0); }
void rt_res_scene_h2(Cpu *c) { c->x[2] = page_size(1); }

// After the device exists and before the game runs (boot.c rt_boot): the upscaler that will really run is known now
// (vk_dlaa.c vk_dlaa_upscaler_ok: NGX answered, libxess.dll came up on this GPU, FSR 3's formats). Without one, a
// reduced scene goes back to the game resolution's size, and an output_res = auto raised for it goes back to what it
// takes without an upscaler: the game's plain scaling at 33..67 %, or SMAA at the screen's size, would be the result.
// vram_mib: the GPU's largest device-local heap; auto keeps at most 2560x1440 at 4.5 GiB or less (the game's render
// targets take about 955 MiB at 3840x2160, before the textures). The sites read the page each time they run and the
// game makes no target before this, so rewriting the page is enough; the pool keeps its larger size, untouched.
void res_settle(int upscaler_ok, unsigned vram_mib) {
    if (!rt_patch_enabled(rt_patch_set("res")) || output_set()) return;
    unsigned w = size_w, h = size_h;
    if (auto_raised) {
        unsigned plain[2];
        auto_pick(0, auto_height(), plain);
        if (!upscaler_ok) {
            w = plain[0], h = plain[1];
            rt_log("resolution: output_res auto: no upscaler can run here: %ux%u, not %ux%u", w, h, size_w, size_h);
        } else if (vram_mib && vram_mib <= 4608 && w > 2560) {
            w = 2560, h = 1440;
            rt_log("resolution: output_res auto: %u MiB of video memory: 2560x1440, not %ux%u", vram_mib, size_w, size_h);
        }
    }
    if (w != size_w || h != size_h) {
        size_w = w, size_h = h, auto_raised = 0;
        uint32_t page[2] = {w, h};
        memcpy((uint8_t *)rt_patch_page() + RES_PAGE, page, sizeof page);
        S.gw = w, S.gh = h;
        unsigned g[2] = {w, h}, s[2];
        scene_for(S.mode, g, s);
        scene_apply(s[0], s[1]);
    }
    if (S.reduced && !upscaler_ok) {
        rt_log("resolution: the upscaler cannot run on this GPU: the scene keeps the game's size, %ux%u, not %ux%u",
               S.gw, S.gh, S.w, S.h);
        snprintf(S.mode, sizeof S.mode, "native");
        scene_apply(S.gw, S.gh);
    }
}

static void wide_init(void);
static void hud_init(void);

void res_init(void) {
    res_pick();
    wide_init();
    hud_init();
    res_shadow_range();
    if (!output_set()) {   // 1.0.1: the res set took output_res above, the scene set scales the scene for upscaling
        scene_init();
        shadow_init(rt_patch_page());
        return;
    }
    if (!bigger()) {
        rt_log("resolution: the game's own (output 1920x1080, scene 1600x900)");
        shadow_init(rt_patch_page());
        return;
    }
    int output = rt_patch_set("output"), scene = rt_patch_set("scene");
    if (output < 0 || scene < 0) {
        rt_log("resolution: the generated C has no output and scene patch sets; the game keeps its own resolution");
        return;
    }
    fs_init();   // romfs.c serves the grown HeapDef from here on (res_heapdef_served)
    if (res_heap_extra_kib() && !R.heapdef_ok) {   // sizes at or below the shipped ones fit the shipped heaps
        rt_log("resolution: no grown HeapDef table; the game keeps its own resolution");
        return;
    }
    uint8_t *page = rt_patch_page();
    double k = RT_V143 ? (double)R.ow / BASE_W : 1.0;   // 1.4.3's own scaling of the scene with the output width
    int scene_on = R.scene_set || (!RT_V143 && (R.ow != BASE_W || R.oh != BASE_H));
    float sc[2] = {(float)(R.sw / k), (float)(R.sh / k)};
    uint64_t out = pack(R.ow, R.oh), none = 0;
    memcpy(page + 16, sc, sizeof sc);
    memcpy(page + 32, &out, 8);
    memcpy(page + 40, &none, 8);
    memcpy(page + 48, &out, 8);
    rt_patch_switch(output, R.ow != BASE_W || R.oh != BASE_H);
    rt_patch_switch(scene, scene_on);
    R.on = 1;
    // Another shape than 16:9 (more than 1% off): the camera takes the output's ratio (recomp/patches-v121.txt and
    // patches-v143.txt, set aspect), so a 21:9 output shows more of the world at the sides instead of a stretched 16:9 view. The HUD's layouts
    // for that shape come from romfs-patches-<w>x<h> (romfs.c overlay_init, tools/make_romfs_patches.py --aspect).
    double ratio = (double)R.ow / R.oh, game = (double)BASE_W / BASE_H;
    int aspect = rt_patch_set("aspect");
    if (!res_is_16x9(R.ow, R.oh)) {
        if (aspect < 0) {
            rt_log("resolution: output %ux%u is not 16:9 and this game version has no camera aspect patch; the view "
                   "is stretched", R.ow, R.oh);
        } else {
            float r = (float)ratio;
            memcpy(page + 56, &r, sizeof r);
            rt_patch_switch(aspect, 1);
            rt_log("resolution: camera aspect ratio %.4f (16:9 is %.4f)", ratio, game);
        }
    } else {
        rt_patch_switch(aspect, 0);   // the page holds no ratio
    }
    shadow_init(page);
    rt_log("resolution: output %ux%u, scene %ux%u%s; texture heap +%llu KiB", R.ow, R.oh, R.sw, R.sh,
           scene_on ? "" : " (the game's own 5/6 of the output)", (unsigned long long)res_heap_extra_kib());
}

// Upscaling: textures are sampled for the output's resolution, not the scene's (NVIDIA's DLSS guide, and AMD's and
// Intel's: a mip bias of log2(render / output)), so their detail survives the scaling. 0 while the scene is the game
// resolution's size; vk_dlaa.c applies it only while an upscaler scales (the game's plain scaling of a biased scene
// would only shimmer).
float res_dlss_lod_bias(void) {
    return S.reduced ? (float)log2((double)S.w / S.gw) : 0.0f;
}

// 16:9 within 1% (1366x768 counts): the shape the game's own layouts and camera are made for.
int res_is_16x9(unsigned w, unsigned h) {
    double rel = ((double)w / h) / ((double)BASE_W / BASE_H);
    return rel >= 0.99 && rel <= 1.01;
}

// The game resolution asks for a romfs-patches-<w>x<h> overlay: it is not 16:9 and the game version has the camera
// aspect patch (set aspect); the UI layouts of another shape only fit a camera that follows the same shape.
int res_wants_wide_overlay(const unsigned out[2]) {
    return !res_is_16x9(out[0], out[1]) && rt_patch_set("aspect") >= 0;
}

// Field of view (AC_FOV, percent of the game's own, 70..140, live): the camera lens's vertical view angle times the
// factor on page +60 where the projection matrix is built from it, and where the scene setup copies it for its
// screen-size work (recomp/patches-v101.txt, set fov: rt_fov_s0). The horizontal angle follows through the aspect
// ratio. Cutscenes keep their authored framing: while the game's cutscene flag is set (rt_fov_s0 reads it directly) the
// angle passes unscaled, and the lens it builds then is the cutscene camera's. 100 switches the set off, so the camera
// code is the game's own. Runs on a settings change (gfx_present.c window_settings_live), the first time at the first
// present.
#define FOV_MIN 70
#define FOV_MAX 140

void rt_fov_s0(Cpu *c) {
    // The game's own cutscene flag (bit 30 of the word at main+0xDB8398), not frame_step_cutscene, which only follows
    // it while the 60 fps patch is on.
    uint32_t volatile *flags = *(uint32_t volatile *volatile *)((uintptr_t)MOD_MAIN_BASE + 0xDB8398ull);
    if (flags && (*flags >> 30 & 1)) return;   // the shot's own framing
    float k;
    memcpy(&k, (const uint8_t *)rt_patch_page() + 60, sizeof k);
    FP_SETS(0, VSF32(0) * k);
}

// The fov set's sites where a lens's projection is built (main+0x9a3010, lens x24; main+0x9a3054, lens x8) are the
// wide set's too, and the fov set comes first in the file, so its helpers also give a taller frame's lens its angle.
static void wide_tall_angle(Cpu *c, const uint8_t *lens);
void rt_fov_lens24_s0(Cpu *c) {
    rt_fov_s0(c);
    wide_tall_angle(c, (const uint8_t *)(uintptr_t)c->x[24]);
}
void rt_fov_lens8_s0(Cpu *c) {
    rt_fov_s0(c);
    wide_tall_angle(c, (const uint8_t *)(uintptr_t)c->x[8]);
}

void res_fov_live(void) {
    static int set = -2, shown = 100;
    if (set == -2) set = rt_patch_set("fov");
    int pct = cfg_int("AC_FOV");
    pct = pct < FOV_MIN ? FOV_MIN : pct > FOV_MAX ? FOV_MAX : pct;
    if (pct == shown) return;
    shown = pct;
    if (set < 0) {
        rt_log("resolution: this game version has no field-of-view patch; AC_FOV=%d does nothing", pct);
        return;
    }
    float k = pct / 100.0f;
    memcpy((uint8_t *)rt_patch_page() + 60, &k, sizeof k);   // before the switch: a site that sees it on reads k
    rt_patch_switch(set, pct != 100);
    rt_log("resolution: field of view %d%% of the game's", pct);
}

void res_wanted(unsigned out[2]) {
    decide();
    out[0] = R.ow, out[1] = R.oh;
}

void res_sizes(unsigned out[2], unsigned scene[2]) {
    decide();
    if (R.on) {
        out[0] = R.ow, out[1] = R.oh;
        scene[0] = R.sw, scene[1] = R.sh;
        return;
    }
    out[0] = BASE_W, out[1] = BASE_H;
    scene[0] = SCENE_W, scene[1] = SCENE_H;
}

// ---- wide pictures (the aspect setting's wide; recomp/patches-v101.txt, set wide). The game's targets stay 16:9; a
// 16:9 lens takes the frame's ratio where its matrices are built (rt_wide_dirty_w8), and the present shows the frame in
// that shape (res_wide_tick hands it to vk_backend.c with the present). The frame's ratio is the window's, from 4:3 to
// 48:9; a window within 1 % of 16:9 keeps 16:9, and so does every frame while a full-screen movie plays (frame_rate.c
// rt_movie_showing), which is a 16:9 picture. It changes between frames only, at the game's present, so the lenses of
// one frame and its present agree. A wider frame shows more of the world at the sides; a taller one (16:10, the Steam
// Deck's 1280x800) keeps the 16:9 view's width and shows more above and below: its lenses get a vertical angle widened
// to match where the projection is built (rt_wide_angle24_s0). The interface camera's lens takes the ratio too: what
// is drawn through it keeps its proportions, in the middle 16:9 of the picture (ui_camera_refresh, rt_wide_view_x8).
// The set is switched on the first time the setting is wide (the default) in a window of another shape than 16:9, and
// stays on, so lenses go back to 16:9 through the same site when the setting leaves wide; in a 16:9 window the game's
// code runs as it is.
#define WIDE_16X9 (16.0f / 9.0f)
#define WIDE_MAX (48.0f / 9.0f)
#define TALL_MIN (4.0f / 3.0f)
#define LENS_VIEW_W 0x268    // the lens's viewport width and height (int32), as main+0x9a30a0 got them
#define LENS_VIEW_H 0x26c
#define LENS_ASPECT 0x278    // float
#define UI_CAMERA 0xDC1910ull   // main: the pointer to the interface camera's handle (main+0x9044f0)
#define FLAT_CAMERAS 0xDC2D50ull   // main: the pointer to the pointer to the flat cameras' array (main+0x9e4030)

static struct {
    int set;            // the wide set (rt_patch_set), -1 when the generated C lacks it
    int on;             // the set is on
    unsigned seen;      // cfg_version the setting was read at
    int wanted;         // the aspect setting is wide
    int told;           // the missing set was logged
    uint32_t now;       // the ratio 16:9 lenses take in the frame the game is building (a float's bits)
    uint32_t window;    // the window's width / height (a float's bits; res_wide_window), 0 before the first swapchain
} W = {.set = -1, .seen = ~0u};

static float bits_f(const uint32_t *p) {
    uint32_t b = __atomic_load_n(p, __ATOMIC_RELAXED);
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

static void f_bits(uint32_t *p, float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    __atomic_store_n(p, b, __ATOMIC_RELAXED);
}

// The present thread, for each new swapchain: the window's shape.
void res_wide_window(unsigned w, unsigned h) {
    if (h) f_bits(&W.window, (float)w / (float)h);
}

// The ratio the window asks for: its own shape, more than 1 % off 16:9, from 4:3 to 48:9. Before the first swapchain,
// the window size setting's when the window is not fullscreen.
static float wide_target(void) {
    if (!W.wanted) return WIDE_16X9;
    float r = bits_f(&W.window);
    if (r <= 0 && !cfg_bool("AC_FULLSCREEN")) {
        unsigned w = 0, h = 0;
        char tail;
        if (sscanf(cfg_value("AC_WINDOW_SIZE"), "%ux%u%c", &w, &h, &tail) == 2 && h) r = (float)w / (float)h;
    }
    if (r <= 0 || (r <= WIDE_16X9 * 1.01f && r >= WIDE_16X9 / 1.01f)) return WIDE_16X9;
    return r > WIDE_MAX ? WIDE_MAX : r < TALL_MIN ? TALL_MIN : r;
}

static void wide_settings(void) {
    unsigned v = cfg_version();
    if (v == W.seen) return;
    W.seen = v;
    W.wanted = !strcmp(cfg_value("AC_ASPECT"), "wide");
}

// 1 once the set is on: from the first time the setting is wide and the window of another shape than 16:9.
static int wide_arm(void) {
    if (W.on) return 1;
    if (!W.wanted || wide_target() == WIDE_16X9) return 0;
    if (W.set < 0) {
        if (!W.told)
            rt_log("resolution: aspect = wide needs the wide patch set, which the generated C lacks: the picture is "
                   "fitted");
        W.told = 1;
        return 0;
    }
    W.on = 1;
    rt_patch_switch(W.set, 1);
    rt_log("resolution: wide pictures on: 16:9 lenses take the window's shape, %.4f (aspect = wide)", wide_target());
    return 1;
}

static void wide_init(void) {
    W.set = rt_patch_set("wide");
    f_bits(&W.now, WIDE_16X9);
    wide_settings();
    if (wide_arm()) f_bits(&W.now, wide_target());
}

// The game's present (vk_backend.c vk_be_present, the game's thread): the shape of the frame it presents, 0 for the
// picture's own; then the ratio of the frame it builds next.
float res_wide_tick(void) {
    wide_settings();
    if (!wide_arm()) return 0;
    // Logged whatever the trace: when a movie takes the picture to 16:9 and when it gives it back, with the open
    // movies, and once more when it has kept it for MOVIE_LOG_PRESENTS presents (half a minute at 60 fps), so a log
    // tells which movie keeps a scene between bars.
    enum { MOVIE_LOG_PRESENTS = 1800 };
    static int movie_was;
    static unsigned movie_presents;
    int movie = rt_movie_showing();
    if (movie != movie_was || (movie && ++movie_presents == MOVIE_LOG_PRESENTS)) {
        char list[256];
        rt_movie_list(list, sizeof list);
        if (movie != movie_was)
            rt_log(movie ? "resolution: wide: 16:9 while a movie shows (%s)"
                         : "resolution: wide: the movie is gone, back to the window's shape (%s)", list);
        else
            rt_log("resolution: wide: still 16:9 after %d presents of a movie (%s)", MOVIE_LOG_PRESENTS, list);
        if (movie != movie_was) movie_presents = 0;
        movie_was = movie;
    }
    float shown = bits_f(&W.now), next = movie ? WIDE_16X9 : wide_target();
    if (next != shown) {
        f_bits(&W.now, next);
        if (rt_trace & RT_TRACE_PERF) rt_log("resolution: wide: lens ratio %.4f -> %.4f", shown, next);
    }
    return shown;
}

static int lens_16x9(const uint8_t *lens) {
    int32_t w, h;
    memcpy(&w, lens + LENS_VIEW_W, sizeof w);
    memcpy(&h, lens + LENS_VIEW_H, sizeof h);
    return w > 0 && (int64_t)w * 9 == (int64_t)h * 16;
}

// The interface camera's matrices are built once, at main+0x9045f4, from the position the message system gives it
// there: 2040 units in front of the middle of the 1280x720 plane, looking at it, y down. When its lens holds another
// ratio than the frame's, the same call again (main+0x9a2fa0: handle, eye, target, up, 0, 1), from a game thread that
// is building a game camera's matrices: the call comes back through rt_wide_dirty_w8 with the interface lens, which
// takes the ratio there. The registers the caller still needs are put back.
static void ui_camera_refresh(Cpu *c, float want) {
    static const _Alignas(16) float eye[4] = {640, 360, -2040, 1}, target[4] = {640, 360, -2039, 1},
                                    up[4] = {0, -1, 0, 1};
    uint64_t handle = *(const uint64_t *)((uintptr_t)MOD_MAIN_BASE + UI_CAMERA);
    const uint8_t *lens = handle ? *(const uint8_t *const *)(uintptr_t)handle : NULL;
    float have;
    if (!lens || !lens_16x9(lens)) return;   // not made yet, or gone
    memcpy(&have, lens + LENS_ASPECT, sizeof have);
    if (have == want) return;
    uint64_t x[31], nzcv = c->nzcv;
    __uint128_t v[32];
    memcpy(x, c->x, sizeof x);
    memcpy(v, c->v, sizeof v);
    rt_call(c, (uint64_t)MOD_MAIN_BASE + 0x9a2fa0ull, 6, handle, (uint64_t)(uintptr_t)eye, (uint64_t)(uintptr_t)target,
            (uint64_t)(uintptr_t)up, (uint64_t)0, (uint64_t)1);
    memcpy(c->x, x, sizeof x);
    memcpy(c->v, v, sizeof v);
    c->nzcv = nzcv;
}

// main+0x9a3004 (ldr w8,[x24,#0x28c]), in the function that builds a lens's matrices: x24 = the lens, x23 = its
// handle. A lens made 16:9 (its viewport says so, whatever ratio it holds now) takes the frame's ratio; when that
// differs from the one it holds, the projection is built anew as for a dirty lens. w8 = the lens's dirty word.
void rt_wide_dirty_w8(Cpu *c) {
    uint8_t *lens = (uint8_t *)(uintptr_t)c->x[24];
    uint32_t dirty;
    memcpy(&dirty, lens + 0x28c, sizeof dirty);
    if (lens_16x9(lens)) {
        int ui = c->x[23] == *(const uint64_t *)((uintptr_t)MOD_MAIN_BASE + UI_CAMERA);
        float want = bits_f(&W.now), have;
        if (!ui) ui_camera_refresh(c, want);
        memcpy(&have, lens + LENS_ASPECT, sizeof have);
        if (have != want) {
            memcpy(lens + LENS_ASPECT, &want, sizeof want);
            dirty = 1;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("resolution: wide: %s lens %p ratio %.4f -> %.4f", ui ? "interface" : "game", (void *)lens, have,
                       want);
        }
    }
    c->x[8] = dirty;
}

// A taller picture than 16:9 keeps the 16:9 view's width: the vertical angle's tangent grows by 16:9 over the ratio.
// s0 = the vertical angle (radians) loaded for a lens's projection; widened when the lens is made 16:9 and holds a
// taller ratio (rt_wide_dirty_w8 gave it the frame's). Also called from the fov set's helpers on the same sites
// (rt_fov_lens24_s0), which win over these while a field of view is set (patches.py: the first set in file order).
static void wide_tall_angle(Cpu *c, const uint8_t *lens) {
    float ratio;
    memcpy(&ratio, lens + LENS_ASPECT, sizeof ratio);
    if (ratio > 0 && ratio < WIDE_16X9 / 1.001f && lens_16x9(lens))
        FP_SETS(0, 2.0f * atanf(tanf(VSF32(0) * 0.5f) * (WIDE_16X9 / ratio)));
}

// main+0x9a3010 (ldr s0,[x24,#0x27c]; x24 = the lens) and main+0x9a3054 (ldr s0,[x8,#0x27c]; x8 = the lens), where a
// lens's projection is built from its vertical angle: after the load, the angle for the lens's ratio.
void rt_wide_angle24_s0(Cpu *c) { wide_tall_angle(c, (const uint8_t *)(uintptr_t)c->x[24]); }
void rt_wide_angle8_s0(Cpu *c) { wide_tall_angle(c, (const uint8_t *)(uintptr_t)c->x[8]); }

// main+0x9a320c and main+0x9a327c (add x8,x4,#0x260), in world to screen and screen to world: x4 = the lens, x8 = its
// viewport {x, y, width, height} (int32). For a lens that holds a wider ratio than its viewport's, a viewport of the
// ratio's width at the same height, centred on the lens's own; for a taller one, of the ratio's height at the same
// width: the interface plane's units for that picture.
void rt_wide_view_x8(Cpu *c) {
    static _Thread_local int32_t view[4];   // read by the callee these sites are the arguments of, on this thread
    uint8_t *lens = (uint8_t *)(uintptr_t)c->x[4];
    int32_t own[4];
    float ratio;
    memcpy(own, lens + 0x260, sizeof own);
    memcpy(&ratio, lens + LENS_ASPECT, sizeof ratio);
    c->x[8] = (uint64_t)(uintptr_t)(lens + 0x260);
    if (!lens_16x9(lens) || ratio <= 0) return;
    if (ratio > WIDE_16X9 * 1.001f) {
        view[2] = (int32_t)lroundf((float)own[3] * ratio);
        view[3] = own[3];
        view[0] = own[0] + (own[2] - view[2]) / 2;
        view[1] = own[1];
    } else if (ratio < WIDE_16X9 / 1.001f) {
        view[2] = own[2];
        view[3] = (int32_t)lroundf((float)own[2] / ratio);
        view[0] = own[0];
        view[1] = own[1] + (own[3] - view[3]) / 2;
    } else {
        return;
    }
    c->x[8] = (uint64_t)(uintptr_t)view;
}

// main+0x99132c, in the visibility cone test: s0 = the lens's vertical angle, from which the game takes half plus 20
// degrees as the cone's half angle, enough for a 16:9 picture's corners. The angle that gives a 16:9 picture the width
// a wider frame shows, or the height a taller one shows, so the cone holds that picture's corners as well.
void rt_wide_cone_s0(Cpu *c) {
    float r = bits_f(&W.now);
    if (r > WIDE_16X9) FP_SETS(0, 2.0f * atanf(tanf(VSF32(0) * 0.5f) * (r / WIDE_16X9)));
    else if (r < WIDE_16X9) FP_SETS(0, 2.0f * atanf(tanf(VSF32(0) * 0.5f) * (WIDE_16X9 / r)));
}

// main+0x9e3f84 (ldr w8,[x19,#0x30]), in the function that writes a flat camera's matrices when it is dirty, called
// for each of them every frame: x19 = the camera, 0x40 bytes: its bounds {left, right, bottom, top, near, far} (floats)
// at +0x18, the dirty word at +0x30. The game draws through the first one only, the 1280x720 plane; the second one,
// which the set's sites bind for what is drawn onto the screen, is kept as the first one's plane at the width (or, for
// a taller frame, the height) this frame's ratio shows, around the same middle, and written as a dirty one when that
// changed. w8 = the camera's dirty word.
void rt_wide_flat_w8(Cpu *c) {
    uint8_t *cam = (uint8_t *)(uintptr_t)c->x[19];
    const uint8_t *first = **(uint8_t *const *const *)((uintptr_t)MOD_MAIN_BASE + FLAT_CAMERAS);
    uint32_t dirty;
    memcpy(&dirty, cam + 0x30, sizeof dirty);
    if (first && cam == first + 0x40) {
        float want[6], have[6], r = bits_f(&W.now), k = r > WIDE_16X9 ? r / WIDE_16X9 : 1,
                                                    t = r < WIDE_16X9 ? WIDE_16X9 / r : 1;
        memcpy(want, first + 0x18, sizeof want);
        memcpy(have, cam + 0x18, sizeof have);
        float mid = (want[0] + want[1]) * 0.5f, half = (want[1] - want[0]) * 0.5f * k;
        want[0] = mid - half, want[1] = mid + half;
        mid = (want[2] + want[3]) * 0.5f, half = (want[3] - want[2]) * 0.5f * t;
        want[2] = mid - half, want[3] = mid + half;
        if (memcmp(have, want, sizeof want)) {
            memcpy(cam + 0x18, want, sizeof want);
            dirty = 1;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("resolution: wide: flat camera %p columns %.1f..%.1f rows %.1f..%.1f -> %.1f..%.1f, %.1f..%.1f",
                       (void *)cam, have[0], have[1], have[2], have[3], want[0], want[1], want[2], want[3]);
        }
    }
    c->x[8] = dirty;
}

// ---- the interface's blocks at the picture's edges. A layout instance (0x1e0 bytes, main+0x90f180; its archive at
// +0xb4, its uid at +0xb8, its own transform at +0x10) holds a tree of parts (0x470 bytes each: owner +0x08, the
// layout's place +0x1d8 and scale +0x1f0, the game's own offsets +0x220 and +0x2f0 and scale +0x238, the drawable
// +0x400, first child +0x428, next sibling +0x430). main+0x910f60 updates a part each frame: its world matrix is its
// own times its parent's, and its children follow.
#define PART_OWNER 0x08
#define PART_PLACE 0x1d8
#define PART_SCALE 0x1f0
#define PART_SCALE2 0x238
#define PART_DRAWN 0x400
#define PART_CHILD 0x428
#define PART_NEXT 0x430
#define PART_SIZE 0x470
#define INST_TRANSFORM 0x10   // position x, y, z at +8 of it
#define INST_ARCHIVE 0xb4
#define INST_UID 0xb8
#define INST_PARTS 0x150      // the parts' array, their count at +0x160
#define INST_PART_COUNT 0x160
#define UI_ARCHIVES 0xdc18f0ull   // main: the pointer to the table of interface archives
#define ANCHOR_SLOTS 1024
#define ANCHOR_PARTS 4096

static struct { const uint8_t *inst; uint32_t arc, uid; int8_t side[2]; } g_anchor[ANCHOR_SLOTS];
static unsigned g_anchor_n;

static float part_f(const uint8_t *p, unsigned off) {
    float f;
    memcpy(&f, p + off, sizeof f);
    return f;
}

// The columns (a = 0) or rows (a = 1), in the layout's own units from its origin, of the drawn parts under p (p's place
// at x on that axis, its scale sx).
static void anchor_span(const uint8_t *p, int a, float x, float sx, float *lo, float *hi, unsigned *n, int depth) {
    if (*(const uint64_t *)(p + PART_DRAWN)) {
        if (x < *lo) *lo = x;
        if (x > *hi) *hi = x;
    }
    if (++*n > ANCHOR_PARTS || depth > 24) return;
    unsigned o = 4u * (unsigned)a;
    for (const uint8_t *k = *(const uint8_t *const *)(p + PART_CHILD); k; k = *(const uint8_t *const *)(k + PART_NEXT))
        anchor_span(k, a, x + sx * part_f(k, PART_PLACE + o), sx * part_f(k, PART_SCALE + o) * part_f(k, PART_SCALE2 + o),
                    lo, hi, n, depth + 1);
}

// The name of the interface archive ("hud2", "menu", "core", ...) a layout instance's +0xb4 stands for, from the
// game's table of them (*(main+0xdc18f0): entries of 0x24 bytes at +8, {the number, the name}, their count at +0x18).
static const char *anchor_archive(uint32_t arc) {
    const uint8_t *tab = *(const uint8_t *const *)((uintptr_t)MOD_MAIN_BASE + UI_ARCHIVES);
    const uint8_t *ents = tab ? *(const uint8_t *const *)(tab + 8) : NULL;
    uint64_t count = tab ? *(const uint64_t *)(tab + 0x18) : 0;
    for (uint64_t i = 0; ents && i < count && i < 256; i++)
        if (!memcmp(ents + i * 0x24, &arc, sizeof arc)) return (const char *)ents + i * 0x24 + 4;
    return "";
}

// The side a layout instance's block sits on along axis a: for columns (a = 0) -1 left, 1 right, for rows (a = 1) -1
// top, 1 bottom (the plane's y grows downwards), 0 neither, from where the layout puts what it draws: the middle of
// the columns or rows its drawn parts start at, in the outer thirds of the 1280 columns or 720 rows or not. Decided
// once per instance and layout, when it is first updated, over all its top parts (the parts no other part has as a
// child), so a row across the screen stays whole; the places are the ones the layout's animation holds then, which a
// block that slides in from its own side does not change the answer of. Only the gameplay HUD's archive has blocks to
// move: a menu is one composition over the middle 16:9, and its parts are clipped to it.
static int anchor_side(const uint8_t *inst, int a) {
    uint32_t arc, uid;
    memcpy(&arc, inst + INST_ARCHIVE, sizeof arc);
    memcpy(&uid, inst + INST_UID, sizeof uid);
    unsigned h = (unsigned)(((uintptr_t)inst >> 4) * 2654435761u) % ANCHOR_SLOTS;
    for (unsigned i = 0; i < ANCHOR_SLOTS; i++, h = (h + 1) % ANCHOR_SLOTS) {
        if (g_anchor[h].inst == inst && g_anchor[h].uid == uid && g_anchor[h].arc == arc) return g_anchor[h].side[a];
        if (!g_anchor[h].inst || g_anchor[h].inst == inst) break;
    }
    if (g_anchor_n >= ANCHOR_SLOTS * 3 / 4) {
        memset(g_anchor, 0, sizeof g_anchor);
        g_anchor_n = 0;
        h = (unsigned)(((uintptr_t)inst >> 4) * 2654435761u) % ANCHOR_SLOTS;
    }
    const uint8_t *parts = *(const uint8_t *const *)(inst + INST_PARTS);
    uint64_t count = *(const uint64_t *)(inst + INST_PART_COUNT);
    float lo[2] = {1e9f, 1e9f}, hi[2] = {-1e9f, -1e9f};
    unsigned n = 0;
    if (parts && count && count <= ANCHOR_PARTS && !strcmp(anchor_archive(arc), "hud2")) {
        uint8_t child[ANCHOR_PARTS / 8] = {0};
        for (uint64_t i = 0; i < count; i++)
            for (const uint8_t *k = *(const uint8_t *const *)(parts + i * PART_SIZE + PART_CHILD); k;
                 k = *(const uint8_t *const *)(k + PART_NEXT)) {
                uint64_t j = (uint64_t)(k - parts) / PART_SIZE;
                if (k < parts || j >= count || child[j >> 3] >> (j & 7) & 1) break;   // not this array's, or a loop
                child[j >> 3] |= (uint8_t)(1u << (j & 7));
            }
        for (uint64_t i = 0; i < count; i++)
            if (!(child[i >> 3] >> (i & 7) & 1)) {
                const uint8_t *p = parts + i * PART_SIZE;
                for (int k = 0; k < 2; k++) {
                    unsigned o = 4u * (unsigned)k, m = 0;
                    anchor_span(p, k, part_f(p, PART_PLACE + o), part_f(p, PART_SCALE + o) * part_f(p, PART_SCALE2 + o),
                                &lo[k], &hi[k], &m, 0);
                    if (!k) n += m;
                }
            }
    }
    int8_t side[2];
    for (int k = 0; k < 2; k++) {
        float mid = (lo[k] + hi[k]) * 0.5f, third = (k ? 360.0f : 640.0f) / 3;
        side[k] = (int8_t)(lo[k] > hi[k] ? 0 : mid < -third ? -1 : mid > third ? 1 : 0);
    }
    if (!g_anchor[h].inst) g_anchor_n++;
    g_anchor[h].side[0] = side[0], g_anchor[h].side[1] = side[1];
    g_anchor[h].arc = arc, g_anchor[h].uid = uid, g_anchor[h].inst = inst;
    if (rt_trace & RT_TRACE_PERF)
        rt_log("resolution: wide: layout %s %04x at %p: columns %.0f..%.0f rows %.0f..%.0f of %u parts, side %d / %d",
               anchor_archive(arc), uid, (void *)inst, lo[0], hi[0], lo[1], hi[1], n, side[0], side[1]);
    return side[a];
}

// main+0x9111d8 (add x0,x19,#0x170), in the update of a layout part: x19 = the part, x21 = its parent's world
// transform, sp+0x10 = the part's world matrix, just made, which the part keeps and its children multiply by. A top
// part (its parent is its instance's transform) of a block that sits on one side moves to that side's edge of the
// picture: in a wider frame the 1280x720 plane shows 720 x ratio columns, so by half the difference along x; in a
// taller one 1280 / ratio rows, so by half the difference along y, for the blocks at the top and the bottom. Only an
// instance placed at the plane's middle is a screen layout; one placed elsewhere is put there by the game, on a place
// in the world or on a part of another layout, which has moved with its own block already.
void rt_wide_anchor(Cpu *c) {
    float r = bits_f(&W.now);
    if (r == WIDE_16X9) return;
    const uint8_t *p = (const uint8_t *)(uintptr_t)c->x[19];
    const uint8_t *own = *(const uint8_t *const *)(p + PART_OWNER);
    if (!own || c->x[21] != (uint64_t)(uintptr_t)(own + INST_TRANSFORM)) return;
    int a = r < WIDE_16X9;   // the axis: 0 columns (wider), 1 rows (taller)
    float move = a ? 640.0f / r - 360.0f : 360.0f * r - 640.0f;
    if (fabsf(part_f(own, INST_TRANSFORM + 8 + 4 * a) - (a ? 360.0f : 640.0f)) >= move * 0.5f) return;
    if (a && fabsf(part_f(own, INST_TRANSFORM + 8) - 640.0f) >= move * 0.5f) return;   // and at the middle column
    int side = anchor_side(own, a);
    if (!side) return;
    float *m = (float *)(uintptr_t)(c->sp + 0x10);
    m[12 + a] += (float)side * move;
}

// ---- the gameplay HUD switched off (the hud setting, AC_HUD, which F12 and the menu's Game tab switch;
// recomp/patches-v101.txt, set nohud). The set's two sites sit where the game sends a layout instance's parts to the
// renderer, and leave out the instances of the gameplay HUD's archive (hud2): nothing of them is drawn, while the game
// keeps updating them, so the HUD is back whole in the first frame after the set goes off. The menus, the title
// screen, the dialogue and the loading screens are layouts of other archives and stay.
static struct {
    int set;          // the nohud set (rt_patch_set), -1 when the generated C lacks it
    int hidden;       // the setting asks for no HUD (and the set is on when the generated C has it)
    unsigned seen;    // cfg_version the setting was read at
} H = {.set = -1, .seen = ~0u};

static void hud_init(void) {
    H.set = rt_patch_set("nohud");
    H.hidden = rt_patch_enabled(H.set);   // AC_NOHUD_PATCH, until the first present reads the setting
}

// The game's present (vk_backend.c vk_be_present, the game's thread): the set follows the setting when it changed.
void res_hud_tick(void) {
    unsigned v = cfg_version();
    if (v == H.seen) return;
    H.seen = v;
    int hide = !cfg_bool("AC_HUD");
    if (hide == H.hidden) return;
    H.hidden = hide;
    if (H.set < 0) {
        if (hide) rt_log("resolution: hud = 0 needs the nohud patch set, which the generated C lacks: the HUD stays");
        return;
    }
    rt_patch_switch(H.set, hide);
    rt_log("resolution: the gameplay HUD is %s (hud = %d)", hide ? "hidden" : "shown", !hide);
}

// 1 for an instance of a layout of the gameplay HUD's archive, by the archive's name in the game's table (a few
// entries, read each time: nothing here relies on an archive keeping its number).
static int hud_instance(const uint8_t *inst) {
    uint32_t arc;
    memcpy(&arc, inst + INST_ARCHIVE, sizeof arc);
    return !strcmp(anchor_archive(arc), "hud2");
}

// main+0x9177ec (ldrb w8,[x23,#0xd1]), in the draw of an instance that has a layer of its own: x23 = the instance,
// w8 = its byte the function tests next, 0 for one it does not draw. A HUD instance reads as 0.
void rt_hud_off23_x8(Cpu *c) {
    if (hud_instance((const uint8_t *)(uintptr_t)c->x[23])) c->x[8] = 0;
}

// main+0x915ea0 (ldr x8,[x22,#0x160]), in the draw of the queued instances: x22 = the instance taken off the list,
// x8 = its parts' count, 0 for one the function passes over. A HUD instance reads as 0.
void rt_hud_off22_x8(Cpu *c) {
    if (hud_instance((const uint8_t *)(uintptr_t)c->x[22])) c->x[8] = 0;
}
