// Adapted from MacNativeGames games/totk/runtime/loader.c (lines 456-477) and services.c (lines 219-235) @ b75be8cf
// (MIT, Copyright (c) 2026 MacNativeGames contributors) and LAS Recomp runtime/main.c (lines 16-35, 141-159) @
// da35b83 (MIT, Copyright (c) 2026 Peiikai).
// The boot driver: the guest main thread runs the game's own nn::init::Start
// with three host callbacks, the second of which runs DT_INIT of every loaded module (main and sdk for
// Astral Chain); Start then calls nnMain.
// --boot ends at the first nvnBootstrapLoader call; the return address is logged, not checked against a table.
// With --seconds S the run continues past that call; rt_frame_presented ends it once the world clock passed S.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "cheats.h"
#include "fs.h"
#include "gfx.h"
#include "mods.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "platform/platform.h"

#define SYM_INIT_START "_ZN2nn4init5StartEmmPFvvES2_"
#define CTOR_DONE 2                      // a module's constructor flag after its DT_INIT ran the init_array

static volatile int32_t inits_done;
uint64_t rt_guest_start_ns;

static uint32_t ctor_state(const RtLoaded *m) { return LD32(m->ctor_flag); }

static const char *caller(const Cpu *c, char *buf, int n) { return ldr_describe(c->x[30] - 4, buf, n); }

// main.npdm META: the main thread's priority (+0xE), core (+0xF) and stack size (+0x1C). kernel.c builds the main
// thread with the 1.2.1 values; a different npdm means a different game build.
static void check_npdm(void) {
    size_t n;
    uint8_t *npdm = ldr_read_verified(rt_config()->exefs, "main.npdm", &n);
    if (n < 0x80 || memcmp(npdm, "META", 4)) rt_fatal("boot: main.npdm has no META header");
    uint32_t stack;
    memcpy(&stack, npdm + 0x1C, 4);
    int prio = npdm[0xE], core = npdm[0xF];
    free(npdm);
    if (prio != 44 || core != 0 || stack != 0x100000)
        rt_fatal("boot: main.npdm asks for priority %d, core %d, stack %#x; kernel.c builds 44, 0, 0x100000", prio,
                 core, stack);
}

// ---- the three callbacks of nn::init::Start (rtld's job on hardware)
static void cb_loader_ready(Cpu *c) {
    (void)c;
    rt_log("boot: rtld callback: modules linked");
}

// DT_INIT registers the module with musl (__nnmusl_init_dso) and runs its init_array; the constructor flag reads 2
// afterwards (MacNativeGames loader.c:458-462). The load order (main, sdk) is the layout order.
static void cb_module_inits(Cpu *c) {
    if (plat_cas32(&inits_done, 0, 0)) rt_fatal("boot: module constructors requested twice");
    for (int i = 0; i < ldr_count(); i++) {
        const RtLoaded *m = ldr_module(i);
        uint64_t entries = m->init_array_size / 8;
        if (!m->init) rt_fatal("boot: %s has no DT_INIT", m->name);
        rt_log("boot: ENTER %s DT_INIT (%llu init_array entries)", m->name, (unsigned long long)entries);
        rt_call(c, m->init, 0);
        uint32_t state = ctor_state(m);
        rt_logf("boot: DT_INIT %s state=%u (%llu)\n", m->name, state, (unsigned long long)entries);
        if (state != CTOR_DONE) rt_fatal("boot: %s DT_INIT left its constructor flag at %u, not 2", m->name, state);
    }
    plat_xchg32(&inits_done, 1);
}

static void cb_module_finis(Cpu *c) {
    char lr[128];
    rt_fatal("boot: the game asked for module finalizers (from %s): the guest exited", caller(c, lr, sizeof lr));
}

// ---- checkpoints (host.h)
static void stats_line(void) {
    RtStats s = rt_stats();
    char calls[48] = "";   // a game run counts host calls only for AC_TRACE=host (main.c) or AC_CORE_DIAG
    if (s.host_calls) snprintf(calls, sizeof calls, ", %llu host calls", (unsigned long long)s.host_calls);
    rt_logf("boot: passed (%llu SVCs, %llu threads%s)\n", (unsigned long long)s.svcs, (unsigned long long)s.threads,
            calls);
}

void rt_boot_event(const char *what, Cpu *c, const char *detail) {
    char at[128];
    caller(c, at, sizeof at);
    if (!strcmp(what, "nvn")) {
        rt_logf("boot: first NVN call nvnBootstrapLoader(\"%s\") from %s\n", detail, at);
        if (rt_config()->seconds) return;   // --seconds: the run goes on into the game
        if (rt_trace & RT_TRACE_HOSTCALL) hostfn_report();
        stats_line();
        fflush(stdout);
        fflush(stderr);
        plat_exit(0);   // other guest threads are still running; nothing of theirs needs an orderly end
    }
    rt_fatal("boot: unknown boot event %s (%s) from %s", what, detail ? detail : "", at);
}

// ---- --seconds S
// A PNG with stored (uncompressed) deflate blocks: no compression library, and the shot is only read by tools.
// rt_write_png (declared in vk.h) is shared with the AC_DUMP frame dump (vk_dump.c).
static uint32_t crc32_update(uint32_t crc, const uint8_t *p, size_t n) {
    static uint32_t table[256];
    if (!table[1])
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    for (size_t i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t hdr[8] = {(uint8_t)(n >> 24), (uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n};
    memcpy(hdr + 4, type, 4);
    uint32_t crc = crc32_update(crc32_update(~0u, hdr + 4, 4), data, n) ^ ~0u;
    uint8_t tail[4] = {(uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc};
    fwrite(hdr, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    fwrite(tail, 1, 4, f);
}

void rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    uint64_t row = 1 + 4ull * w, raw_n = row * h;
    uint8_t *raw = malloc(raw_n);
    if (!raw) rt_fatal("boot: out of memory for the shot");
    for (uint32_t y = 0; y < h; y++) {
        raw[y * row] = 0;   // filter: none
        memcpy(raw + y * row + 1, rgba + 4ull * w * y, 4ull * w);
    }
    uint64_t blocks = (raw_n + 65534) / 65535;
    uint8_t *z = malloc(2 + raw_n + 5 * blocks + 4);
    if (!z) rt_fatal("boot: out of memory for the shot");
    uint64_t zn = 0;
    z[zn++] = 0x78, z[zn++] = 0x01;
    uint32_t a = 1, b = 0;
    for (uint64_t off = 0; off < raw_n; off += 65535) {
        uint32_t k = raw_n - off > 65535 ? 65535 : (uint32_t)(raw_n - off);
        z[zn++] = off + k >= raw_n;
        z[zn++] = (uint8_t)k, z[zn++] = (uint8_t)(k >> 8), z[zn++] = (uint8_t)~k, z[zn++] = (uint8_t)(~k >> 8);
        memcpy(z + zn, raw + off, k);
        zn += k;
        for (uint32_t i = 0; i < k; i++) a = (a + raw[off + i]) % 65521, b = (b + a) % 65521;
    }
    uint32_t ad = b << 16 | a;
    z[zn++] = (uint8_t)(ad >> 24), z[zn++] = (uint8_t)(ad >> 16), z[zn++] = (uint8_t)(ad >> 8), z[zn++] = (uint8_t)ad;
    FILE *f = fopen(path, "wb");
    if (!f) rt_fatal("boot: cannot write %s", path);
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                        (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 6, 0, 0, 0};
    fwrite(sig, 1, 8, f);
    png_chunk(f, "IHDR", ihdr, 13);
    png_chunk(f, "IDAT", z, (uint32_t)zn);
    png_chunk(f, "IEND", NULL, 0);
    if (fclose(f)) rt_fatal("boot: cannot write %s", path);
    free(z);
    free(raw);
}

static void save_shot(const tex_t *t, const char *path) {
    int fmt = t->d.format;
    int bgra = fmt == NVN_FORMAT_BGRA8 || fmt == NVN_FORMAT_BGRA8_SRGB;
    if (!bgra && fmt != NVN_FORMAT_RGBA8 && fmt != NVN_FORMAT_RGBA8_SRGB)
        rt_fatal("boot: --shot supports 8-bit RGBA and BGRA targets, the presented texture has NVN format %#x", fmt);
    uint32_t w = t->d.width, h = t->d.height;
    uint8_t *px = malloc(4ull * w * h);
    if (!px) rt_fatal("boot: out of memory for the shot");
    view_t view;
    memset(&view, 0, sizeof view);
    NVNcopyRegion r = {.width = (int)w, .height = (int)h, .depth = 1};
    if (gfx_replay_async()) gfx_backend()->finish();   // the presented frame may still be queued for the replay
    gfx_backend()->read_texels(t, &view, &r, px, 4ull * w, 4ull * w * h);
    if (bgra)
        for (uint64_t i = 0; i < 4ull * w * h; i += 4) {
            uint8_t x = px[i];
            px[i] = px[i + 2];
            px[i + 2] = x;
        }
    rt_write_png(path, px, w, h);
    free(px);
    rt_logf("boot: shot %s (%ux%u)\n", path, w, h);
}

// AC_SHOTS=lo-hi:step [AC_SHOTS_DIR=dir]: every step-th present in [lo, hi] saved as <dir>/f<n>.png (n the
// present), box-filtered to 480x270; AC_SHOTS=w<lo>-<hi>:<step> takes them by world-clock seconds instead, the first
// present at or after each step (full size with AC_SHOTS_FULL=1), for finding short glitches. Pipeline builds stay asynchronous, so the
// pictures show what a player sees.
static void series_shot(const tex_t *t, uint64_t n) {
    static int parsed;
    static unsigned long long lo, hi, step, next;
    static const char *dir;
    static int full, world;
    if (!parsed) {
        full = cfg_get("AC_SHOTS_FULL") != NULL;
        const char *e = cfg_get("AC_SHOTS");
        double wl, wh, ws;
        if (e && e[0] == 'w' && sscanf(e + 1, "%lf-%lf:%lf", &wl, &wh, &ws) == 3 && ws > 0)   // world-clock seconds
            world = 1, lo = (unsigned long long)(wl * 1e6), hi = (unsigned long long)(wh * 1e6),
            step = (unsigned long long)(ws * 1e6), next = lo;
        else if (!e || sscanf(e, "%llu-%llu:%llu", &lo, &hi, &step) != 3 || !step) hi = 0;
        dir = cfg_get("AC_SHOTS_DIR") ? cfg_get("AC_SHOTS_DIR") : ".";
        if (hi) plat_file_mkdir_p(dir);
        parsed = 1;
    }
    if (world) {   // the first present at or after each step of the world clock
        int64_t w = frame_step_world_us();
        if (w < 0 || (unsigned long long)w < next || next > hi) return;
        while (next <= (unsigned long long)w) next += step;
    } else if (n < lo || n > hi || (n - lo) % step) {
        return;
    }
    int fmt = t->d.format;
    int bgra = fmt == NVN_FORMAT_BGRA8 || fmt == NVN_FORMAT_BGRA8_SRGB;
    if (!bgra && fmt != NVN_FORMAT_RGBA8 && fmt != NVN_FORMAT_RGBA8_SRGB) return;
    uint32_t w = t->d.width, h = t->d.height, ow = full ? w : 480, oh = full ? h : 270;
    uint8_t *px = malloc(4ull * w * h), *out = malloc(4ull * ow * oh);
    if (!px || !out) rt_fatal("boot: out of memory for a AC_SHOTS frame");
    view_t view;
    memset(&view, 0, sizeof view);
    NVNcopyRegion r = {.width = (int)w, .height = (int)h, .depth = 1};
    if (gfx_replay_async()) gfx_backend()->finish();   // the presented frame may still be queued for the replay
    gfx_backend()->read_texels(t, &view, &r, px, 4ull * w, 4ull * w * h);
    for (uint32_t y = 0; y < oh; y++)
        for (uint32_t x = 0; x < ow; x++) {
            uint32_t x0 = x * w / ow, x1 = (x + 1) * w / ow, y0 = y * h / oh, y1 = (y + 1) * h / oh;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;
            uint32_t sum[4] = {0}, k = 0;
            for (uint32_t sy = y0; sy < y1; sy++)
                for (uint32_t sx = x0; sx < x1; sx++, k++)
                    for (int c = 0; c < 4; c++) sum[c] += px[4ull * (sy * (uint64_t)w + sx) + (uint32_t)c];
            uint8_t *o = out + 4ull * (y * ow + x);
            for (int c = 0; c < 4; c++) o[c] = (uint8_t)(sum[bgra && c != 3 ? 2 - c : c] / k);
            o[3] = 255;
        }
    char path[PLAT_PATH_MAX];
    snprintf(path, sizeof path, "%s/f%05llu.png", dir, (unsigned long long)n);
    rt_write_png(path, out, ow, oh);
    free(out);
    free(px);
}

// A --seconds run ends at the first present after the world clock (game time since the save's first frame,
// frame_step_world_us) passed S seconds, so a run covers the same game time at any frame rate; any other frame is
// counted by the back end only.
// AC_PRESENT_SHOTS=w<lo>-<hi>:<step> [AC_SHOTS_DIR=dir] (test): what the window showed, the swapchain image of the
// first present at or after each step of the world clock (seconds), at full size as <dir>/p<ms>.png: with frame
// generation, its pictures and AC_DLSSG_MVDEBUG's overlay, which AC_SHOTS (the game's window texture) does not have.
static void present_series(void) {
    static int parsed;
    static double lo, hi, step, next;
    static const char *dir;
    if (!parsed) {
        parsed = 1;
        const char *e = cfg_get("AC_PRESENT_SHOTS");
        if (!e || e[0] != 'w' || sscanf(e + 1, "%lf-%lf:%lf", &lo, &hi, &step) != 3 || step <= 0) hi = -1;
        dir = cfg_get("AC_SHOTS_DIR") ? cfg_get("AC_SHOTS_DIR") : ".";
        if (hi >= 0) plat_file_mkdir_p(dir);
        next = lo;
    }
    int64_t w = frame_step_world_us();
    if (hi < 0 || w < 0 || (double)w < next * 1e6 || next > hi) return;
    while (next * 1e6 <= (double)w) next += step;
    char path[PLAT_PATH_MAX];
    snprintf(path, sizeof path, "%s/p%06lld.png", dir, (long long)(w / 1000));
    vk_present_shot(path);
}

void rt_frame_presented(const tex_t *t, uint64_t n) {
    const RtConfig *cfg = rt_config();
    series_shot(t, n);
    present_series();
    int64_t w = cfg->seconds ? frame_step_world_us() : -1;
    static int present_shot;   // AC_PRESENT_SHOT=<file>: the window's picture half a second before the run ends
    if (!present_shot && w >= 0 && (double)w >= cfg->seconds * 1e6 - 5e5) {
        present_shot = 1;
        const char *ps = cfg_get("AC_PRESENT_SHOT");
        if (ps && *ps) vk_present_shot(ps);
    }
    if (w < 0 || (double)w < cfg->seconds * 1e6) return;
    if (cfg->shot) save_shot(t, cfg->shot);
    GfxStats g = gfx_stats();
    rt_logf("boot: frames %llu presented (nvn calls %llu, distinct %llu, unimplemented %llu, validation errors %llu, "
           "shader failures %llu, draws skipped %llu)\n", (unsigned long long)g.frames,
           (unsigned long long)g.nvn_calls, (unsigned long long)g.nvn_distinct, (unsigned long long)g.unimplemented,
           (unsigned long long)g.validation_errors, (unsigned long long)g.shader_failures,
           (unsigned long long)g.draws_skipped);
    if (rt_trace & RT_TRACE_HOSTCALL) hostfn_report();   // the timed run's calls per host function, busiest first
    stats_line();
    fflush(stdout);
    fflush(stderr);
    fs_save_flush();
    vk_pipeline_save();   // no-op without the Vulkan back end
    prof_write();
    fflush(stderr);
    plat_exit(0);
}

// ---- steps 9-14
static uint64_t cb_addr[3];

static void guest_main(Cpu *c) {
    uint32_t handle = kern_main_thread(&c);
    uint64_t start = ldr_symbol(SYM_INIT_START);
    if (!start) rt_fatal("boot: %s is not exported", SYM_INIT_START);
    char d[128];
    rt_log("boot: ENTER nn::init::Start at %s, main thread %#x", ldr_describe(start, d, sizeof d), handle);
    // nn::init::Start(main thread handle, arguments region, loader_ready, module_inits, module_finis)
    rt_call_top(c, start, 5, (uint64_t)handle, (uint64_t)0, cb_addr[0], cb_addr[1], cb_addr[2]);
    rt_fatal("boot: nn::init::Start returned");
}

static void prewarm_main(void *arg) {
    (void)arg;
    vk_pipeline_prewarm();
    const char *prep = cfg_get("AC_TEST_PREPARE_SHADERS");   // Task 2.3 test hook; Task 2.6 builds the preparation
    if (prep && !strcmp(prep, "1")) {
        int cpus = (int)plat_cpu_usable();
        gfx_shader_prepare_recorded(cpus, cpus / 2 > 16 ? 16 : cpus / 2 < 1 ? 1 : cpus / 2);
    }
    return;
}

// The scene size and an output_res = auto were decided from the settings before the device existed (resolution.c
// res_init); with the device up and before the game makes its render targets, what can really upscale here is known
// (vk_dlaa.c vk_dlaa_upscaler_ok) and the sizes follow it (res_settle). Without a GPU (AC_GPU=null) nothing renders.
static void settle_resolution(void) {
    const char *gpu = rt_config()->gpu;
    if ((gpu && !strcmp(gpu, "null")) || !vk_init()) return;
    const VkPhysicalDeviceMemoryProperties *m = &vk_ctx()->mem_props;
    VkDeviceSize big = 0;
    for (uint32_t i = 0; i < m->memoryHeapCount; i++)
        if ((m->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && m->memoryHeaps[i].size > big)
            big = m->memoryHeaps[i].size;
    uint64_t limit = vk_mem_vram_limit();   // AC_VRAM_LIMIT_MIB: the card size the port acts on
    if (limit && big > limit) big = limit;
    res_settle(vk_dlaa_upscaler_ok(), (unsigned)(big >> 20));
    vk_dlaa_warm();   // the upscaler's own set-up for these sizes now, not at the first frame after a load
}

void rt_boot(void) {
    check_npdm();
    if (!rt_config()->shot) vk_pipeline_async_enable();   // a --shot image must not depend on build timing
    // Shaders and pipelines recorded in earlier runs are made ready while the modules load; the guest waits for them.
    PlatThread *warm = plat_thread_new(prewarm_main, NULL, "prewarm");
    libc_init();   // host_libc: its overrides must be in place before the relocations
    ldr_load(rt_config()->exefs);
    rt_patches_init();
    mods_exefs_switch();   // after the patch sets' settings: an enabled mod's ExeFS patches on
    frame_step_init();
    res_init();
    quest_init();
    cheats_init();   // after the patch sets: a cheat's set follows its setting
    cb_addr[0] = hostfn_register("<rtld loader_ready>", cb_loader_ready);
    cb_addr[1] = hostfn_register("<rtld module_inits>", cb_module_inits);
    cb_addr[2] = hostfn_register("<rtld module_finis>", cb_module_finis);
    hostfn_seal();
    uint64_t held = plat_now_ms();
    if (warm) plat_thread_join(warm);
    else vk_pipeline_prewarm();
    settle_resolution();
    held = plat_now_ms() - held;
    rt_log("boot: guest starts %llu ms after process start (%llu ms of it waiting for the prewarm)",
           (unsigned long long)(plat_process_uptime_ns() / 1000000), (unsigned long long)held);
    rt_guest_start_ns = kern_now_ns();
    kern_run_on_guest_main(guest_main);
    rt_fatal("boot: the guest main thread ended");
}
