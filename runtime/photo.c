// Photo mode capture: each photo the game takes is saved as a PNG in screenshots/ next to the executable, from the
// game's own output, and the game's album gets the same picture in place of the one it made.
//
// How the game takes a photo (1.0.1, found with frame dumps and guest backtraces of its ReadTexels): ZR in photo mode
// runs main+0x57b060, which records a chain of passes after the frame's scaling pass (SMAA's blending, where an
// upscaler runs) and before the HUD. The chain samples the output (the game-resolution RGBA8 target) into targets of
// the scene's size, filters them, and draws the result into a 1280x720 RGBA8 texture. A few frames later main+0x57b150
// reads that texture back (nvnTextureReadTexels), hands it to nn::image::JpegEncoder::SetPixelData and stores the JPEG
// in GameData.dat (12 slots of 0x30d70 bytes in the save we looked at). The chain copies texel for texel, which holds
// only while the scene is as large as the output: the res set gives every dynamic-resolution level the output's size,
// but the scene set of the upscaling modes makes the scene smaller, so the copy kept the output's top-left corner (at
// 1120x630 for a 1920x1080 output: 58 % of each side), and the album's last pass then read the light target before
// tone mapping, a white picture. A picture of the wide set (aspect = wide) is a wider camera squeezed into the 16:9
// targets, which the album stored squeezed.
//
// What this does instead: the set photo of recomp/patches-v101.txt calls rt_photo_arm inside main+0x57b060 once it
// knows the photo goes ahead, and rt_photo_end once the function has recorded the photo's passes. The next
// SetRenderTargets that thread records after the first (the chain's first pass) is preceded by a capture record
// (nvn_api_cmd.c, OP_PHOTO) naming the last of the game's window textures set as a target before it, which is the
// frame's output after the scaling pass, and the shape of the frame (resolution.c res_wide_shape); the next one after
// the second by OP_PHOTO_FILL, naming the album texture the photo names (rt_photo_arm). The replay (vk_photo.c) copies
// the output into a host buffer in submission order, so the picture has no HUD and comes after any upscaler at
// whatever scene size, and blits the middle of the picture as shown into an album image of its own; at OP_PHOTO_FILL
// that image goes into the album texture, which "save this picture?" shows and main+0x57b150 reads back. Once the GPU
// is done with the copy, a thread of its own (below) turns it into:
//   - the album picture: the middle of the shown picture at the album's 16:9 shape (all of it for a 16:9 frame), area
//     averaged in linear light to 1280x720 (finer than the GPU's linear blit). JpegEncoder::SetPixelData (observed
//     here) gets it in place of the pixels the game read back, for the photo it is due for; the game's album stays
//     16:9, as its layouts and slots are.
//   - the screenshot: the output at its own size, stretched to the frame's shape on one axis (Catmull-Rom) the way
//     the present shows it, as screenshots/AstralChain_<date>_<time>.png (png.c).
// The game's thread waits for the album picture only when SetPixelData comes before it is made (at most WAIT_MS, with
// its core given up); without it (no Vulkan back end, a format the copy does not take, the wait running out) the JPEG
// keeps what the game read back. Checked on the Arc 140T with one recorded photo of Harmony Square replayed: at a
// 1920x1080 output with FSR 3 balanced (a 1120x630 scene) "save this picture?" showed, and the album held, the
// top-left 58 % in white before, and the whole picture now; XeSS performance at 2560x1440, no upscaler with SMAA off,
// and aspect = wide in 2560x1080 and 1920x1200 windows give the whole picture too (the 16:9 middle for the wide ones),
// and PNGs of 1920x1080, 2560x1440, 2560x1080 and 1920x1200.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpu.h"
#include "host.h"
#include "photo.h"
#include "png.h"
#include "platform/platform.h"

#define ALBUM_W PHOTO_ALBUM_W
#define ALBUM_H PHOTO_ALBUM_H
#define WAIT_MS 2000   // SetPixelData waits at most this long for the album picture

typedef struct {
    uint32_t seq, w, h;   // the photo, the output's size
    float shape;          // the frame's shape (width / height as shown), 0: the output's own
    uint8_t *rgb;         // w x h x 3, sRGB as the output holds it
    uint8_t *album;       // ALBUM_W x ALBUM_H x 4 (RGBA), NULL when it could not be made
    int refs;             // under lock: the worker, and cur while it names the photo
    const uint8_t *px;    // the back end's copy, until the worker converted it
    size_t stride;
    int bgra;
    void (*release)(void *);
    void *ctx;
} Shot;

static PlatMutex lock = PLAT_MUTEX_INIT;
static PlatCond cv;
static int cv_ready;
static uint32_t next_seq, want_seq;   // the last photo recorded
static uint32_t used_seq;             // the last photo SetPixelData took (or gave up on)
static uint32_t done_seq;             // the last photo whose album picture is made, or that failed
static Shot *cur;                     // the photo done_seq names, NULL when it failed

static void cv_init(void) {   // under lock
    if (!cv_ready) plat_cond_init(&cv), cv_ready = 1;
}

static void unref(Shot *s) {
    plat_mutex_lock(&lock);
    int last = --s->refs == 0;
    plat_mutex_unlock(&lock);
    if (!last) return;
    free(s->rgb);
    free(s->album);
    free(s);
}

// Photo seq is done (s its pictures, or NULL when it failed): SetPixelData may go on.
static void finish(uint32_t seq, Shot *s) {
    Shot *old = NULL;
    plat_mutex_lock(&lock);
    cv_init();
    if (seq >= done_seq) {
        done_seq = seq;
        old = cur;
        cur = s;
        if (s) s->refs++;
    }
    plat_cond_broadcast(&cv);
    plat_mutex_unlock(&lock);
    if (old) unref(old);
}

// ---- the game's side

// main+0x57b098, after its mov x0, x19: in the function that records the photo's passes, past its checks that a
// photo is due. x20 is the photo: +8 holds the album texture's agl object, whose NVNtexture is at +0x48 (the texture
// main+0x57b150 reads back; found by comparing that ReadTexels's texture with the object's words).
void rt_photo_arm(Cpu *c) {
    const void *album = NULL;
    uint64_t obj = c->x[20] ? *(const uint64_t *)(uintptr_t)(c->x[20] + 8) : 0;
    if (obj && !(obj & 7)) album = (const void *)(uintptr_t)(obj + 0x48);   // nvn_photo_stage checks the texture
    nvn_photo_stage(c->thread, 1, album);
}

// main+0x57b110, after its adrp: the same function has recorded the photo's passes.
void rt_photo_end(Cpu *c) { nvn_photo_stage(c->thread, 3, NULL); }

uint32_t photo_take(float *shape) {
    plat_mutex_lock(&lock);
    uint32_t seq = want_seq = ++next_seq;
    plat_mutex_unlock(&lock);
    *shape = res_wide_shape();
    return seq;
}

void photo_capture_failed(uint32_t seq, const char *why) {
    rt_log("photo: photo %u not captured (%s): the album keeps the game's picture and no screenshot is saved", seq, why);
    finish(seq, NULL);
}

// ---- resampling

static float lin_of[256];      // sRGB byte -> linear
static uint8_t srgb_of[4097];  // linear * 4096, rounded -> sRGB byte (a step of 1/4096 is under one sRGB level)

static void luts(void) {
    if (srgb_of[4096]) return;   // the same tables whichever thread builds them first
    for (int i = 0; i < 256; i++) {
        double v = i / 255.0;
        lin_of[i] = (float)(v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4));
    }
    for (int i = 0; i <= 4096; i++) {
        double l = i / 4096.0, v = l <= 0.0031308 ? l * 12.92 : 1.055 * pow(l, 1 / 2.4) - 0.055;
        srgb_of[i] = (uint8_t)lrint(v * 255.0);
    }
}

static uint8_t enc(float l) {
    int i = (int)(l * 4096.0f + 0.5f);
    return srgb_of[i < 0 ? 0 : i > 4096 ? 4096 : i];
}

// Area weights of n outputs over the source range [a0, a1) of an axis of len pixels: output i covers
// a0 + [i, i + 1) * (a1 - a0) / n, and each source pixel weighs what it covers of that. first[i], cnt[i] and the
// weights at w + i * taps.
typedef struct { int32_t *first, *cnt; float *w; int taps; } Area;

static int area_make(Area *ar, double a0, double a1, uint32_t n, uint32_t len) {
    double sc = (a1 - a0) / n;
    ar->taps = (int)ceil(sc) + 2;
    ar->first = malloc(sizeof(int32_t) * n), ar->cnt = malloc(sizeof(int32_t) * n);
    ar->w = malloc(sizeof(float) * n * (size_t)ar->taps);
    if (!ar->first || !ar->cnt || !ar->w) return 0;
    for (uint32_t i = 0; i < n; i++) {
        double b0 = a0 + i * sc, b1 = b0 + sc;
        int32_t f = (int32_t)floor(b0), l = (int32_t)ceil(b1) - 1;
        if (f < 0) f = 0;
        if (l > (int32_t)len - 1) l = (int32_t)len - 1;
        if (l < f) l = f;
        if (l - f + 1 > ar->taps) l = f + ar->taps - 1;
        float *w = ar->w + (size_t)i * ar->taps, sum = 0;
        for (int32_t p = f; p <= l; p++) {
            double o = fmin(b1, p + 1.0) - fmax(b0, (double)p);
            w[p - f] = o > 0 ? (float)o : 0;
            sum += w[p - f];
        }
        for (int32_t p = f; p <= l; p++) w[p - f] = sum > 0 ? w[p - f] / sum : 1.0f / (float)(l - f + 1);
        ar->first[i] = f, ar->cnt[i] = l - f + 1;
    }
    return 1;
}

static void area_free(Area *ar) { free(ar->first), free(ar->cnt), free(ar->w); }

// The source rectangle [x0, x1) x [y0, y1) of rgb (sw x sh x 3), area averaged in linear light into dst: dw x dh
// RGBA pixels, rows dstride bytes apart, alpha 255. 0 when memory ran out.
static int area_rgba(const uint8_t *rgb, uint32_t sw, uint32_t sh, double x0, double y0, double x1, double y1,
                     uint8_t *dst, uint32_t dw, uint32_t dh, size_t dstride) {
    luts();
    Area ax = {0}, ay = {0};
    int ok = area_make(&ax, x0, x1, dw, sw) && area_make(&ay, y0, y1, dh, sh);
    int32_t r0 = ok ? ay.first[0] : 0, r1 = ok ? ay.first[dh - 1] + ay.cnt[dh - 1] : 0;
    float *tmp = ok ? malloc(sizeof(float) * 3 * dw * (size_t)(r1 - r0)) : NULL;
    if (tmp) {
        for (int32_t r = r0; r < r1; r++) {   // across: every source row the rectangle touches
            const uint8_t *s = rgb + (size_t)r * sw * 3;
            float *t = tmp + (size_t)(r - r0) * dw * 3;
            for (uint32_t i = 0; i < dw; i++) {
                const float *w = ax.w + (size_t)i * ax.taps;
                const uint8_t *p = s + (size_t)ax.first[i] * 3;
                float a = 0, b = 0, c = 0;
                for (int k = 0; k < ax.cnt[i]; k++, p += 3) a += w[k] * lin_of[p[0]], b += w[k] * lin_of[p[1]], c += w[k] * lin_of[p[2]];
                t[3 * i] = a, t[3 * i + 1] = b, t[3 * i + 2] = c;
            }
        }
        for (uint32_t j = 0; j < dh; j++) {   // down
            const float *w = ay.w + (size_t)j * ay.taps;
            uint8_t *o = dst + (size_t)j * dstride;
            for (uint32_t i = 0; i < dw; i++) {
                float a = 0, b = 0, c = 0;
                const float *t = tmp + ((size_t)(ay.first[j] - r0) * dw + i) * 3;
                for (int k = 0; k < ay.cnt[j]; k++, t += (size_t)dw * 3) a += w[k] * t[0], b += w[k] * t[1], c += w[k] * t[2];
                o[4 * i] = enc(a), o[4 * i + 1] = enc(b), o[4 * i + 2] = enc(c), o[4 * i + 3] = 255;
            }
        }
    }
    free(tmp);
    area_free(&ax), area_free(&ay);
    return tmp != NULL;
}

// The shown picture is the output stretched to the frame's shape on one axis.
void photo_middle(uint32_t w, uint32_t h, float shape, double want, double r[4]) {
    double shown = shape > 0 ? shape : (double)w / h;
    double fx = shown > want ? want / shown : 1, fy = shown < want ? shown / want : 1;
    r[0] = w * (1 - fx) / 2, r[2] = w * (1 + fx) / 2;
    r[1] = h * (1 - fy) / 2, r[3] = h * (1 + fy) / 2;
}

// Catmull-Rom along one axis, in the sRGB bytes: n_in samples to n_out (n_out > n_in, a stretch).
static void cr_axis(const uint8_t *src, uint8_t *dst, uint32_t n_in, uint32_t n_out, uint32_t lines, size_t s_step,
                    size_t s_line, size_t d_step, size_t d_line) {
    double sc = (double)n_in / n_out;
    for (uint32_t o = 0; o < n_out; o++) {
        double x = (o + 0.5) * sc - 0.5;
        int32_t i = (int32_t)floor(x);
        float t = (float)(x - i), t2 = t * t, t3 = t2 * t;
        float w[4] = {0.5f * (-t3 + 2 * t2 - t), 0.5f * (3 * t3 - 5 * t2 + 2), 0.5f * (-3 * t3 + 4 * t2 + t),
                      0.5f * (t3 - t2)};
        size_t at[4];
        for (int k = 0; k < 4; k++) {
            int32_t p = i - 1 + k;
            at[k] = (size_t)(p < 0 ? 0 : p > (int32_t)n_in - 1 ? (int32_t)n_in - 1 : p) * s_step;
        }
        for (uint32_t l = 0; l < lines; l++) {
            const uint8_t *s = src + l * s_line;
            uint8_t *d = dst + l * d_line + o * d_step;
            for (int c = 0; c < 3; c++) {
                float v = w[0] * s[at[0] + c] + w[1] * s[at[1] + c] + w[2] * s[at[2] + c] + w[3] * s[at[3] + c];
                d[c] = (uint8_t)(v <= 0 ? 0 : v >= 255 ? 255 : v + 0.5f);
            }
        }
    }
}

// ---- the worker

static void screenshot(const Shot *s) {
    char dir[PLAT_PATH_MAX], path[PLAT_PATH_MAX + 64];
    if (!plat_exe_path(dir, sizeof dir)) {
        rt_log("photo: photo %u: the executable's folder is unknown, no screenshot saved", s->seq);
        return;
    }
    char *cut = strrchr(dir, '/'), *cut2 = strrchr(dir, '\\');
    if (cut2 > cut) cut = cut2;
    char sep = cut ? *cut : '/';   // the executable path's own separator
    if (cut) *cut = 0;
    if (strlen(dir) + sizeof "/screenshots" > sizeof dir) return;
    snprintf(dir + strlen(dir), sizeof dir - strlen(dir), "%cscreenshots", sep);
    plat_file_mkdir_p(dir);
    struct tm t;
    plat_local_time(&t);
    int n = snprintf(path, sizeof path, "%s%cAstralChain_%04d-%02d-%02d_%02d-%02d-%02d", dir, sep, t.tm_year + 1900,
                     t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    snprintf(path + n, sizeof path - (size_t)n, ".png");
    for (int k = 2; plat_file_exists(path) && k < 100; k++) snprintf(path + n, sizeof path - (size_t)n, "_%d.png", k);
    // The shown picture: the output stretched to the frame's shape along the axis it differs on.
    uint32_t w = s->w, h = s->h;
    double own = (double)w / h;
    if (s->shape > own * 1.001) w = (uint32_t)lrint(h * s->shape);
    else if (s->shape > 0 && s->shape < own / 1.001) h = (uint32_t)lrint(w / s->shape);
    const uint8_t *px = s->rgb;
    uint8_t *wide = NULL;
    if (w != s->w || h != s->h) {
        wide = malloc((size_t)w * h * 3);
        if (!wide) {
            rt_log("photo: photo %u: out of memory for the %ux%u screenshot", s->seq, w, h);
            return;
        }
        if (w != s->w) cr_axis(s->rgb, wide, s->w, w, h, 3, (size_t)s->w * 3, 3, (size_t)w * 3);
        else cr_axis(s->rgb, wide, s->h, h, w, (size_t)s->w * 3, 3, (size_t)w * 3, 3);
        px = wide;
    }
    PngBufs pb = {0};
    uint64_t t0 = plat_now_ns();
    int ok = png_write_rgb(path, px, w, h, 3, (size_t)w * 3, &pb);
    png_bufs_free(&pb);
    free(wide);
    if (ok)
        rt_log("photo: photo %u saved as %s (%ux%u%s, written in %.0f ms)", s->seq, path, w, h,
               w != s->w || h != s->h ? ", the output stretched to the frame's shape" : "", (plat_now_ns() - t0) / 1e6);
    else
        rt_log("photo: photo %u: cannot write %s", s->seq, path);
}

static void worker(void *arg) {
    Shot *s = arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    s->rgb = malloc((size_t)s->w * s->h * 3);
    if (s->rgb)
        for (uint32_t y = 0; y < s->h; y++) {
            const uint8_t *p = s->px + y * s->stride;
            uint8_t *o = s->rgb + (size_t)y * s->w * 3;
            int r = s->bgra ? 2 : 0, b = s->bgra ? 0 : 2;
            for (uint32_t x = 0; x < s->w; x++, p += 4, o += 3) o[0] = p[r], o[1] = p[1], o[2] = p[b];
        }
    s->release(s->ctx);
    s->px = NULL;
    if (!s->rgb) {
        rt_log("photo: photo %u: out of memory for the %ux%u picture", s->seq, s->w, s->h);
        finish(s->seq, NULL);
        unref(s);
        return;
    }
    s->album = malloc((size_t)ALBUM_W * ALBUM_H * 4);
    double r[4];
    photo_middle(s->w, s->h, s->shape, (double)ALBUM_W / ALBUM_H, r);
    if (s->album && !area_rgba(s->rgb, s->w, s->h, r[0], r[1], r[2], r[3], s->album, ALBUM_W, ALBUM_H, ALBUM_W * 4)) {
        free(s->album);
        s->album = NULL;
    }
    finish(s->seq, s);
    screenshot(s);
    unref(s);
}

void photo_captured(uint32_t seq, const uint8_t *px, uint32_t w, uint32_t h, size_t stride, int bgra, float shape,
                    void (*release)(void *), void *ctx) {
    Shot *s = calloc(1, sizeof *s);
    if (s) {
        *s = (Shot){.seq = seq, .w = w, .h = h, .shape = shape, .refs = 1, .px = px, .stride = stride, .bgra = bgra,
                    .release = release, .ctx = ctx};
        if (plat_thread_spawn(worker, s, "photo", 0)) return;
        free(s);
    }
    release(ctx);
    photo_capture_failed(seq, "no thread for it");
}

// ---- the album picture

// void nn::image::JpegEncoder::SetPixelData(const void *pixels, PixelFormat format, const Dimension &size,
// int lineAlignment): main+0x57b150 passes the photo it read back (format 0, RGBA, 1280x720). The photo due gets the
// capture's album picture in place of those pixels.
static void obs_jpeg_pixels(Cpu *c) {
    uint8_t *px = (uint8_t *)(uintptr_t)c->x[1];
    const int32_t *dim = (const int32_t *)(uintptr_t)c->x[3];
    int fmt = (int)c->x[2], align = (int)c->x[4];
    plat_mutex_lock(&lock);
    cv_init();
    uint32_t want = want_seq;
    int due = want && want != used_seq;
    if (due) used_seq = want;
    if (due && done_seq < want) {   // the GPU or the worker is not done yet: wait without the core
        plat_mutex_unlock(&lock);
        int away = kern_core_away();
        uint64_t end = plat_now_ns() + WAIT_MS * 1000000ull;
        plat_mutex_lock(&lock);
        for (uint64_t now; done_seq < want && (now = plat_now_ns()) < end;) plat_cond_wait_ns(&cv, &lock, end - now);
        plat_mutex_unlock(&lock);
        kern_core_back(&away);
        plat_mutex_lock(&lock);
        if (done_seq < want) rt_log("photo: photo %u still not captured after %d ms: the album keeps the game's picture", want, WAIT_MS);
    }
    Shot *s = due && cur && cur->seq == want ? cur : NULL;
    if (s) s->refs++;
    plat_mutex_unlock(&lock);
    if (s) {
        int w = dim ? dim[0] : 0, h = dim ? dim[1] : 0;
        size_t stride = (size_t)(align > 0 ? (w + align - 1) / align * align : w) * 4;
        if (fmt != 0 || w <= 0 || h <= 0 || !px) {
            rt_log("photo: photo %u: the album asks for format %d at %dx%d, which this does not make: its own picture",
                   s->seq, fmt, w, h);
        } else if (w == ALBUM_W && h == ALBUM_H && s->album) {
            for (int y = 0; y < h; y++) memcpy(px + (size_t)y * stride, s->album + (size_t)y * ALBUM_W * 4, ALBUM_W * 4);
            double own = (double)s->w / s->h;
            rt_log("photo: photo %u: the album gets %s the %ux%u output", s->seq,
                   s->shape > own * 1.001 ? "the 16:9 middle of the wider picture made from" :
                   s->shape > 0 && s->shape < own / 1.001 ? "the 16:9 middle of the taller picture made from" : "the picture of",
                   s->w, s->h);
        } else {
            double r[4];
            photo_middle(s->w, s->h, s->shape, (double)w / h, r);
            if (area_rgba(s->rgb, s->w, s->h, r[0], r[1], r[2], r[3], px, (uint32_t)w, (uint32_t)h, stride))
                rt_log("photo: photo %u: the album gets the %ux%u output at %dx%d", s->seq, s->w, s->h, w, h);
        }
        unref(s);
    }
    rt_original(c);
}
HOST_OBSERVER("_ZN2nn5image11JpegEncoder12SetPixelDataEPKvNS0_11PixelFormatERKNS0_9DimensionEi", obs_jpeg_pixels);
