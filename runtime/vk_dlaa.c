// DLAA: NVIDIA DLSS at the scene's own size (AC_DLAA, astralchain.ini dlaa). NGX is loaded at run time from the driver's
// NGX core: on Windows _nvngx.dll, found through HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore; on Linux
// libnvidia-ngx.so.1, which the NVIDIA driver installs on the library path. NGX loads the DLSS model from the folder of
// the executable: nvngx_dlss.dll on Windows, libnvidia-ngx-dlss.so.<version> on Linux. No NVIDIA SDK code or header is
// compiled in: the few NGX types used are declared here from the SDK's documented ABI, the way the Metroid Dread
// recomp's dlss.c reaches the core (read for behaviour only).
//
// The core's parameter object is a C++ class whose virtual functions are called through its vtable: the Set overloads
// in slots 0-7 and the Get overloads in 8-15. MSVC (the Windows core) lays out each group of overloads in reverse
// declaration order, the Itanium C++ ABI (the Linux core, built with GCC) in declaration order. vtable_ok checks the
// layout with a set-then-get round trip before anything relies on it.
//
// Where DLAA runs in the frame, the jitter and the motion vectors: see the sections below.
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_exec.h"
#include "vk_rec.h"
#include "dlaa_mv_spv.h"
#include "dlaa_objmask_spv.h"
#include "platform/platform.h"

typedef int NgxResult;   // NVSDK_NGX_Result: 1 success, 0xBAD00000 | n failures
#define NGX_OK(r) (((unsigned)(r) & 0xFFF00000u) != 0xBAD00000u)
#define NGX_VERSION_API 0x15
#define NGX_FEATURE_SUPERSAMPLING 1
#define NGX_PERFQ_DLAA 5   // NVSDK_NGX_PerfQuality_Value: 0 performance, 1 balanced, 2 quality, 3 ultra performance
typedef struct NgxParam NgxParam;
typedef struct NgxHandle NgxHandle;
typedef struct { void *slot[17]; } NgxParamVtbl;
struct NgxParam { const NgxParamVtbl *vt; };
typedef struct {   // NVSDK_NGX_ImageViewInfo_VK
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
} NgxImageView;
typedef struct {   // NVSDK_NGX_Resource_VK
    union { NgxImageView img; struct { VkBuffer buffer; unsigned bytes; } buf; } r;
    int type;      // 0 image view, 1 buffer
    _Bool rw;
} NgxResourceVK;

typedef NgxResult (*PFN_Init)(unsigned long long app, const wchar_t *path, VkInstance, VkPhysicalDevice, VkDevice,
                              PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int version, const void *info);
typedef NgxResult (*PFN_ReqExt)(unsigned *ni, const char ***ie, unsigned *nd, const char ***de);
typedef NgxResult (*PFN_Params)(NgxParam **);
typedef NgxResult (*PFN_DestroyParams)(NgxParam *);
typedef NgxResult (*PFN_Create)(VkCommandBuffer, int feature, NgxParam *, NgxHandle **);
typedef NgxResult (*PFN_Evaluate)(VkCommandBuffer, const NgxHandle *, const NgxParam *, void *callback);
typedef NgxResult (*PFN_Release)(NgxHandle *);

static struct {
    void *core;   // the NGX core library (an HMODULE on Windows)
    PFN_Init init;
    PFN_ReqExt req_ext;
    PFN_Params caps, alloc_params;
    PFN_DestroyParams destroy_params;
    PFN_Create create;
    PFN_Evaluate evaluate;
    PFN_Release release;
    int tried, loaded, available;
    int wanted;   // -1 not read yet
} N = {.wanted = -1};

typedef void (*SetI)(NgxParam *, const char *, int);
typedef void (*SetU)(NgxParam *, const char *, unsigned);
typedef void (*SetF)(NgxParam *, const char *, float);
typedef void (*SetP)(NgxParam *, const char *, void *);
typedef NgxResult (*GetI)(const NgxParam *, const char *, int *);
typedef NgxResult (*GetU)(const NgxParam *, const char *, unsigned *);
typedef NgxResult (*GetF)(const NgxParam *, const char *, float *);
typedef NgxResult (*GetP)(const NgxParam *, const char *, void **);
// The SDK declares Set(u64), Set(float), Set(double), Set(u32), Set(int), Set(ID3D11Resource *), Set(ID3D12Resource *),
// Set(void *), then the Get overloads in the same order, then Reset.
#ifdef _WIN32
enum { S_P = 0, S_D12, S_D11, S_I, S_U32, S_D, S_F, S_U64, G_P, G_D12, G_D11, G_I, G_U32, G_D, G_F, G_U64 };
#else
enum { S_U64 = 0, S_F, S_D, S_U32, S_I, S_D11, S_D12, S_P, G_U64, G_F, G_D, G_U32, G_I, G_D11, G_D12, G_P };
#endif
static void p_seti(NgxParam *p, const char *n, int v) { ((SetI)p->vt->slot[S_I])(p, n, v); }
static void p_setu(NgxParam *p, const char *n, unsigned v) { ((SetU)p->vt->slot[S_U32])(p, n, v); }
static void p_setf(NgxParam *p, const char *n, float v) { ((SetF)p->vt->slot[S_F])(p, n, v); }
static void p_setp(NgxParam *p, const char *n, void *v) { ((SetP)p->vt->slot[S_P])(p, n, v); }
static int p_geti(const NgxParam *p, const char *n, int *v) { return NGX_OK(((GetI)p->vt->slot[G_I])(p, n, v)); }
static int p_getu(const NgxParam *p, const char *n, unsigned *v) { return NGX_OK(((GetU)p->vt->slot[G_U32])(p, n, v)); }
static int p_getf(const NgxParam *p, const char *n, float *v) { return NGX_OK(((GetF)p->vt->slot[G_F])(p, n, v)); }
static int p_getp(const NgxParam *p, const char *n, void **v) { return NGX_OK(((GetP)p->vt->slot[G_P])(p, n, v)); }

#ifdef _WIN32
#define DLSS_MODEL "nvngx_dlss.dll"
#else
#define DLSS_MODEL "libnvidia-ngx-dlss.so.<version>"
#endif

// NGX is brought up only when dlaa is on at start: its device extensions can only be asked for at device creation, and
// loading the driver's NGX core on every NVIDIA machine would cost start time and change the device with DLAA off. So
// dlaa applies after a restart; switched off, nothing else of DLAA runs.
int vk_dlaa_wanted(void) {
    static int wanted = -1;
    if (wanted < 0) wanted = cfg_bool("AC_DLAA");
    return wanted;
}

static int core_load(void) {
    if (N.tried) return N.loaded;
    N.tried = 1;
#ifdef _WIN32
    char dir[MAX_PATH] = {0}, path[MAX_PATH + 32];
    DWORD n = sizeof dir;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", "FullPath", RRF_RT_REG_SZ,
                     NULL, dir, &n) != ERROR_SUCCESS) {
        rt_log("dlaa: the driver has no NGX core (registry NGXCore missing): DLAA off");
        return 0;
    }
    snprintf(path, sizeof path, "%s\\_nvngx.dll", dir);
    N.core = LoadLibraryA(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (error %lu): DLAA off", path, GetLastError());
        return 0;
    }
#define FN(f, name) N.f = (void *)GetProcAddress((HMODULE)N.core, name)
#else
    // The driver's NGX core; the dynamic loader finds it where the driver put it (/usr/lib, /usr/lib/x86_64-linux-gnu).
    const char *path = "libnvidia-ngx.so.1";
    N.core = plat_lib_open(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (not an NVIDIA driver, or one without NGX): DLAA off", path);
        return 0;
    }
#define FN(f, name) N.f = plat_lib_sym(N.core, name)
#endif
    FN(init, "NVSDK_NGX_VULKAN_Init_Ext2");
    FN(req_ext, "NVSDK_NGX_VULKAN_RequiredExtensions");
    FN(caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    FN(alloc_params, "NVSDK_NGX_VULKAN_AllocateParameters");
    FN(destroy_params, "NVSDK_NGX_VULKAN_DestroyParameters");
    FN(create, "NVSDK_NGX_VULKAN_CreateFeature");
    FN(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
    FN(release, "NVSDK_NGX_VULKAN_ReleaseFeature");
#undef FN
    if (!N.init || !N.req_ext || !N.caps || !N.alloc_params || !N.create || !N.evaluate || !N.release) {
        rt_log("dlaa: the NGX core %s lacks Vulkan entry points: DLAA off", path);
        return 0;
    }
    N.loaded = 1;
    rt_log("dlaa: NGX core %s", path);
    return 1;
}

static int listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return 1;
    return 0;
}

void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < ni; i++)
        if (!listed(ext, *n, ie[i]) && *n < max) ext[(*n)++] = ie[i];
}

void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n,
                         uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < nd; i++) {
        int ok = 0;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) rt_log("dlaa: the device lacks %s, which NGX asks for", de[i]);
        else if (!listed(ext, *n, de[i]) && *n < max) ext[(*n)++] = de[i];
    }
}

static int vtable_ok(void) {
    NgxParam *p = NULL;
    if (!NGX_OK(N.alloc_params(&p)) || !p) return 0;
    int i = 0;
    unsigned u = 0;
    float f = 0;
    void *v = NULL;
    p_seti(p, "Astral Chain.Test.I", -12345);
    p_setu(p, "Astral Chain.Test.U", 4000000000u);
    p_setf(p, "Astral Chain.Test.F", 1.25f);
    p_setp(p, "Astral Chain.Test.P", (void *)p);
    int ok = p_geti(p, "Astral Chain.Test.I", &i) && i == -12345 && p_getu(p, "Astral Chain.Test.U", &u) && u == 4000000000u &&
             p_getf(p, "Astral Chain.Test.F", &f) && f == 1.25f && p_getp(p, "Astral Chain.Test.P", &v) && v == (void *)p;
    if (N.destroy_params) N.destroy_params(p);
    if (!ok) rt_log("dlaa: NGX parameter layout check failed (i=%d u=%u f=%g p=%p): DLAA off", i, u, f, v);
    return ok;
}

#ifndef _WIN32
// UTF-8 to UTF-32 wchar_t, without the C library's locale (the runtime never sets one, so mbstowcs would stop at the
// first non-ASCII byte). Returns 0 on malformed input or when out has no room.
static int utf8_to_wide(const char *in, wchar_t *out, size_t cap) {
    const unsigned char *u = (const unsigned char *)in;
    size_t n = 0;
    while (*u) {
        unsigned c = *u++, more = c < 0x80 ? 0 : c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 4;
        if (more == 4) return 0;
        if (more) c &= 0x3F >> more;
        for (unsigned k = 0; k < more; k++, u++) {
            if ((*u & 0xC0) != 0x80) return 0;
            c = c << 6 | (*u & 0x3F);
        }
        if (n + 1 >= cap) return 0;
        out[n++] = (wchar_t)c;
    }
    out[n] = 0;
    return 1;
}
#endif

void vk_dlaa_init(void) {
    if (!vk_dlaa_wanted() || !core_load()) return;
#ifdef _WIN32
    wchar_t wdir[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, wdir, MAX_PATH);
    while (len && wdir[len - 1] != L'\\') len--;
    wdir[len] = 0;
#else
    // The executable's folder, where NGX looks for libnvidia-ngx-dlss.so and writes its logs, as a wchar_t string:
    // 32-bit code points here, as the Linux core expects.
    char dir[PLAT_PATH_MAX];
    wchar_t wdir[PLAT_PATH_MAX];
    char *cut = plat_exe_path(dir, sizeof dir) ? plat_last_sep(dir) : NULL;
    if (!cut || !utf8_to_wide((cut[1] = 0, dir), wdir, PLAT_PATH_MAX)) {
        rt_log("dlaa: cannot name the executable's folder for NGX: DLAA off");
        return;
    }
#endif
    // 231313132: the application id NVIDIA's DLSS samples use for builds without an NVIDIA-issued id
    const VkCtx *c = vk_ctx();
    NgxResult r = N.init(231313132ull, wdir, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                         vkGetDeviceProcAddr, NGX_VERSION_API, NULL);
    if (!NGX_OK(r)) {
        rt_log("dlaa: NGX init failed (%08x): DLAA off", (unsigned)r);
        return;
    }
    if (!vtable_ok()) return;
    NgxParam *caps = NULL;
    int avail = 0, driver = 0;
    if (NGX_OK(N.caps(&caps)) && caps) {
        p_geti(caps, "SuperSampling.Available", &avail);
        p_geti(caps, "SuperSampling.NeedsUpdatedDriver", &driver);
    }
    N.available = avail != 0;
    rt_log("dlaa: NGX up; DLSS %s%s", avail ? "available" : "NOT available (is " DLSS_MODEL " next to the executable?)",
           driver ? "; the driver needs an update" : "");
}

// ---------------------------------------------------------------- where DLAA runs in Astral Chain's frame
// Measured (AC_DUMP, frame 4200 outside the police HQ, 1920x1080 scene, the same order at ssaa = 2160p): Astral Chain
// renders forward. A depth-only prepass writes the scene depth (D32F_S8 at the scene size); the opaque draws, and after
// the shadow map the rest of the opaque and the transparents, go into the R11G11B10F light target with that depth
// (between them a full-screen copy of it and a half-size pass). The bloom chain then reads it at half size and down
// to 1/32; a composite pass writes scene and bloom into another R11G11B10F target, the tonemap and post passes go to
// RGBA8, and the HUD is drawn over the last of them.
// So: after the depth prepass, a pass at the scene size with exactly one colour target (R11G11B10F or RGBA16F) and
// the scene depth arms the frame with that pair; the next pass at most a quarter as wide (the bloom's third level,
// after the light target's last pass) ends the scene (no jitter after it) and, unless the scene set made the scene
// smaller than the output (scaled), runs DLAA on the armed target first. The bloom's first two levels have read the
// scene unresolved, which their blur hides; the composite, post and HUD work on the anti-aliased picture.
// Upscaling (a scene smaller than the output: AC_DLSS quality/balanced/performance through the scene patch set,
// resolution.c): measured (AC_DUMP, 960x540 scene, 1920x1080 output, the night highway save in a cutscene and in play)
// the whole scene runs at its size: depth prepass, light target, bloom chain, and the composite, a full-screen pass
// with the scene depth that writes bloom, tonemap and grading (values 0..1) into another R11G11B10F target of the
// scene's size. The SMAA edge and weight passes follow at that size, then the SMAA blending pass is the only scaling:
// one full-screen draw, no depth, that reads the composite and the weights into the RGBA8 target of the output's size
// the HUD is then drawn over. DLSS runs at the begin of that pass, on the composite (as LDR unless HDR output is on),
// into the size's output image; from there to the frame's end, passes larger than the composite sample that image in its place
// (vk_dlaa_swap_img, vk_exec_draw.c tex_info; after the TotK Recomp's, same author, MIT), so the blending draw scales
// DLSS's picture and the HUD is never jittered nor passed through DLSS. SMAA is off meanwhile (vk_dlaa_skip_draw).
static struct {
    VkExecTarget color, depth;   // the armed pair, valid in the frame it was armed
    int armed, done, gbuf;       // this frame: armed on the light target, DLAA ran, the G-buffer was drawn
    int ended;                   // this frame: the scene is drawn (jitter over)
    VkExecTarget comp;           // upscaling: this frame's composite (the scene after bloom, tonemap and grading)
    int comp_ok;
    struct Size {                // a DLSS feature and its work images per scene size (dynamic resolution flips it)
        uint32_t w, h, ow, oh;   // render size, output size
        VkImage out, mv;
        VmaAllocation out_mem, mv_mem;
        VkImageView out_view, mv_view;
        VkImage x[4];            // at the render size: the object marks of this and last frame (R8, dlaa_objmask.comp),
        VmaAllocation x_mem[4];  // the bias mask DLSS reads (R8), and the camera's vectors before the object pass
        VkImageView x_view[4];   // (RG16F)
        int obj_cur, obj_ok;     // x[obj_cur] takes this frame's marks; obj_ok: x[obj_cur ^ 1] holds last frame's
        NgxHandle *feat;
        uint64_t used;           // evals count when it last ran
    } sizes[5], *last;           // last: the one the previous DLAA frame ran with
    VkImage last_color[2];       // the targets DLAA ran on, latest first: another one means a new history
    NgxParam *params;
    int failed, reset;
    uint64_t evals, frames_missed;
    // jitter (vk_dlaa_viewport)
    VkImage jdepth;              // the scene depth of the last frame DLAA ran on: passes drawing with it are jittered
    float jx, jy;                // the offset the next scene draws take, render pixels (window x right, y down)
    float ux, uy;                // the offset this frame's scene draws took (DLSS and the motion vectors get these)
    uint32_t jframe;
    int jit_ok, jit_used, hold;  // jitter allowed this frame / a draw used it / frames left without it
    uint64_t holds;
    // camera (vk_dlaa_draw) and motion vectors (mv_pass)
    uint64_t cam_addr;           // this frame's scene camera block (c3 of a scene draw's vertex stage), 0: none seen
    struct { uint64_t addr; uint32_t n; } cams[4];   // this frame's camera blocks and the scene draws that bound each
    double cam_prev[16];         // last frame's view-projection, for the reprojection
    int cam_prev_ok;             // cam_prev holds last frame's
    VkBuffer mvm;                // MVM_SLOTS reprojection matrices for dlaa_mv.comp, host-visible
    VmaAllocation mvm_mem;
    uint8_t *mvm_map;
    uint32_t mvm_slot, mvm_stride;
    VkDescriptorSetLayout mv_dsl;
    VkPipelineLayout mv_layout;
    VkPipeline mv_pipe;
    VkSampler mv_samp;
    int mv_ok;                   // -1 not built, 0 failed, 1 ready
    uint64_t mv_frames, mv_none, masked;   // masked: frames with the characters' bias mask
    uint64_t cuts;                         // camera cuts (mv_pass): a new history
    double view_prev[16];                  // last frame's view (rows 0-3 of the camera block), for cuts
    int view_ok;
    uint64_t why[3];             // history resets: scene size changed, no camera, scene target changed
    int on;                      // the dlaa setting, read at each present (the menu switches it live)
    int preset, preset_dirty;    // the DLSS model features are created with (preset_value); a change pending
    unsigned cfg_seen;           // cfg_version when preset was read
    uint64_t last_eval_ns;       // plat_now_ns of the previous evaluation (FrameTimeDeltaInMsec)
} D = {.mv_ok = -1, .cfg_seen = ~0u};

const VkTexImage *vk_dlaa_swap_img;
VkImageView vk_dlaa_swap_view;

static int dl_active(void) { return N.available && !D.failed && D.on; }

// The scene set makes the scene smaller than the output (res_sizes; a mode change takes effect once the game rebuilt
// its scene targets, resolution.c rt_res_rebuild_w8): DLSS scales instead of DLAA.
static int scaled(void) {
    unsigned o[2], s[2];
    res_sizes(o, s);
    return s[0] < o[0];
}

// A scene target of this width is smaller than the output, whose targets ssaa scales as well. Asked of the target and
// not of scaled(): res_sizes moves when the game starts its rebuild, and the frames already drawn at the old size are
// still to come. Judged by the setting, a frame at the old small size ran in-place DLAA, which made a feature of its
// own for that frame, and a frame at the old full size waited for a scaling pass that never came and shook.
static int target_scaled(uint32_t w) {
    unsigned o[2], s[2];
    res_sizes(o, s);
    return (float)w + 2.0f < (float)o[0] * vk_render_scale();
}

// A target of the output's size, with what ssaa makes of it (its rounding is vk_texture.c's: within two pixels).
static int output_sized(const VkExecTarget *c) {
    unsigned o[2], s[2];
    res_sizes(o, s);
    float k = vk_render_scale();
    return fabsf((float)c->width - (float)o[0] * k) <= 2.0f && fabsf((float)c->height - (float)o[1] * k) <= 2.0f;
}

static void dl_fail(const char *what, NgxResult r) {
    rt_log("dlaa: %s failed (%08x): DLAA off for this run", what, (unsigned)r);
    D.failed = 1;
}

static int image2d(uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage, VkImage *img, VmaAllocation *mem,
                   VkImageView *view) {
    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, mem, NULL) != VK_SUCCESS) return 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, view) == VK_SUCCESS;
}

static void free_size(struct Size *e) {
    const VkCtx *c = vk_ctx();
    // an evicted size or a new model: the images may still be in flight. vkDeviceWaitIdle needs every queue externally
    // synchronised, and the present and the upload threads submit too.
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    if (e->out_view) vkDestroyImageView(c->device, e->out_view, NULL);
    if (e->mv_view) vkDestroyImageView(c->device, e->mv_view, NULL);
    if (e->out) vmaDestroyImage(c->vma, e->out, e->out_mem);
    if (e->mv) vmaDestroyImage(c->vma, e->mv, e->mv_mem);
    for (int i = 0; i < 4; i++) {
        if (e->x_view[i]) vkDestroyImageView(c->device, e->x_view[i], NULL);
        if (e->x[i]) vmaDestroyImage(c->vma, e->x[i], e->x_mem[i]);
    }
    if (e->feat) N.release(e->feat);
    memset(e, 0, sizeof *e);
}

// The entry for a scene and output size: its output (RGBA16F at the output size, written by DLSS, then blitted into
// the target) and motion vectors (RG16F at the scene size, render pixels), both kept in GENERAL like every image of the
// back end. The TotK Recomp keeps each size's feature because its dynamic resolution flips between them. Here the scene
// set gives every dynamic-resolution level one size, so only a mode change brings another, already a hitch of a target
// rebuild: making a new entry frees those unused for SIZE_IDLE evaluations, or five modes at a 4K output would hold
// five features' worth of video memory for the rest of the run. The least recently used goes if none is free.
#define SIZE_IDLE 300u
static struct Size *size_for(VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    struct Size *e = NULL, *lru = &D.sizes[0];
    for (int i = 0; i < 5; i++) {
        if (D.sizes[i].w == w && D.sizes[i].h == h && D.sizes[i].ow == ow && D.sizes[i].oh == oh) return &D.sizes[i];
        if (!D.sizes[i].out && !e) e = &D.sizes[i];
        if (D.sizes[i].used < lru->used) lru = &D.sizes[i];
    }
    if (!e) {
        if (D.last == lru) D.last = NULL;
        free_size(e = lru);
    }
    for (int i = 0; i < 5; i++)
        if (D.sizes[i].out && D.sizes[i].used + SIZE_IDLE < D.evals) {
            if (D.last == &D.sizes[i]) D.last = NULL;
            free_size(&D.sizes[i]);
        }
    VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageUsageFlags ux = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // the motion image is also the per-object pass's colour target (vk_mvobj_execute)
    int made = image2d(ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, u, &e->out, &e->out_mem, &e->out_view) &&
               image2d(w, h, VK_FORMAT_R16G16_SFLOAT, u | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &e->mv, &e->mv_mem, &e->mv_view);
    for (int i = 0; i < 4 && made; i++)
        made = image2d(w, h, i < 3 ? VK_FORMAT_R8_UNORM : VK_FORMAT_R16G16_SFLOAT, ux, &e->x[i], &e->x_mem[i], &e->x_view[i]);
    if (!made) {   // out of video memory, most likely: what was made goes back, DLSS turns off (evaluate) for this run
        free_size(e);
        return NULL;
    }
    VkImageMemoryBarrier2 ib[6];
    VkImage im[6] = {e->out, e->mv, e->x[0], e->x[1], e->x[2], e->x[3]};
    for (int i = 0; i < 6; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 6,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    e->w = w, e->h = h, e->ow = ow, e->oh = oh, e->obj_cur = 0, e->obj_ok = 0;
    return e;
}

static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static NgxResourceVK res_image(VkImageView v, VkImage img, VkFormat f, uint32_t w, uint32_t h, VkImageAspectFlags a,
                               int rw) {
    NgxResourceVK r;
    memset(&r, 0, sizeof r);
    r.r.img = (NgxImageView){v, img, {a, 0, 1, 0, 1}, f, w, h};
    r.type = 0;
    r.rw = rw;
    return r;
}

// ---------------------------------------------------------------- timing (AC_DLAA_TIMES=1)
// GPU timestamps around the camera motion dispatch, the per-object motion pass and DLSS's evaluation, and the CPU
// time of NGX's call, of draining the recorder before it and of recording the object pass, summed and logged every
// TIMES_SUMMARY frames:
//   dlaa times: gpu camera motion a ms, object motion b ms, DLSS c ms (frames timed); cpu: recorder drain d ms, NGX
//   evaluate e ms, object pass recording f ms
// Each frame takes its own slot of TIMES_SLOTS; a slot's results are read when it comes round again (by then the GPU is
// long done with it; not ready: that frame is left out). After the TotK Recomp's vk_dlaa.c timing section.
enum { T_MV0, T_MV1, T_OBJ, T_DL0, T_DL1, T_N };
#define TIMES_SLOTS 8u
#define TIMES_SUMMARY 600
static struct {
    int on;                      // -1 not read
    VkQueryPool pool;
    uint32_t slot;
    int began;                   // this frame's mv_pass began a slot (a frame without camera motion takes none)
    int used[TIMES_SLOTS];       // the slot holds a frame's timestamps (all T_N written)
    double gpu[3];               // ms summed: camera motion, object motion, DLSS
    uint64_t frames, cpu_drain, cpu_ngx, cpu_mv;   // ns summed
} T = {.on = -1};

static int times_on(void) {
    if (T.on < 0) {
        T.on = rt_knob("DLAA_TIMES", "0");
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = TIMES_SLOTS * T_N};
        if (T.on && vkCreateQueryPool(vk_ctx()->device, &qi, NULL, &T.pool) != VK_SUCCESS) T.on = 0;
    }
    return T.on;
}

static void times_mark(VkCommandBuffer cb, int which) {
    if (times_on()) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, T.pool, T.slot * T_N + (uint32_t)which);
}

// At the start of a frame's timing (evaluate): the slot's last results are read and it is reset.
static void times_begin(VkCommandBuffer cb) {
    if (!times_on()) return;
    T.slot = (T.slot + 1) % TIMES_SLOTS;
    if (T.used[T.slot]) {
        uint64_t q[T_N];
        if (vkGetQueryPoolResults(vk_ctx()->device, T.pool, T.slot * T_N, T_N, sizeof q, q, sizeof q[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            double ms = vk_ctx()->props.limits.timestampPeriod / 1e6;   // ticks to ms
            T.gpu[0] += (double)(q[T_MV1] - q[T_MV0]) * ms, T.gpu[1] += (double)(q[T_OBJ] - q[T_MV1]) * ms;
            T.gpu[2] += (double)(q[T_DL1] - q[T_DL0]) * ms;
            if (++T.frames % TIMES_SUMMARY == 0) {
                rt_log("dlaa times: gpu camera motion %.3f ms, object motion %.3f ms, DLSS %.3f ms (%llu frames timed); "
                       "cpu: recorder drain %.3f ms, NGX evaluate %.3f ms, object pass recording %.3f ms",
                       T.gpu[0] / TIMES_SUMMARY, T.gpu[1] / TIMES_SUMMARY, T.gpu[2] / TIMES_SUMMARY,
                       (unsigned long long)T.frames, T.cpu_drain / 1e6 / TIMES_SUMMARY, T.cpu_ngx / 1e6 / TIMES_SUMMARY,
                       T.cpu_mv / 1e6 / TIMES_SUMMARY);
                T.gpu[0] = T.gpu[1] = T.gpu[2] = 0, T.cpu_drain = T.cpu_ngx = T.cpu_mv = 0;
            }
        }
    }
    T.used[T.slot] = 0, T.began = 1;
    vkCmdResetQueryPool(cb, T.pool, T.slot * T_N, T_N);
}

// ---------------------------------------------------------------- motion vectors
// Camera motion only (runtime/shaders/dlaa_mv.comp): this frame's view-projection is read from the replay's uniform
// snapshot (vk_usnap.c, as of the submit call), the reprojection from this frame's clip space to last frame's is made
// from it and last frame's in double precision (inv4, mv_pass), and the compute pass takes every pixel's depth through
// it. Characters and anything else that moves on its own get the camera's motion only, which DLSS mostly copes with
// by rejecting their history.
#define CAM_BYTES 256u   // vec4 rows 0-15: view, projection, view-projection (8-11), inverse view
#define CAM_VP 128u      // byte offset of the view-projection in the block
#define MVM_SLOTS 16u    // frames the GPU may still be reading the reprojection blocks of
#define CUT_DEG 30.0     // a camera cut: the view turned more than this in one frame (degrees)
#define CUT_DIST 5.0     // or moved more than this (units, metres)

static int mv_init(void) {
    if (D.mv_ok >= 0) return D.mv_ok;
    D.mv_ok = 0;
    const VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 3, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlaa_mv_spv,
                                   .pCode = dlaa_mv_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    VkDeviceSize al = c->props.limits.minUniformBufferOffsetAlignment;
    D.mvm_stride = (uint32_t)((64 + al - 1) / al * al);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)MVM_SLOTS * D.mvm_stride,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo minfo;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &D.mv_dsl) != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &D.mv_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &D.mv_layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &D.mv_samp) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS ||
        vmaCreateBuffer(c->vma, &bi, &ai, &D.mvm, &D.mvm_mem, &minfo) != VK_SUCCESS)
        goto fail;
    D.mvm_map = minfo.pMappedData;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = D.mv_layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &D.mv_pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return D.mv_ok = 1;
fail:
    rt_log("dlaa: the motion-vector pass could not be set up: DLAA runs with zero motion");
    return 0;
}

// The characters' bias mask (dlss_char_mask): runtime/shaders/dlaa_objmask.comp, built at its first use.
static struct {
    int ok;                      // -1 not built, 0 failed, 1 ready
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
} MK = {.ok = -1};

static int mask_init(void) {
    if (MK.ok >= 0) return MK.ok;
    MK.ok = 0;
    const VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding b[5];
    for (uint32_t i = 0; i < 5; i++)
        b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 5, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof dlaa_objmask_spv, .pCode = dlaa_objmask_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &MK.dsl) != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &MK.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &MK.layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = MK.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &MK.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return MK.ok = 1;
fail:
    rt_log("dlaa: the characters' mask pass could not be set up: DLSS runs without it");
    return 0;
}

// How strongly DLSS favours this frame on moving objects and where they just were (AC_DLSS_CHAR_MASK, percent, live;
// 0 off). With per-object vectors the history already follows them, but an animated character still leaves a faint
// copy of itself where its vectors are a little off; the mask has DLSS take less history there, at the cost of some
// of its anti-aliasing on those pixels. As the TotK Recomp's char_bias, whose default is 65; here 0: with the
// per-object vectors, AC_MV_CHECK's output against input on the pixels that move on their own read the same with the
// mask at 0 and 65 (0.0572 and 0.0573, a running character outside the police HQ).
static float char_bias(void) {
    int n = cfg_int("AC_DLSS_CHAR_MASK");
    return n <= 0 ? 0.0f : n >= 100 ? 1.0f : (float)n / 100.0f;
}

// 4x4 inverse by Gauss-Jordan with partial pivoting, row-major, in double precision. 0: singular.
// (As the TotK Recomp's vk_dlaa.c inv4.)
static int inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-300) return 0;
        for (int k = 0; k < 8; k++) { double t = a[col][k]; a[col][k] = a[piv][k]; a[piv][k] = t; }
        double d = a[col][col];
        for (int k = 0; k < 8; k++) a[col][k] /= d;
        for (int r = 0; r < 4; r++)
            if (r != col) {
                double f = a[r][col];
                for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) o[r * 4 + k] = a[r][k + 4];
    return 1;
}

// The pass, recorded through the recorder like the replay's own commands. 0: no camera this frame, or none last frame
// (the vectors stay zero and DLSS starts a new history).
static int mv_pass(VkExec *ex, struct Size *e, VkImageView zv, float jx, float jy, float bias, int *masked) {
    VkCommandBuffer cb = ex->cb;
    uint32_t w = e->w, h = e->h;
    *masked = 0;
    const float *cr = mv_init() && D.cam_addr ? (const float *)vk_usnap_bytes(D.cam_addr + CAM_VP, 64) : NULL;
    const float *vw = cr ? (const float *)vk_usnap_bytes(D.cam_addr, 64) : NULL;   // the view (rows 0-3), for cuts
    if (!cr) {
        D.mv_none++;
        D.cam_prev_ok = 0, D.view_ok = 0, e->obj_ok = 0;
        return 0;
    }
    // this frame's view-projection as the snapshot holds it; kept as last frame's for the next
    double cur_vp[16], ci[16], m[16];
    for (int i = 0; i < 16; i++) cur_vp[i] = cr[i];
    int have_m = D.cam_prev_ok && inv4(cur_vp, ci);
    if (have_m)
        for (int r = 0; r < 4; r++)
            for (int k = 0; k < 4; k++) {
                double v = 0;
                for (int j = 0; j < 4; j++) v += D.cam_prev[r * 4 + j] * ci[j * 4 + k];
                m[r * 4 + k] = v;
            }
    memcpy(D.cam_prev, cur_vp, sizeof cur_vp);
    D.cam_prev_ok = 1;
    // A camera cut (a cutscene's next shot, a respawn): the view turned more than CUT_DEG or moved more than CUT_DIST
    // units (metres) in one frame, which no continuous camera does at 60 fps. The view, not the picture, is compared: a
    // cutscene's fast zoom moves the picture's corners by half the screen a frame for seconds. The history is then of
    // another shot, so DLSS starts a new one from zero motion, as NVIDIA's DLSS guide asks for cuts.
    int cut = 0;
    if (have_m && vw && D.view_ok) {
        double tr = 0, dist2 = 0;   // trace of R_prev R_cur^T; camera positions -R^T t
        double pc[3] = {0, 0, 0}, pp[3] = {0, 0, 0};
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 3; k++) tr += D.view_prev[r * 4 + k] * vw[r * 4 + k];
        for (int k = 0; k < 3; k++)
            for (int r = 0; r < 3; r++) pc[k] -= vw[r * 4 + k] * vw[r * 4 + 3], pp[k] -= D.view_prev[r * 4 + k] * D.view_prev[r * 4 + 3];
        for (int k = 0; k < 3; k++) dist2 += (pc[k] - pp[k]) * (pc[k] - pp[k]);
        double c = (tr - 1) / 2, deg = acos(c > 1 ? 1 : c < -1 ? -1 : c) * 57.29577951308232;
        cut = deg > CUT_DEG || dist2 > CUT_DIST * CUT_DIST;
        if (cut && ++D.cuts <= 20)
            rt_log("dlaa: camera cut at DLAA frame %llu (turned %.0f degrees, moved %.1f): new history",
                   (unsigned long long)D.evals, deg, sqrt(dist2));
    }
    // kept on every way out, or the first frame after a gap in the motion is measured against the view before the gap
    if (vw) for (int i = 0; i < 16; i++) D.view_prev[i] = vw[i];
    D.view_ok = vw != NULL;
    if (!have_m || cut) return e->obj_ok = 0;   // nothing to reproject into: zero motion, new history
    uint32_t ms = D.mvm_slot++ % MVM_SLOTS;
    float *blk = (float *)(D.mvm_map + (size_t)ms * D.mvm_stride);
    for (int i = 0; i < 16; i++) blk[i] = (float)m[i];
    vmaFlushAllocation(vk_ctx()->vma, D.mvm_mem, (VkDeviceSize)ms * D.mvm_stride, 64);   // a no-op on coherent memory
    VkDescriptorImageInfo zi = {D.mv_samp, zv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, e->mv_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo rep = {D.mvm, (VkDeviceSize)ms * D.mvm_stride, 64};
    VkWriteDescriptorSet wr[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &zi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &rep}};
    times_begin(cb);
    times_mark(cb, T_MV0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_layout, 0, 3, wr);
    float pc[8] = {(float)w, (float)h, jx, jy, nvn_depth_minus_w() ? 1.0f : 0.0f, 0, 0, 0};
    vkCmdPushConstants(cb, D.mv_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    times_mark(cb, T_MV1);
    vk_mvcheck_camera(cb, e->mv, w, h, jx, jy);   // AC_MV_CHECK (test)
    // the camera's vectors kept aside for the characters' mask, which marks where the object pass changed them
    int mask = bias > 0 && mask_init();
    if (mask) {
        VkImageCopy ic = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                          {w, h, 1}};
        vkCmdCopyImage(cb, e->mv, VK_IMAGE_LAYOUT_GENERAL, e->x[3], VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
        full_barrier(cb);
    }
    // what moves on its own over the camera's motion (vk_exec_draw.c, per-object motion)
    uint64_t c0 = T.on > 0 ? kern_now_ns() : 0;
    if (vk_mvobj_execute(ex, D.cam_addr, e->mv_view, &D.depth, w, h)) full_barrier(cb);
    if (T.on > 0) T.cpu_mv += kern_now_ns() - c0;
    times_mark(cb, T_OBJ);
    vk_mvcheck_final(cb, ex->fence, e->mv, D.color.image, D.color.format, D.color.level, D.color.base_layer, w, h, jx, jy);
    if (mask) {   // DLSS's bias mask, and this frame's marks for the next (dlaa_objmask.comp)
        VkDescriptorImageInfo xi[5] = {{VK_NULL_HANDLE, e->x_view[3], VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, e->x_view[e->obj_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, e->mv_view, VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, e->x_view[2], VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, e->x_view[e->obj_cur], VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet mw[5];
        for (int i = 0; i < 5; i++)
            mw[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .pImageInfo = &xi[i]};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, MK.pipe);
        rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, MK.layout, 0, 5, mw);
        float mp[4] = {(float)w, (float)h, bias, e->obj_ok ? 1.0f : 0.0f};
        vkCmdPushConstants(cb, MK.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof mp, mp);
        vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
        full_barrier(cb);
        *masked = 1;
    }
    e->obj_cur ^= 1, e->obj_ok = mask;   // the marks were written only with the mask
    if (T.on > 0) T.used[T.slot] = 1;   // the DLSS marks follow in evaluate
    D.mv_frames++;
    return 1;
}

// ---------------------------------------------------------------- jitter
// The scene's draws (every pass drawing with the scene depth, until the DLAA point) have their viewport moved by a
// sub-pixel Halton(2,3) offset, which shifts what they rasterise as a jittered projection would; full-screen passes
// that shade by window position are unaffected. Jitter only follows a frame DLSS ran on: a jittered frame that DLSS
// does not resolve shakes on screen, so a missed DLAA point stops the jitter for a second.
static float halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

void vk_dlaa_viewport(VkExec *ex, VkViewport *v) {
    if (!D.jit_ok || D.ended || !ex->depth.image || ex->depth.image != D.jdepth) return;
    v->x += D.jx, v->y += D.jy;
    D.ux = D.jx, D.uy = D.jy, D.jit_used = 1;
}

int vk_dlaa_available(void) { return dl_active(); }

int vk_dlaa_scene_draw(const VkExec *ex) {
    const VkImage z = ex->depth.image;
    return dl_active() && !D.ended && z && ((D.armed && z == D.depth.image) || (D.jdepth && z == D.jdepth));
}

void vk_dlaa_jitter(float j[2]) { j[0] = D.jit_ok ? D.jx : 0.0f, j[1] = D.jit_ok ? D.jy : 0.0f; }

// The camera block is the vertex stage's c3 (256 bytes, the layout in dlaa_mv.comp) of a draw with the scene depth: the
// armed pass's, or before arming (the depth prepass) the depth of the last frame DLAA ran on. The shadow map passes
// bind another camera with another depth target, the HUD has no scene depth.
void vk_dlaa_draw(VkExec *ex) {
    if (!dl_active() || D.ended) return;
    const p_bind_buf *b = &ex->ubo[0][0];   // constant bank c3: the replay keeps user banks from c3 on
    const VkExecTarget *z = &ex->depth;
    if (!(b->size == CAM_BYTES && b->addr && z->image &&
          ((D.armed && z->image == D.depth.image) || (D.jdepth && z->image == D.jdepth))))
        return;
    // the block most scene draws bind: a few draws into the scene depth bind another camera block (measured in the
    // night highway save, where the last one seen alternated between two cameras 160 units apart)
    int i = 0, low = 0;
    for (; i < 4 && D.cams[i].addr && D.cams[i].addr != b->addr; i++)
        if (D.cams[i].n < D.cams[low].n) low = i;
    if (i == 4) i = low, D.cams[i].addr = b->addr, D.cams[i].n = 0;   // full: the rarest makes room
    D.cams[i].addr = b->addr, D.cams[i].n++;
    if (D.cams[i].addr != D.cam_addr) {
        uint32_t best = 0;
        for (int k = 0; k < 4; k++)
            if (D.cams[k].addr && D.cams[k].n > best) best = D.cams[k].n, D.cam_addr = D.cams[k].addr;
    }
}

// The jitter moves on right after DLAA ran, not at the present: the replay can record the next frame's scene before
// the present of this one is handled, and a step taken there landed in the middle of a frame (worse after an off/on
// switch, which shifted that order). Jitter needs a frame DLSS ran on; a frame without the DLAA point stops it.
// The sequence is as long as NVIDIA's DLSS guide asks for the upscaling ratio, 8 x (output / scene)^2 phases (18 in
// Quality, 24 in Balanced, 32 in Performance), and 16 at least (DLAA).
static void next_jitter(uint32_t w, uint32_t ow) {
    static int nj = -1;
    if (nj < 0) nj = !rt_knob("DLAA_JITTER", "1");   // AC_DLAA_JITTER=0: no jitter (test)
    double r = w ? (double)ow / w : 1.0;
    uint32_t phases = (uint32_t)ceil(8.0 * r * r - 1e-6);
    if (phases < 16) phases = 16;
    if (phases > 72) phases = 72;
    uint32_t i = D.jframe++ % phases + 1;
    D.jx = halton(i, 2) - 0.5f, D.jy = halton(i, 3) - 0.5f;
    D.jit_ok = !D.hold && !nj;
}

// The DLSS model (AC_DLSS_PRESET, live): default takes NVIDIA's model per mode (evaluate: K for DLAA, Quality and
// Balanced, M for Performance, L for Ultra Performance), 0 here. A letter a..m takes that model in every mode: k pins
// K, j is the other transformer model, which NVIDIA describes as slightly less ghosting at the cost of extra flicker, e
// the older CNN model, cheaper on slower RTX cards. The value is an NVSDK_NGX_DLSS_Hint_Render_Preset (1 A ... 13 M),
// read when a feature is created; a change recreates the features (preset_check).
static int preset_value(void) {
    const char *v = cfg_get("AC_DLSS_PRESET");
    int c = v && v[0] && !v[1] ? v[0] | 0x20 : 0;
    return c >= 'a' && c <= 'm' ? c - 'a' + 1 : 0;
}

// The preset, read again when a setting changed (the menu); a different one recreates the features at the next DLAA run.
static void preset_check(void) {
    unsigned v = cfg_version();
    if (v == D.cfg_seen) return;
    D.cfg_seen = v;
    int p = preset_value();
    if (p == D.preset) return;
    if (D.evals) {
        if (p) rt_log("dlaa: DLSS preset %c in every mode; the features are created again", 'A' + p - 1);
        else rt_log("dlaa: DLSS presets per mode; the features are created again");
    }
    D.preset = p, D.preset_dirty = 1, D.reset = 1;
}

// The quality setting DLSS is created with, from the scene's share of the output width.
static int perf_quality(uint32_t w, uint32_t ow) {
    if (w >= ow) return NGX_PERFQ_DLAA;
    double k = (double)w / ow;
    return k >= 0.66 ? 2 : k >= 0.57 ? 1 : k >= 0.49 ? 0 : 3;
}

// src: what DLSS reads, NULL for the light target; dst: the scaling pass's target, NULL for DLAA in place. In place,
// DLSS's picture is blitted into the light target; scaled, it stays in the size's output image, which the scaling pass
// samples in place of src (vk_dlaa_swap_img).
static void evaluate(VkExec *ex, const VkExecTarget *src, const VkExecTarget *dst) {
    const VkExecTarget *t = src ? src : &D.color, *z = &D.depth;
    int up = dst != NULL;
    if (!dst) dst = t;
    uint32_t w = t->width, h = t->height, ow = dst->width, oh = dst->height;
    VkCommandBuffer cb = ex->cb;
    vk_exec_barrier(ex);   // what drew the scene is done (no pass is open here)
    if (D.preset_dirty) {   // another model: features are created with it, so every size gets a new one
        for (int i = 0; i < 5; i++)
            if (D.sizes[i].out) free_size(&D.sizes[i]);
        D.last = NULL, D.preset_dirty = 0;
    }
    struct Size *e = size_for(cb, w, h, ow, oh);
    if (!e) return dl_fail("work image creation", 0);
    if (e != D.last) D.reset = 1, D.why[0]++, e->obj_ok = 0;   // another size's history is from another moment
    if (!D.params && !NGX_OK(N.alloc_params(&D.params))) return dl_fail("parameter allocation", 0);
    NgxParam *p = D.params;
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    float jx = D.jit_used ? D.ux : 0.0f, jy = D.jit_used ? D.uy : 0.0f;
    static int no_mv = -1;
    if (no_mv < 0) no_mv = !rt_knob("DLAA_MV", "1");   // AC_DLAA_MV=0: zero motion (test)
    int masked = 0;
    T.began = 0;
    if (no_mv || !mv_pass(ex, e, zv, jx, jy, char_bias(), &masked)) {
        VkClearColorValue zero = {{0, 0, 0, 0}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, e->mv, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        full_barrier(cb);
        if (!no_mv) D.reset = 1, D.why[1]++;   // no camera: the history cannot be moved along
    }
    int timed = T.on > 0 && T.began && T.used[T.slot];   // mv_pass began this frame's slot and got to its end
    if (timed) times_mark(cb, T_DL0);
    uint64_t c0 = timed ? kern_now_ns() : 0;
    rec_drain();   // NGX records into the command buffer itself: the recorder thread must have made every call so far
    if (timed) T.cpu_drain += kern_now_ns() - c0;
    if (!e->feat) {
        p_setu(p, "CreationNodeMask", 1);
        p_setu(p, "VisibilityNodeMask", 1);
        p_setu(p, "Width", w);
        p_setu(p, "Height", h);
        p_setu(p, "OutWidth", ow);
        p_setu(p, "OutHeight", oh);
        p_seti(p, "PerfQualityValue", perf_quality(w, ow));
        // MVLowRes: motion vectors at the render size. In place (the light target): IsHDR | AutoExposure, linear scene
        // light without an exposure texture. Scaled (the composite, tonemapped and graded, 0..1): neither, unless HDR
        // output keeps highlights above 1 in it (vk_hdr.c).
        int create = (1 << 1) | (!up || vk_hdr_wanted() ? (1 << 0) | (1 << 6) : 0);
        p_seti(p, "DLSS.Feature.Create.Flags", create);
        p_seti(p, "DLSS.Enable.Output.Subrects", 0);
        // The model per mode (NVSDK_NGX_DLSS_Hint_Render_Preset: 1 A ... 11 K, 12 L, 13 M), as NVIDIA recommends for
        // DLSS 4.5 (310.x): K (transformer) for DLAA, Quality and Balanced, M for Performance, L for Ultra Performance.
        // M and L cost more on cards without FP8 (RTX 20/30). dlss_preset (D.preset, preset_check) takes one model for
        // every mode instead. An older nvngx_dlss.dll without a model takes its own default.
        int pk = D.preset ? D.preset : 11, pm = D.preset ? D.preset : 13, pl = D.preset ? D.preset : 12;
        p_setu(p, "DLSS.Hint.Render.Preset.DLAA", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.UltraQuality", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.Quality", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.Balanced", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.Performance", (unsigned)pm);
        p_setu(p, "DLSS.Hint.Render.Preset.UltraPerformance", (unsigned)pl);
        NgxResult r = N.create(cb, NGX_FEATURE_SUPERSAMPLING, p, &e->feat);
        if (!NGX_OK(r) || !e->feat) return dl_fail("DLSS feature creation", r);
        D.reset = 1;
        int q = perf_quality(w, ow), pr = q == 0 ? pm : q == 3 ? pl : pk;
        rt_log("dlaa: DLSS feature created at %ux%u -> %ux%u, quality value %d, preset %c (scene format %d, depth format "
               "%d)", w, h, ow, oh, q, 'A' + pr - 1, (int)t->format, (int)z->format);
    }
    // Astral Chain draws its light target into two images in turn: only one seen in neither of the last two frames
    // starts a new history
    if (t->image != D.last_color[0] && t->image != D.last_color[1]) D.reset = 1, D.why[2]++;
    if (t->image != D.last_color[0]) D.last_color[1] = D.last_color[0], D.last_color[0] = t->image;
    VkImageView cv = vk_exec_target_view(t->img, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                                         t->base_layer, 1, 0);
    NgxResourceVK rc = res_image(cv, t->image, t->format, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK ro = res_image(e->out_view, e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    NgxResourceVK rz = res_image(zv, z->image, z->format, w, h, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    NgxResourceVK rm = res_image(e->mv_view, e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "Color", &rc);
    p_setp(p, "Output", &ro);
    p_setp(p, "Depth", &rz);
    p_setp(p, "MotionVectors", &rm);
    p_setf(p, "Jitter.Offset.X", jx);
    p_setf(p, "Jitter.Offset.Y", jy);
    p_setf(p, "Sharpness", 0.0f);
    p_seti(p, "Reset", D.reset);
    // the characters' bias mask when mv_pass made one this frame; the parameter object is reused, so it is cleared
    // explicitly otherwise
    NgxResourceVK rk = res_image(e->x_view[2], e->x[2], VK_FORMAT_R8_UNORM, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "DLSS.Input.Bias.Current.Color.Mask", masked ? (void *)&rk : NULL);
    p_setu(p, "DLSS.Input.Bias.Current.Color.Subrect.Base.X", 0);
    p_setu(p, "DLSS.Input.Bias.Current.Color.Subrect.Base.Y", 0);
    p_setf(p, "MV.Scale.X", 1.0f);
    p_setf(p, "MV.Scale.Y", 1.0f);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Width", w);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Height", h);
    p_setf(p, "DLSS.Pre.Exposure", 1.0f);
    p_setf(p, "DLSS.Exposure.Scale", 1.0f);
    // The time since the previous evaluation, which DLSS weighs its history by; 60 fps before the first and after a
    // gap (a pause, a load), where no history carries over anyway.
    uint64_t now = plat_now_ns();
    double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
    D.last_eval_ns = now;
    p_setf(p, "FrameTimeDeltaInMsec", dt > 0 && dt < 100 ? (float)dt : 16.67f);
    c0 = timed ? kern_now_ns() : 0;
    NgxResult r = N.evaluate(cb, e->feat, p, NULL);
    if (timed) T.cpu_ngx += kern_now_ns() - c0;
    if (!NGX_OK(r)) return dl_fail("DLSS evaluation", r);
    if (timed) times_mark(cb, T_DL1);   // through the recorder, after NGX's own commands
    vk_mvprev_swap();   // the objects' blocks of this frame become last frame's for the next
    D.reset = 0;
    D.last = e, e->used = D.evals + 1;
    full_barrier(cb);
    vk_mvcheck_output(cb, e->out, ow, oh);   // AC_MV_CHECK (test)
    // DLAA: the result into the scene target (the bloom, the post passes and the HUD read it and draw over it). Scaled:
    // the scaling pass reads the output image in place of the composite; only the debug views below are blitted, into
    // that image.
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, up ? 0 : dst->level, up ? 0 : dst->base_layer, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}}};
    // AC_DLAA_MVDEBUG=1 (test): the motion vectors in place of the picture, x in red and y in green, render pixels as
    // scene light (the post passes still run over them)
    // AC_DLAA_MASKDEBUG=1 (test): the characters' bias mask in place of the picture, in red (2: this frame's object
    // marks). The game's exposure makes values of 1 dark; a readback is the surer test.
    static int mvdebug = -1, maskdebug = -1;
    if (mvdebug < 0) {
        const char *md = cfg_get("AC_DLAA_MASKDEBUG");
        mvdebug = rt_knob("DLAA_MVDEBUG", "0"), maskdebug = md && *md ? atoi(md) : 0;
    }
    int mk = maskdebug && masked;
    if (mvdebug || mk) b.srcOffsets[1] = (VkOffset3D){(int32_t)w, (int32_t)h, 1};
    VkImage from = mvdebug ? e->mv : mk ? e->x[maskdebug == 2 ? e->obj_cur ^ 1 : 2] : e->out;
    if (from != e->out || !up) {
        vkCmdBlitImage(cb, from, VK_IMAGE_LAYOUT_GENERAL, up ? e->out : dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                       VK_FILTER_NEAREST);
        full_barrier(cb);
    }
    if (up) vk_dlaa_swap_img = t->img, vk_dlaa_swap_view = e->out_view;   // the scaling draw reads DLSS's picture
    ex->recorded += 4;
    vk_exec_draw_new_cb();   // NGX bound pipelines and descriptor sets of its own: the next draw binds everything again
    D.jdepth = z->image;
    next_jitter(w, ow);
    D.masked += masked != 0;
    if (++D.evals == 1 || D.evals == 300 || D.evals % 3600 == 0)   // 300: one early line with the frames counted
        rt_log("dlaa: %llu frames anti-aliased (scene %ux%u, output %ux%u; %llu frames without the DLAA point, %llu with camera "
               "motion, %llu with the characters' mask, %llu without a camera, %llu jitter holds; resets: %llu size, %llu "
               "camera (%llu cuts), %llu target)",
               (unsigned long long)D.evals, w, h, ow, oh, (unsigned long long)D.frames_missed,
               (unsigned long long)D.mv_frames, (unsigned long long)D.masked, (unsigned long long)D.mv_none,
               (unsigned long long)D.holds, (unsigned long long)D.why[0], (unsigned long long)D.why[1],
               (unsigned long long)D.cuts, (unsigned long long)D.why[2]);
}

static int scene_color(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// A D32F(S8) depth at least as wide as the narrowest scene a mode asks for: Ultra Performance, a third of the output
// (640 at 1920x1080; 544 at the game's own 1600x900, which a fixed 640 never recognised).
static int scene_depth(const VkExecTarget *z) {
    unsigned o[2], s[2];
    res_sizes(o, s);
    return z->image && z->width >= o[0] / 3 && (z->format == VK_FORMAT_D32_SFLOAT_S8_UINT || z->format == VK_FORMAT_D32_SFLOAT);
}

void vk_dlaa_pass(VkExec *ex) {
    const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
    if (!dl_active() || D.done) return;
    // The depth prepass: no colour target, a D32F(S8) depth of a scene's width. Only light-target passes after it arm the
    // frame, so nothing drawn ahead of the scene can. The shadow map passes look the same; they fall between light-target
    // passes, so they leave the armed pair as it is.
    if (ex->pass_ncolor == 0 && scene_depth(z)) {
        D.gbuf = 1;
        return;
    }
    // A light-target pass: one HDR colour target with a depth of its size. The composite after the bloom has that shape
    // too, but comes after the end of the scene. A smaller pair never takes over an armed one (as the TotK Recomp found
    // with a half-size effect pass after the lighting: DLSS ran on it and wrote its picture over the scene).
    if (D.gbuf && !D.ended && ex->pass_ncolor == 1 && c->image && scene_color(c->format) && scene_depth(z) &&
        c->width == z->width && c->height == z->height && c->samples <= 1 && !c->is3d &&
        (!D.armed || (c->width >= D.color.width && c->height >= D.color.height))) {
        D.color = *c, D.depth = *z, D.armed = 1;
        return;
    }
    if (!D.armed) return;
    if (!D.ended) {
        if (!ex->pass_w || ex->pass_w * 4 > D.color.width) return;
        D.ended = 1;
        if (!target_scaled(D.color.width)) D.done = 1, evaluate(ex, NULL, NULL);   // DLAA in place; else wait for the scaling pass
        return;
    }
    // upscaling: the composite, a pass at the light target's size into one HDR-format target with the scene depth
    // (the latest such pass before the scaling pass)
    if (ex->pass_ncolor == 1 && c->image && scene_color(c->format) && z->image == D.depth.image &&
        c->width == D.color.width && c->height == D.color.height && c->samples <= 1 && !c->is3d) {
        D.comp = *c, D.comp_ok = 1;
        return;
    }
    // the game's scaling pass: one colour target of the output's size, wider than the scene, no depth (the SMAA blend,
    // which writes the RGBA8 picture the HUD is drawn over). As the TotK Recomp asks for the output's size: a pass of
    // another size (2560x1440 at ssaa = 2160p, on the title screen) made a DLSS feature of its own after each mode change.
    // A frame without a composite is left unresolved (a missed DLAA point).
    if (D.comp_ok && ex->pass_ncolor == 1 && c->image && !z->image && c->width > D.color.width && output_sized(c) &&
        c->samples <= 1 && !c->is3d) {
        D.done = 1;
        evaluate(ex, &D.comp, c);
    }
}

// AC_DLSS, read at each present (the menu switches it live): the scene size of the mode (resolution.c) and the
// texture bias that goes with it while DLSS runs. After the TotK Recomp's vk_dlaa.c dlss_mode (same author, MIT).
// A mode only means a smaller scene while DLSS can scale it. With dlaa switched off, NGX not up (no RTX GPU beside the
// DLL) or DLSS failed in this run, the scene goes back to the game's size, or the game's plain scaling would show it at
// 33..67 % of the output: the setting stays as it is, and the mode comes back with dlaa.
// AC_DLSS_CYCLE=<frames> (test): every that many presents the mode moves on to the next one, as a click in the menu
// would, so one scripted run goes through all of them.
static void dlss_mode(void) {
    static char mode[32];
    static float bias = 0;
    static int cycle = -1;
    static uint64_t presents;
    if (cycle < 0) {
        const char *v = cfg_get("AC_DLSS_CYCLE");
        cycle = v && *v ? atoi(v) : 0;
    }
    if (cycle > 0 && ++presents % (uint64_t)cycle == 0) {
        static const char *const modes[] = {"dlaa", "quality", "balanced", "performance", "ultra_performance"};
        int i = 0;
        while (i < 5 && strcmp(cfg_value("AC_DLSS"), modes[i])) i++;
        cfg_set("AC_DLSS", modes[(i + 1) % 5]);
    }
    const char *m = N.available && !D.failed && cfg_bool("AC_DLAA") ? cfg_value("AC_DLSS") : "dlaa";
    if (strcmp(m, mode)) {
        snprintf(mode, sizeof mode, "%s", m);
        res_dlss_mode(m);
    }
    float b = dl_active() ? res_dlss_lod_bias() : 0.0f;
    if (b != bias) bias = b, vk_sampler_dlss_bias(b);
}

void vk_dlaa_frame(void) {
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE;
    dlss_mode();
    if (N.available) preset_check();
    int on = N.available && cfg_bool("AC_DLAA");
    if (on != D.on) {
        rt_log("dlaa: %s", on ? "on" : "off");
        D.on = on;
        // either way the next DLAA frame starts a new history from an unjittered frame
        D.reset = 1, D.cam_prev_ok = D.view_ok = 0, D.jit_ok = D.jit_used = D.armed = D.done = D.gbuf = D.ended = 0;
        D.comp_ok = 0;
        D.hold = 0;
        return;
    }
    if (!dl_active()) return;
    if (!D.done) D.frames_missed++;
    if (D.jit_used && !D.done) D.hold = 60, D.holds++;   // jittered but not resolved: that frame shook
    else if (D.hold) D.hold--;
    if (!D.done) D.jit_ok = 0;
    D.armed = D.done = D.jit_used = D.gbuf = D.ended = D.comp_ok = 0;
    D.cam_addr = 0;
    memset(D.cams, 0, sizeof D.cams);
}

// ---------------------------------------------------------------- the game's SMAA (AC_SMAA, live)
// Astral Chain anti-aliases with SMAA 1x, three full-screen passes after the composite (measured: AC_DUMP frame 4200
// outside the police HQ; the key was fxaa before, cfg.c migrate_smaa): edge detection into an RGBA8 target, blend
// weights from those edges and SMAA's area (160x560) and search (64x16) textures, then neighbourhood blending of the
// HDR picture by the weights, which also writes the 8-bit picture the HUD is drawn over. With smaa = 0 the edge pass
// is skipped and the weights pass clears its target instead of drawing, so the blending pass, which still runs, blends
// nothing. Shaders are known by the FNV-1a hash of their SPIR-V, as vk_dump.c names them; a translator change that
// gives them other hashes makes the switch do nothing.
// While DLSS upscales, SMAA is off whatever the setting: its weights are found on the small picture, and the blending
// pass, which also scales, would apply them to DLSS's full-size one, smearing edges by a scene pixel. With the weights
// zero that pass only scales (a plain lookup of DLSS's picture at each output pixel). A switch that stops finding them
// says so once, as the TotK Recomp's FXAA switch does: the first skip, or a minute of frames (3600) with none.
static const uint64_t SMAA_EDGES = 0x312b4b91aa889b89ull, SMAA_WEIGHTS = 0xc7dc68a5786be9ceull;

static uint64_t spirv_hash(const GfxShader *s) {
    static struct { const GfxShader *s; uint64_t h; } tab[8192];
    uintptr_t k = ((uintptr_t)s >> 4) * 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0, j = (uint32_t)(k >> 51); i < 8192; i++, j = (j + 1) & 8191) {
        if (tab[j].s == s) return tab[j].h;
        if (!tab[j].s) {
            uint64_t words = 0, h = 0xcbf29ce484222325ull;
            const uint8_t *b = (const uint8_t *)gfx_shader_spirv(s, &words);
            for (uint64_t n = 0; b && n < words * 4; n++) h = (h ^ b[n]) * 0x100000001b3ull;
            tab[j].s = s, tab[j].h = h;
            return h;
        }
    }
    return 0;   // table full: no skip
}

int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs) {
    static uint64_t frame_read = ~0ull;
    static int smaa = 1;
    static uint64_t off_frames;
    static int found, reported;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f != frame_read) {   // once a frame: the menu switches it live
        frame_read = f;
        smaa = cfg_bool("AC_SMAA") && !(scaled() && dl_active());
        if (!smaa && !found && !reported && ++off_frames == 3600) {
            reported = 1;
            rt_log("dlaa: smaa off found no draw of the game's SMAA in %llu frames (its fragment shaders are not "
                   "SMAA_EDGES and SMAA_WEIGHTS): SMAA stays on", (unsigned long long)off_frames);
        }
    }
    if (smaa || !fs) return 0;   // nothing to skip: no shader lookup per draw at the defaults
    uint64_t h = spirv_hash(fs);
    if ((h == SMAA_EDGES || h == SMAA_WEIGHTS) && !found) {
        found = 1;
        rt_log("dlaa: SMAA off skips the game's SMAA (fragment shader %016llx)", (unsigned long long)h);
    }
    if (h == SMAA_EDGES) return 1;
    if (h == SMAA_WEIGHTS) {   // zero weights: the blending pass passes the picture through
        VkClearAttachment ca = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0};
        VkClearRect r = {.rect = {{0, 0}, {ex->pass_w, ex->pass_h}}, .baseArrayLayer = 0, .layerCount = 1};
        vkCmdClearAttachments(ex->cb, 1, &ca, 1, &r);
        ex->recorded++;
        return 1;
    }
    return 0;
}
