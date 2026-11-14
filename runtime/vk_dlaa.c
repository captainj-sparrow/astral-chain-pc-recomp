// The scene's temporal upscaler, chosen by AC_UPSCALING (astralchain.ini upscaling = off, auto, dlss, xess or fsr3;
// cfg.c cfg_upscaling) in the mode AC_UPSCALING_QUALITY sized the scene for at start (native, or a smaller scene:
// resolution.c): NVIDIA DLSS (DLAA at the scene's own size), Intel XeSS (vk_xess.c) or AMD FSR 3 (vk_fsr3.c); with none,
// AC_ANTI_ALIASING leaves the game's own SMAA on or off. This file finds the scene in Astral Chain's frame, jitters its
// draws, makes the motion vectors (the camera's, then what moves on its own over them) and runs the backend: at the
// scene's own size on its light target before the bloom, or from a smaller scene to the game resolution at the game's
// scaling pass; the sections below say how. All of it is portable Vulkan except loading NGX, which is Windows-only
// (core_load); XeSS's library is Windows-only as well (vk_xess.c), while FSR 3's passes are compiled in and run
// everywhere.
//
// NGX is loaded at run time from the driver's NGX core (_nvngx.dll), which sits beside NVIDIA's Vulkan driver: once
// the instance exists that driver is in the process and its module gives the folder (core_load). The registry key older
// drivers and Proton write (HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore) is the fallback. The DLSS model
// (nvngx_dlss.dll) is loaded by NGX from the folder of
// astralchain.exe. No NVIDIA SDK code or header is compiled in: the few NGX types used are declared here from the SDK's
// documented ABI, the way the Metroid Dread recomp's dlss.c reaches the core (read for behaviour only).
// The core's parameter object is a C++ class whose virtual functions are called through the MSVC vtable: the Set
// overloads in slots 0-7 and the Get overloads in 8-15, each group in reverse declaration order. vtable_ok checks that
// layout with a set-then-get round trip before anything relies on it.
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
#include "vk_fsr3.h"
#include "vk_xess.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "dlaa_mv_spv.h"

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
typedef NgxResult (*PFN_Shutdown)(VkDevice);

static struct {
    void *core;              // the NGX core's module (HMODULE)
    PFN_Init init;
    PFN_ReqExt req_ext;
    PFN_Params caps, alloc_params;
    PFN_DestroyParams destroy_params;
    PFN_Create create;
    PFN_Evaluate evaluate;
    PFN_Release release;
    PFN_Shutdown shutdown;   // NVSDK_NGX_VULKAN_Shutdown1; NULL in a core without it
    int tried, loaded, available;
    int inited;              // NVSDK_NGX_VULKAN_Init_Ext2 succeeded: vk_dlaa_release shuts NGX down
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
enum { S_P = 0, S_D12, S_D11, S_I, S_U32, S_D, S_F, S_U64, G_P, G_D12, G_D11, G_I, G_U32, G_D, G_F, G_U64 };
static void p_seti(NgxParam *p, const char *n, int v) { ((SetI)p->vt->slot[S_I])(p, n, v); }
static void p_setu(NgxParam *p, const char *n, unsigned v) { ((SetU)p->vt->slot[S_U32])(p, n, v); }
static void p_setf(NgxParam *p, const char *n, float v) { ((SetF)p->vt->slot[S_F])(p, n, v); }
static void p_setp(NgxParam *p, const char *n, void *v) { ((SetP)p->vt->slot[S_P])(p, n, v); }
static int p_geti(const NgxParam *p, const char *n, int *v) { return NGX_OK(((GetI)p->vt->slot[G_I])(p, n, v)); }
static int p_getu(const NgxParam *p, const char *n, unsigned *v) { return NGX_OK(((GetU)p->vt->slot[G_U32])(p, n, v)); }
static int p_getf(const NgxParam *p, const char *n, float *v) { return NGX_OK(((GetF)p->vt->slot[G_F])(p, n, v)); }
static int p_getp(const NgxParam *p, const char *n, void **v) { return NGX_OK(((GetP)p->vt->slot[G_P])(p, n, v)); }

// NGX is brought up only when the upscaler at start may be DLSS (upscaling = dlss or auto): its device extensions can
// only be asked for at device creation, and loading the driver's NGX core on every NVIDIA machine would cost start time
// and change the device without DLSS. So DLSS chosen later applies after a restart; switched off, nothing of it runs.
int vk_dlaa_wanted(void) {
    static int wanted = -1;
    if (wanted < 0) {
        const char *up = cfg_upscaling();
        wanted = !strcmp(up, "dlss") || !strcmp(up, "auto");
    }
    return wanted;
}

// Loads the NGX core. It must run after vkCreateInstance: the Vulkan loader has then loaded NVIDIA's ICD
// (nvoglv64.dll), and the core is the _nvngx.dll in that module's folder, so the two always come from the same driver.
// Recent drivers no longer write the NGXCore registry key, which is why it only comes second.
static int core_load(void) {
    if (N.tried) return N.loaded;
    N.tried = 1;
#ifndef _WIN32
    rt_log("dlaa: NGX is loaded on Windows only: DLAA off");
    return 0;
#else
    wchar_t path[MAX_PATH * 2 + 32] = {0};
    HMODULE icd = GetModuleHandleW(L"nvoglv64.dll");
    DWORD len = icd ? GetModuleFileNameW(icd, path, MAX_PATH * 2) : 0;
    while (len && path[len - 1] != L'\\') len--;
    if (len) {
        wcscpy(path + len, L"_nvngx.dll");
        N.core = LoadLibraryW(path);
    }
    if (!N.core) {
        DWORD n = MAX_PATH * 2 * sizeof(wchar_t);
        if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", L"FullPath",
                         RRF_RT_REG_SZ, NULL, path, &n) == ERROR_SUCCESS) {
            wcscat(path, L"\\_nvngx.dll");
            N.core = LoadLibraryW(path);
        }
    }
    if (!N.core) {
        rt_log("dlaa: no NGX core (_nvngx.dll) beside NVIDIA's Vulkan driver%s or through the NGXCore registry key: DLAA off",
               icd ? "" : " (not loaded: no NVIDIA GPU?)");
        return 0;
    }
#define FN(f, name) N.f = (void *)GetProcAddress((HMODULE)N.core, name)
    FN(init, "NVSDK_NGX_VULKAN_Init_Ext2");
    FN(req_ext, "NVSDK_NGX_VULKAN_RequiredExtensions");
    FN(caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    FN(alloc_params, "NVSDK_NGX_VULKAN_AllocateParameters");
    FN(destroy_params, "NVSDK_NGX_VULKAN_DestroyParameters");
    FN(create, "NVSDK_NGX_VULKAN_CreateFeature");
    FN(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
    FN(release, "NVSDK_NGX_VULKAN_ReleaseFeature");
    FN(shutdown, "NVSDK_NGX_VULKAN_Shutdown1");
#undef FN
    if (!N.init || !N.req_ext || !N.caps || !N.alloc_params || !N.create || !N.evaluate || !N.release) {
        rt_log("dlaa: the NGX core %ls lacks Vulkan entry points: DLAA off", path);
        return 0;
    }
    N.loaded = 1;
    rt_log("dlaa: NGX core %ls", path);
    return 1;
#endif
}

static int listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return 1;
    return 0;
}

void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n,
                         uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    // The instance already exists (core_load needs it), so NGX's instance extensions cannot join it. The only one it
    // asks for is part of Vulkan 1.1 and our instance is 1.3; anything else is logged, since DLSS may then fail.
    for (unsigned i = 0; i < ni; i++)
        if (strcmp(ie[i], VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            rt_log("dlaa: NGX asks for the instance extension %s, which is not enabled", ie[i]);
    for (unsigned i = 0; i < nd; i++) {
        int ok = 0;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) rt_log("dlaa: the device lacks %s, which NGX asks for", de[i]);
        else if (!listed(ext, *n, de[i]) && *n < max) ext[(*n)++] = de[i];
    }
}

__attribute__((unused))   // called from vk_dlaa_init on Windows only
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

void vk_dlaa_init(void) {
    if (!vk_dlaa_wanted() || !core_load()) return;
#ifdef _WIN32   // core_load loads nothing elsewhere
    wchar_t wdir[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, wdir, MAX_PATH);
    while (len && wdir[len - 1] != L'\\') len--;
    wdir[len] = 0;
    // 231313132: the application id NVIDIA's DLSS samples use for builds without an NVIDIA-issued id
    const VkCtx *c = vk_ctx();
    NgxResult r = N.init(231313132ull, wdir, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                         vkGetDeviceProcAddr, NGX_VERSION_API, NULL);
    if (!NGX_OK(r)) {
        rt_log("dlaa: NGX init failed (%08x): DLAA off", (unsigned)r);
        return;
    }
    N.inited = 1;
    if (!vtable_ok()) return;
    NgxParam *caps = NULL;
    int avail = 0, driver = 0;
    if (NGX_OK(N.caps(&caps)) && caps) {
        p_geti(caps, "SuperSampling.Available", &avail);
        p_geti(caps, "SuperSampling.NeedsUpdatedDriver", &driver);
    }
    N.available = avail != 0;
    rt_log("dlaa: NGX up; DLSS %s%s", avail ? "available" : "NOT available (is nvngx_dlss.dll next to astralchain.exe?)",
           driver ? "; the driver needs an update" : "");
#endif
}

// ---------------------------------------------------------------- where the upscaler runs in Astral Chain's frame
// Measured (AC_DUMP in play at 2560x1440, frame 1400 of build/replay_ingame.txt; the same structure outside the police
// HQ and on the night highway). The game renders forward, with no G-buffer:
//   a depth-only prepass writes the scene depth (D32F_S8, at the scene's size);
//   the opaque draws go into an R11G11B10F light target with that depth; a full-screen copy of the light target and a
//   half-size pass from it follow (what the transparents refract), then the shadow map (a depth-only pass of its own
//   size), more opaque draws and the transparents, all into the light target with the scene depth;
//   the bloom chain then reads the light target at half size and on down to 1/32; a composite (tonemap and grade) writes
//   another R11G11B10F target with the scene depth attached; SMAA (edges, weights, blend) makes the RGBA8 picture the
//   HUD is drawn over.
// The light target alternates between two images from one frame to the next. Detection, per pass (vk_dlaa_pass):
//   a pass with no colour target and a D32F(S8) depth at least 640 wide is a depth prepass (the shadow map passes look
//   the same; they fall between light-target passes and cannot arm, having no colour target);
//   after one, a pass with one R11G11B10F or RGBA16F target and a depth of its own size arms the frame on that pair (a
//   larger pair may take over, a smaller never: the half-size pass is not one, it has no depth);
//   the first pass at most a quarter as wide as the armed target (the bloom's third level) ends the scene: the motion
//   vectors are made there, and at the scene's own size the upscaler runs there too, in place on the light target. The
//   bloom's first two levels read the scene unresolved, which their blur hides; the half-size refraction pass inside the
//   scene rules out ending at half size.
// Upscaling (a scene smaller than the game resolution: resolution.c, the scene patch set). Measured (pull request #59's
// frame dumps at a 960x540 scene and a 1920x1080 output, the night highway save in a cutscene and in play): the whole
// scene runs at its size, the depth prepass, the light target, the bloom chain and the composite, which writes bloom,
// tonemap and grading (values 0..1, above 1 with HDR output: vk_hdr.c) into another R11G11B10F target of the scene's
// size with the scene depth attached. SMAA's edge and weight passes follow at that size; then SMAA's blending pass is
// the game's only scaling: one full-screen draw, no depth, from the composite into the RGBA8 target of the output's size
// that the HUD is drawn over. So the motion vectors are still made where the scene ends, and the upscaler runs at the
// begin of the blending pass, on the composite, into an image of the output's size; from there to the frame's end,
// passes wider than the composite that sample it sample that image in its place (vk_dlaa_swap_img, vk_exec_draw.c
// tex_info): the blending draw copies the upscaled picture, and the HUD is never jittered nor upscaled. SMAA's edge and
// weight passes are skipped meanwhile (vk_dlaa_skip_draw). (As in the TotK Recomp, same author, MIT.)
// The TotK heuristics this file had before (a three-target G-buffer, a D32F depth, the camera in c4) never found
// Astral Chain's.
static struct {
    VkExecTarget color, depth;   // the armed pair, valid in the frame it was armed
    int armed, done, gbuf;       // this frame: armed on the light target, the upscaler ran (or failed), a depth prepass
    int resolved;                // this frame: the upscaler's picture replaces the scene's, or will at the scaling pass
                                 // (SMAA is skipped then)
    int ended;                   // this frame: the scene is drawn (jitter over)
    int scaled;                  // this frame's scene is smaller than the game resolution: upscaled at the scaling pass
    VkExecTarget comp;           // upscaling: this frame's composite (the scene after bloom, tonemap and grading)
    int comp_ok;
    int sharp_done;              // this frame's upscaled scene was sharpened before its HUD (sharpen)
    int backend;                 // BK_*, chosen at each present (vk_dlaa_frame)
    struct Size {                // the work images per scene size and output size (handheld and docked differ), and
        uint32_t w, h, ow, oh;   // DLSS's feature
        VkImage out, mv, sharp;  // out: the upscaler's picture (output size); sharp: out after RCAS, made when needed
        VmaAllocation out_mem, mv_mem, sharp_mem;
        VkImageView out_view, mv_view, sharp_view;
        NgxHandle *feat;
        uint64_t used;           // evals count when it last ran
    } sizes[3], *last, *cur;     // last: the one the previous frame ran with; cur: this frame's (motion, then resolve)
    VkImage last_color[2];       // the light targets of the last two frames: either keeps the history
    NgxParam *params;
    int failed, reset;           // DLSS failed in this run (XeSS and FSR 3 keep their own) / the next one starts anew
    uint32_t gap;                // presents in a row without the AA point
    uint64_t evals, frames_missed;
    // jitter (vk_dlaa_viewport)
    VkImage jdepth;              // the scene depth of the last frame the AA ran on: passes drawing with it are jittered
    float jx, jy;                // the offset the next scene draws take, render pixels (window x right, y down)
    float ux, uy;                // the offset this frame's scene draws took (the AA and the motion vectors get these)
    uint32_t jframe;
    int jit_ok, jit_used, hold;  // jitter allowed this frame / a draw used it / frames left without it
    uint64_t holds;
    // camera (vk_dlaa_draw): the c3 blocks the scene draws bind this frame, copied when first seen, with their counts
    struct Cam {
        uint64_t addr;
        uint32_t n;
        int have;                // block holds the 256 bytes
        float block[64];
    } cams[4];
    double cam_prev[16];         // last frame's view-projection
    double view_prev[16], focal_prev;
    int cam_prev_ok, view_ok;
    // motion vectors (mv_pass)
    VkDescriptorSetLayout mv_dsl;
    VkPipelineLayout mv_layout;
    VkPipeline mv_pipe;
    VkSampler mv_samp;
    VkBuffer mvm;                // MVM_SLOTS reprojection matrices, host-visible
    VmaAllocation mvm_mem;
    uint8_t *mvm_map;
    uint32_t mvm_stride, mvm_slot;
    int mv_ok;                   // -1 not built, 0 failed, 1 ready
    int blit_ok;                 // -1 not checked; the light target's format takes a blit
    uint64_t mv_frames, mv_none, cuts;
    uint64_t why[4];             // history resets: scene size changed, no camera, scene target changed, gap
    uint64_t last_eval_ns;       // plat_now_ns of the previous evaluation (DLSS's FrameTimeDeltaInMsec, FSR 3's dt)
    float proj[16];              // the last camera's projection (rows 4-7 of its block): FSR 3's near, far and fov
    int proj_ok;
} D = {.mv_ok = -1, .blit_ok = -1};

VkImage vk_dlaa_swap_img;
VkImageView vk_dlaa_swap_view;
uint32_t vk_dlaa_swap_w;
VkImage vk_dlaa_swap_pic;   // the image behind vk_dlaa_swap_view (vk_hdr.c records it as the frame's scene)

enum { BK_NONE, BK_DLSS, BK_XESS, BK_FSR3 };

static int dl_active(void) {
    return D.backend == BK_DLSS   ? N.available && !D.failed
           : D.backend == BK_XESS ? vk_xess_available()
           : D.backend == BK_FSR3 ? vk_fsr3_available()
                                  : 0;
}

// The backend for the setting (cfg.c cfg_upscaling, read at each present: the menu switches it live), among those that
// can run in this run: DLSS where NGX came up (loaded at start for dlss and auto), XeSS where libxess.dll came up
// (loaded at start for xess), FSR 3 wherever its pipelines can be made (it is built in). auto is DLSS where it can run,
// else FSR 3. While the scene is reduced (resolution.c) something must scale it: a choice that cannot run (off, a
// library not loaded at start, a backend that failed in this run) leaves the first that can of DLSS, XeSS and FSR 3, so
// the game's plain scaling never shows a scene at 33..67 % of the output; the menu says that the choice waits for a
// restart. can() makes nothing: FSR 3's pipelines are made when it is picked (dl_active).
static int can(int bk) {
    return bk == BK_DLSS ? N.available && !D.failed : bk == BK_XESS ? vk_xess_available() : bk == BK_FSR3 ? !vk_fsr3_failed() : 0;
}

static int chosen(void) {
    const char *up = cfg_upscaling();
    return !strcmp(up, "dlss") ? BK_DLSS
           : !strcmp(up, "xess") ? BK_XESS
           : !strcmp(up, "fsr3") ? BK_FSR3
           : !strcmp(up, "auto") ? (can(BK_DLSS) ? BK_DLSS : BK_FSR3)
                                 : BK_NONE;
}

static int pick(void) {
    int bk = chosen();
    if (bk != BK_NONE && can(bk)) return bk;
    if (!res_scene_reduced()) return BK_NONE;
    for (int b = BK_DLSS; b <= BK_FSR3; b++)
        if (can(b)) return b;
    return BK_NONE;
}

// resolution.c res_settle, after the device exists and before the game runs: the upscaler the setting asks for can run
// here (NGX answered, libxess.dll came up on this GPU, FSR 3's pipelines can be made).
int vk_dlaa_upscaler_ok(void) {
    int bk = chosen();
    if (bk == BK_FSR3) return vk_pipeline_push_fn() ? vk_fsr3_available() : !vk_fsr3_failed();
    return bk != BK_NONE && can(bk);
}

static void dl_fail(const char *what, NgxResult r) {
    rt_log("dlaa: %s failed (%08x): DLSS off for this run", what, (unsigned)r);
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
    if (vmaCreateImage(c->vma, &ii, &ai, img, mem, NULL) != VK_SUCCESS) return *img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, view) == VK_SUCCESS;
}

// New images into GENERAL, the layout every image of the back end is kept in.
static void to_general(VkCommandBuffer cb, const VkImage *im, uint32_t n) {
    VkImageMemoryBarrier2 ib[3];
    for (uint32_t i = 0; i < n; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
}

// DLSS features whose size was freed while submissions that evaluate them may still be on the GPU: each is released
// once the timeline has passed the newest value issued when it was retired (features_sweep, at each present). The
// replay records one submission at a time and the present runs on the same thread in queue order, so nothing recorded
// later can use a retired feature. (After the TotK Recomp's vk_dlaa.c retire_feature.)
static struct { NgxHandle *feat; uint64_t after; } dead_feat[16];

static void features_sweep(int all) {
    uint64_t done = vk_sync_fence_done();
    for (int i = 0; i < 16; i++)
        if (dead_feat[i].feat && (all || dead_feat[i].after <= done)) N.release(dead_feat[i].feat), dead_feat[i].feat = NULL;
}

static void retire_feature(NgxHandle *f) {
    for (int i = 0; i < 16; i++)
        if (!dead_feat[i].feat) {
            dead_feat[i].feat = f, dead_feat[i].after = vk_sync_issued();
            return;
        }
    // no room (sixteen sizes retired within a few frames): the slow way, wait for the GPU and release everything
    vk_queue_lock();   // vkDeviceWaitIdle needs every queue externally synchronised; other threads submit too
    vkDeviceWaitIdle(vk_ctx()->device);
    vk_queue_unlock();
    features_sweep(1);
    N.release(f);
}

// An evicted size, or the AA off for a while (vk_dlaa_frame). The images may still be in flight, so they go to the
// deferred destruction of vk_memory.c and the feature to dead_feat: no wait for the device in the middle of a frame.
// The HDR present may hold the upscaler's picture as the frame's scene (vk_hdr_draw): it lets go of it.
static void free_size(struct Size *e) {
    if (e->out) vk_hdr_image_gone(e->out);
    if (e->sharp) vk_hdr_image_gone(e->sharp);
    vk_defer_view(e->out_view), vk_defer_view(e->mv_view), vk_defer_view(e->sharp_view);
    vk_defer_image(e->out, e->out_mem), vk_defer_image(e->mv, e->mv_mem), vk_defer_image(e->sharp, e->sharp_mem);
    if (e->feat) retire_feature(e->feat);
    if (D.last == e) D.last = NULL;
    if (D.cur == e) D.cur = NULL;
    memset(e, 0, sizeof *e);
}

// The entry for a scene size and output size: its output (RGBA16F at the output's size, written by the upscaler, then
// blitted into the light target in place or sampled by the scaling pass) and motion vectors (RG16F at the scene's
// size, render pixels), both kept in GENERAL like every image of the back end. Handheld and docked draw the scene at
// different sizes; each keeps its entry, the least recently used goes for a new one.
static struct Size *size_for(VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    struct Size *e = NULL, *lru = &D.sizes[0];
    for (int i = 0; i < 3; i++) {
        if (D.sizes[i].w == w && D.sizes[i].h == h && D.sizes[i].ow == ow && D.sizes[i].oh == oh) return &D.sizes[i];
        if (!D.sizes[i].out && !e) e = &D.sizes[i];
        if (D.sizes[i].used < lru->used) lru = &D.sizes[i];
    }
    if (!e) free_size(e = lru);
    VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // the motion image is also the per-object pass's colour target (vk_mvobj_execute)
    if (!image2d(ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, u, &e->out, &e->out_mem, &e->out_view) ||
        !image2d(w, h, VK_FORMAT_R16G16_SFLOAT, u | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &e->mv, &e->mv_mem,
                 &e->mv_view)) {
        free_size(e);   // what was made of the pair goes, or it would stay in the entry for the rest of the run
        return NULL;
    }
    VkImage im[2] = {e->out, e->mv};
    to_general(cb, im, 2);
    e->w = w, e->h = h, e->ow = ow, e->oh = oh;
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

// ---------------------------------------------------------------- motion vectors
// The camera's motion first (runtime/shaders/dlaa_mv.comp). The camera block is the vertex stage's c3 of the scene's draws,
// 256 bytes of row-major vec4 rows: 0-3 the view, 4-7 the projection, 8-11 their product the view-projection, 12-15 the
// inverse view (measured: c3 of every scene pass in the dump above; row 8 is the projection's first row times the view).
// The reprojection from this frame's clip space to last frame's, last frame's view-projection times the inverse of this
// frame's, is made on the CPU in double precision and given to the pass as one matrix: a still camera gives the identity
// and exactly zero motion, where a float inverse per pixel left a fraction of a pixel of noise on distant surfaces that
// the AA followed into a shimmer. Characters and anything else that moves on its own are then drawn over those vectors
// with their own (vk_exec_draw.c, per-object motion; vk_mvprev.c keeps last frame's blocks): with the camera's motion
// alone XeSS left a trail behind every moving enemy.
#define CAM_BYTES 256u   // vec4 rows 0-15: view, projection, view-projection, inverse view
#define MVM_SLOTS 16u    // frames the GPU may still be reading the reprojection matrices of
#define CUT_DEG 30.0     // a camera cut: the view turned more than this in one frame (degrees)
#define CUT_DIST 5.0     // or moved more than this (units, metres)
#define CUT_ZOOM 2.0     // or its focal length changed by more than this many stops (a factor of 4)
#define GAP_PRESENTS 6u  // presents in a row without the AA point (100 ms at 60 fps) that start a new history
#define OFF_PRESENTS 600u   // presents without DLSS, or without any temporal AA, before its memory is given back

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
    rt_log("dlaa: the motion-vector pass could not be set up: the AA runs with zero motion");
    return 0;
}

// 4x4 inverse by Gauss-Jordan with partial pivoting, row-major, in double precision. 0: singular.
static int inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-300) return 0;
        for (int k = 0; k < 8; k++) {
            double t = a[col][k];
            a[col][k] = a[piv][k], a[piv][k] = t;
        }
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

// This frame's camera: the block most scene draws bound (a few draws into the scene depth bind another camera block;
// in the night highway save the last one seen alternated between two cameras 160 units apart). NULL: none seen. *addr:
// its guest address, which the per-object pass keeps its draws to.
static const float *frame_camera(uint64_t *addr) {
    const struct Cam *best = NULL;
    for (int i = 0; i < 4; i++)
        if (D.cams[i].have && (!best || D.cams[i].n > best->n)) best = &D.cams[i];
    *addr = best ? best->addr : 0;
    return best ? best->block : NULL;
}

// The pass, recorded through the recorder like the replay's own commands. 1: vectors written; 0: no camera this frame
// or none last frame; -1: a camera cut. On 0 and -1 the vectors are left for the caller to clear (zero motion) and the
// history starts anew. On 1 the objects that moved on their own are drawn over the camera's vectors (vk_mvobj_execute).
static int mv_pass(VkExec *ex, const struct Size *e, VkImageView zv, float jx, float jy) {
    VkCommandBuffer cb = ex->cb;
    uint32_t w = e->w, h = e->h;
    uint64_t cam_addr = 0;
    const float *cam = mv_init() ? frame_camera(&cam_addr) : NULL;
    if (!cam) {
        D.mv_none++;
        D.cam_prev_ok = 0, D.view_ok = 0;
        return 0;
    }
    const float *vw = cam, *pj = cam + 16, *vp = cam + 32;
    memcpy(D.proj, pj, sizeof D.proj), D.proj_ok = 1;
    // its y scale, a diagonal element and so the same in either matrix order; the jitter moves the viewport, not this
    double focal = isfinite(pj[5]) ? fabs((double)pj[5]) : 0;
    double cur_vp[16], ci[16], m[16];
    for (int i = 0; i < 16; i++) cur_vp[i] = vp[i];
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
    // A camera cut (a cutscene's next shot, a respawn): the view turned more than CUT_DEG, moved more than CUT_DIST
    // units or changed its focal length by more than CUT_ZOOM stops in one frame, which no continuous camera does. The
    // three count together, each as its share of its limit, so a cut to a nearby angle or a closer shot along the same
    // line (a little of each) is one too. The view and projection are compared, not the picture: a cutscene's fast zoom
    // moves the picture's corners by half the screen a frame for seconds, which is why the zoom's limit is wide. The
    // history is then of another shot, so the AA starts a new one from zero motion.
    int cut = 0;
    if (have_m && D.view_ok) {
        double tr = 0, dist2 = 0;   // trace of R_prev R_cur^T; camera positions -R^T t
        double pc[3] = {0, 0, 0}, pp[3] = {0, 0, 0};
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 3; k++) tr += D.view_prev[r * 4 + k] * vw[r * 4 + k];
        for (int k = 0; k < 3; k++)
            for (int r = 0; r < 3; r++)
                pc[k] -= vw[r * 4 + k] * vw[r * 4 + 3], pp[k] -= D.view_prev[r * 4 + k] * D.view_prev[r * 4 + 3];
        for (int k = 0; k < 3; k++) dist2 += (pc[k] - pp[k]) * (pc[k] - pp[k]);
        double c = (tr - 1) / 2, deg = acos(c > 1 ? 1 : c < -1 ? -1 : c) * 57.29577951308232;
        double zoom = focal > 0 && D.focal_prev > 0 ? fabs(log2(focal / D.focal_prev)) : 0;
        double score = deg / CUT_DEG + sqrt(dist2) / CUT_DIST + zoom / CUT_ZOOM;
        cut = score > 1.0;
        if (cut && ++D.cuts <= 40)
            rt_log("dlaa: camera cut at frame %llu (turned %.1f degrees, moved %.2f, zoomed %.2f stops; score %.2f): new "
                   "history", (unsigned long long)D.evals, deg, sqrt(dist2), zoom, score);
    }
    // kept on every way out, or the first frame after a gap is measured against the view before the gap
    for (int i = 0; i < 16; i++) D.view_prev[i] = vw[i];
    D.view_ok = 1;
    D.focal_prev = focal;
    if (!have_m) return 0;
    if (cut) return -1;
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
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_layout, 0, 3, wr);
    float pc[8] = {(float)w, (float)h, jx, jy, nvn_depth_minus_w() ? 1.0f : 0.0f, 0, 0, 0};
    vkCmdPushConstants(cb, D.mv_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    // what moves on its own over the camera's motion (vk_exec_draw.c, per-object motion); it leaves the replay's
    // bindings for the next draw to set again
    if (vk_mvobj_execute(ex, cam_addr, e->mv_view, &D.depth, w, h)) full_barrier(cb);
    D.mv_frames++;
    return 1;
}

// ---------------------------------------------------------------- jitter
// The scene's draws (every pass drawing with the scene depth, until the AA point) have their viewport moved by a
// sub-pixel Halton(2,3) offset, which shifts what they rasterise as a jittered projection would; the camera block keeps
// its unjittered matrices. Jitter only follows a frame the AA ran on: a jittered frame that is not resolved shakes on
// screen, so a missed AA point stops the jitter for a second.
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

// The same test as vk_dlaa_viewport, for the per-object pass, whose draws take the original's viewport again and must
// take the same jitter off gl_FragCoord.
void vk_dlaa_jitter(const VkExec *ex, float j[2]) {
    int on = D.jit_ok && !D.ended && ex->depth.image && ex->depth.image == D.jdepth;
    j[0] = on ? D.jx : 0.0f, j[1] = on ? D.jy : 0.0f;
}

// A backend runs and the AA point can use the motion image (otherwise the per-object listing would have no consumer).
int vk_dlaa_available(void) { return dl_active() && D.mv_ok != 0 && D.blit_ok != 0; }

int vk_dlaa_scene_draw(const VkExec *ex) {
    const VkImage z = ex->depth.image;
    return dl_active() && !D.ended && z && ((D.armed && z == D.depth.image) || (D.jdepth && z == D.jdepth));
}

// The camera block is the vertex stage's c3 (256 bytes) of a draw with the scene depth: the armed pass's, or before
// arming (the depth prepass) the depth of the last frame the AA ran on. The shadow map passes bind another camera with
// another depth target, the HUD has no scene depth. Each block is copied when first seen this frame, from the uniform
// snapshot of the submission that draws with it (what the GPU reads), so the AA point may come in a later submission.
void vk_dlaa_draw(VkExec *ex) {
    if (!dl_active() || D.ended) return;
    const p_bind_buf *b = &ex->ubo[0][0];   // constant bank c3: the replay keeps user banks from c3 on
    const VkExecTarget *z = &ex->depth;
    if (!(b->size == CAM_BYTES && b->addr && z->image &&
          ((D.armed && z->image == D.depth.image) || (D.jdepth && z->image == D.jdepth))))
        return;
    int i = 0, low = 0;
    for (; i < 4 && D.cams[i].addr && D.cams[i].addr != b->addr; i++)
        if (D.cams[i].n < D.cams[low].n) low = i;
    if (i == 4) i = low, D.cams[i].addr = 0;   // full: the rarest makes room
    struct Cam *c = &D.cams[i];
    if (c->addr != b->addr) {
        if (!vk_usnap_copy(b->addr, CAM_BYTES, c->block))   // no snapshot (AC_UBO_SNAPSHOT=0): guest memory
            memcpy(c->block, (const void *)(uintptr_t)b->addr, CAM_BYTES);
        c->addr = b->addr, c->n = 0;
        c->have = 1;
    }
    c->n++;
}

// The jitter moves on right after the upscaler ran, not at the present: the replay can record the next frame's scene
// before the present of this one is handled, and a step taken there landed in the middle of a frame. The sequence is as
// long as NVIDIA's DLSS guide asks for the upscaling ratio, 8 x (output / scene)^2 phases (18 in Quality, 24 in
// Balanced, 32 in Performance, 72 in Ultra Performance), and 16 at least (the scene at the output's size). XeSS's and
// AMD's guides ask for about as many.
static uint32_t jitter_phases(uint32_t w, uint32_t ow) {
    double r = w ? (double)ow / w : 1.0;
    uint32_t phases = (uint32_t)ceil(8.0 * r * r - 1e-6);
    return phases < 16 ? 16 : phases > 72 ? 72 : phases;
}

static void next_jitter(uint32_t w, uint32_t ow) {
    static int nj = -1;
    if (nj < 0) nj = !rt_knob("DLAA_JITTER", "1");   // AC_DLAA_JITTER=0: no jitter (test)
    uint32_t i = D.jframe++ % jitter_phases(w, ow) + 1;
    D.jx = halton(i, 2) - 0.5f, D.jy = halton(i, 3) - 0.5f;
    D.jit_ok = !D.hold && !nj;
}

// The light target takes the AA's RGBA16F picture with a 1:1 blit; a format the device cannot blit into keeps the AA off.
static int blit_ok(VkFormat f) {
    if (D.blit_ok < 0) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &fp);
        D.blit_ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
        if (!D.blit_ok) rt_log("dlaa: the device cannot blit into the scene's format %d: the temporal AA stays off", (int)f);
    }
    return D.blit_ok;
}

// The backend's name, at the scene's own size (native) or upscaling.
static const char *backend_name(int b, int up) {
    return b == BK_DLSS   ? (up ? "DLSS" : "DLAA")
           : b == BK_XESS ? (up ? "XeSS" : "XeSS Native AA")
           : b == BK_FSR3 ? (up ? "FSR 3" : "FSR 3 Native AA")
                          : "off";
}

// The scene camera's near and far plane distance and vertical field of view, for FSR 3's depth to view distance: the
// projection (kept by mv_pass) taken back from the depth buffer's nearest and farthest value at the centre of the view,
// as the depth buffer holds them (0..1, or -1..1 as z_ndc with NEAR_IS_MINUS_W). Without a camera, a plausible one (0.1
// to 1000 units, 60 degrees): FSR then only misjudges distances in its disocclusion test.
static void camera_planes(float *near_z, float *far_z, float *fov_y) {
    *near_z = 0.1f, *far_z = 1000.0f, *fov_y = 1.0471976f;
    double m[16], inv[16];
    for (int i = 0; i < 16; i++) m[i] = D.proj[i];
    if (!D.proj_ok || !isfinite(m[5]) || fabs(m[5]) < 1e-6 || !inv4(m, inv)) return;
    double d[2], zn[2] = {nvn_depth_minus_w() ? -1.0 : 0.0, 1.0};
    for (int k = 0; k < 2; k++) {   // the view-space point of ndc (0, 0, z): row 2 over row 3 of the inverse
        double vz = inv[2 * 4 + 2] * zn[k] + inv[2 * 4 + 3], vw = inv[3 * 4 + 2] * zn[k] + inv[3 * 4 + 3];
        d[k] = fabs(vw) > 1e-12 ? fabs(vz / vw) : 1e9;
    }
    if (!(d[0] > 0) || !isfinite(d[0])) return;
    *near_z = (float)d[0];
    *far_z = (float)(d[1] > d[0] && d[1] < 1e9 ? d[1] : d[0] * 1e6);   // an infinite projection: far enough
    *fov_y = (float)(2.0 * atan(1.0 / fabs(m[5])));
}

// The quality setting DLSS is created with, from the scene's share of the output width
// (NVSDK_NGX_PerfQuality_Value: 0 performance, 1 balanced, 2 quality, 3 ultra performance, 5 DLAA). The limits lie
// halfway between the modes' shares (67, 58, 50 and 33 %), because resolution.c moves a scene to the nearest size on
// its 32x18 grid, which can land a share a little off its mode's (Quality at 2560x1440 is 1696x954, 66.25 %).
static int perf_quality(uint32_t w, uint32_t ow) {
    if (w >= ow) return NGX_PERFQ_DLAA;
    double k = (double)w / ow;
    return k >= 0.625 ? 2 : k >= 0.54 ? 1 : k >= 0.42 ? 0 : 3;
}

// ---------------------------------------------------------------- sharpening before the HUD
// One RCAS (vk_fsr.c vk_fsr_rcas, at sharpness) on the upscaler's picture, after the game's tonemap and grading, the
// range RCAS is made for, and before the game draws its HUD, so the HUD stays as the game drew it. Only where the
// present would otherwise sharpen the whole picture at 1:1: the window copies the game's image (the last present did
// not scale it, vk_fsr_present_scales), the window scaler is FSR (bilinear sharpens nothing, before or after), and no
// HDR output (RCAS's limiter works on 0..1). The frame then tells the present to copy it (vk_dlaa_frame's return,
// vk_fsr_presharp_set); frames without an upscaled scene (the title, loads, native modes) are sharpened at the present
// as before. *img, *view: the picture the scaling pass is to read, replaced by the sharpened one.
static void sharpen(VkCommandBuffer cb, struct Size *e, VkImage *img, VkImageView *view) {
    if (vk_hdr_wanted() || vk_fsr_present_scales() || !vk_fsr_rcas_on()) return;
    if (!e->sharp) {
        if (!image2d(e->ow, e->oh, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     &e->sharp, &e->sharp_mem, &e->sharp_view)) {
            if (e->sharp) vmaDestroyImage(vk_ctx()->vma, e->sharp, e->sharp_mem);
            e->sharp = VK_NULL_HANDLE, e->sharp_view = VK_NULL_HANDLE;
            return;
        }
        to_general(cb, &e->sharp, 1);
    }
    if (!vk_fsr_rcas(cb, e->out_view, e->sharp_view, e->ow, e->oh)) return;
    full_barrier(cb);
    D.sharp_done = 1;
    *img = e->sharp, *view = e->sharp_view;
}

// ---------------------------------------------------------------- the upscaler, in two halves
// motion, where the scene ends (the bloom's third level): this frame's work images, the history checks and the motion
// vectors (the camera's, then the objects' over them, in the submission that listed them). ow x oh: the output, the
// scene's own size in place. 0: the frame stays unresolved.
static int motion(VkExec *ex, uint32_t ow, uint32_t oh) {
    const VkExecTarget *t = &D.color, *z = &D.depth;
    uint32_t w = t->width, h = t->height;
    VkCommandBuffer cb = ex->cb;
    if (ow == w && oh == h && !blit_ok(t->format)) return 0;
    vk_exec_barrier(ex);   // what drew the scene is done (no pass is open here)
    struct Size *e = size_for(cb, w, h, ow, oh);
    if (!e) {
        rt_log("dlaa: work image creation failed (%ux%u to %ux%u)", w, h, ow, oh);
        return 0;
    }
    if (e != D.last) D.reset = 1, D.why[0]++;   // another size's history is from another moment
    if (D.gap >= GAP_PRESENTS) D.reset = 1, D.why[3]++;
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    float jx = D.jit_used ? D.ux : 0.0f, jy = D.jit_used ? D.uy : 0.0f;
    static int no_mv = -1;
    if (no_mv < 0) no_mv = !rt_knob("DLAA_MV", "1");   // AC_DLAA_MV=0: zero motion (test)
    int mv = no_mv ? 0 : mv_pass(ex, e, zv, jx, jy);
    if (mv <= 0) {
        vk_mvobj_drop();   // the per-object pass needs the camera's vectors under it
        VkClearColorValue zero = {{0, 0, 0, 0}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, e->mv, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        full_barrier(cb);
        if (!no_mv) D.reset = 1, D.why[1] += mv == 0;   // no camera or a cut: the history cannot be moved along
    }
    // the light target alternates between two images: one seen in either of the last two frames keeps the history
    if (t->image != D.last_color[0] && t->image != D.last_color[1]) D.reset = 1, D.why[2]++;
    if (t->image != D.last_color[0]) D.last_color[1] = D.last_color[0], D.last_color[0] = t->image;
    vk_exec_draw_new_cb();   // the motion passes bound pipelines and sets of their own: the next draw binds everything
    ex->dirty = VK_EXEC_DIRTY_ALL;
    D.cur = e;
    return 1;
}

// resolve: the backend on src (the light target in place, or the composite at the game's scaling pass) into the size's
// output image; in place, a blit of it back over the light target, which the bloom, the composite and the HUD then read
// and draw over; upscaling, the swap that has the scaling draw read it instead of the composite (vk_dlaa_swap_img). 0:
// the frame keeps the unresolved scene (or the game's plain scaling of it).
static int resolve(VkExec *ex, const VkExecTarget *src) {
    struct Size *e = D.cur;
    const VkExecTarget *t = src, *z = &D.depth;
    if (!e) return 0;
    uint32_t w = e->w, h = e->h, ow = e->ow, oh = e->oh;
    int up = ow != w || oh != h;
    // linear scene light (the light target), or the composite with HDR output, which keeps highlights above 1
    int hdr = !up || vk_hdr_wanted();
    VkCommandBuffer cb = ex->cb;
    if (up) vk_exec_barrier(ex);   // the composite and SMAA's passes are done (no pass is open here)
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    VkImageView cv = vk_exec_target_view(t->img, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                                         t->base_layer, 1, 0);
    float jx = D.jit_used ? D.ux : 0.0f, jy = D.jit_used ? D.uy : 0.0f;
    // The time since the previous evaluation, which DLSS and FSR 3 weigh their history by (33 ms at 30 fps); 60 fps
    // before the first and after a gap (a pause, a load), where no history carries over anyway.
    uint64_t now = plat_now_ns();
    double dt = D.last_eval_ns ? (double)(now - D.last_eval_ns) / 1e6 : 0;
    D.last_eval_ns = now;
    float dt_ms = dt > 0 && dt < 100 ? (float)dt : 16.67f;
    // NGX and XeSS record into the command buffer themselves: the recorder thread must have made every call so far.
    // FSR 3's passes go through the recorder like the replay's own commands.
    if (D.backend != BK_FSR3) rec_drain();
    if (D.backend == BK_FSR3) {
        Fsr3Eval f = {.color = cv, .depth = zv, .mv = e->mv_view, .out = e->out_view, .w = w, .h = h, .ow = ow, .oh = oh,
                      .jx = jx, .jy = jy, .phases = jitter_phases(w, ow), .dt_ms = dt_ms, .reset = D.reset, .hdr = hdr};
        camera_planes(&f.near_z, &f.far_z, &f.fov_y);
        if (!vk_fsr3_evaluate(cb, &f)) return 0;   // off for the rest of the run (vk_fsr3.c)
    } else if (D.backend == BK_DLSS) {
        if (!D.params && !NGX_OK(N.alloc_params(&D.params))) return dl_fail("parameter allocation", 0), 0;
        NgxParam *p = D.params;
        if (!e->feat) {
            int q = perf_quality(w, ow);
            p_setu(p, "CreationNodeMask", 1);
            p_setu(p, "VisibilityNodeMask", 1);
            p_setu(p, "Width", w);
            p_setu(p, "Height", h);
            p_setu(p, "OutWidth", ow);
            p_setu(p, "OutHeight", oh);
            p_seti(p, "PerfQualityValue", q);
            // MVLowRes: motion vectors at the render size. Linear scene light: IsHDR | AutoExposure, no exposure
            // texture; the tonemapped composite (0..1) neither.
            p_seti(p, "DLSS.Feature.Create.Flags", (1 << 1) | (hdr ? (1 << 0) | (1 << 6) : 0));
            p_seti(p, "DLSS.Enable.Output.Subrects", 0);
            // The model per mode (NVSDK_NGX_DLSS_Hint_Render_Preset: 11 K, 12 L, 13 M), as NVIDIA recommends for DLSS
            // 4.5 (310.x): K (the transformer) for DLAA, Quality and Balanced, M for Performance, L for Ultra
            // Performance. An older nvngx_dlss.dll without a model takes its own default.
            p_setu(p, "DLSS.Hint.Render.Preset.DLAA", 11);
            p_setu(p, "DLSS.Hint.Render.Preset.UltraQuality", 11);
            p_setu(p, "DLSS.Hint.Render.Preset.Quality", 11);
            p_setu(p, "DLSS.Hint.Render.Preset.Balanced", 11);
            p_setu(p, "DLSS.Hint.Render.Preset.Performance", 13);
            p_setu(p, "DLSS.Hint.Render.Preset.UltraPerformance", 12);
            NgxResult r = N.create(cb, NGX_FEATURE_SUPERSAMPLING, p, &e->feat);
            if (!NGX_OK(r) || !e->feat) return dl_fail("DLSS feature creation", r), 0;
            D.reset = 1;
            rt_log("dlaa: DLSS feature created at %ux%u to %ux%u, quality value %d, preset %c, %s colour (scene format %d, "
                   "depth format %d)", w, h, ow, oh, q, q == 0 ? 'M' : q == 3 ? 'L' : 'K', hdr ? "linear" : "LDR",
                   (int)t->format, (int)z->format);
        }
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
        p_setf(p, "MV.Scale.X", 1.0f);
        p_setf(p, "MV.Scale.Y", 1.0f);
        p_setu(p, "DLSS.Render.Subrect.Dimensions.Width", w);
        p_setu(p, "DLSS.Render.Subrect.Dimensions.Height", h);
        p_setf(p, "DLSS.Pre.Exposure", 1.0f);
        p_setf(p, "DLSS.Exposure.Scale", 1.0f);
        p_setf(p, "FrameTimeDeltaInMsec", dt_ms);
        NgxResult r = N.evaluate(cb, e->feat, p, NULL);
        if (!NGX_OK(r)) return dl_fail("DLSS evaluation", r), 0;
    } else {
        XessFrame f = {.color = {t->image, cv, t->format, t->level, t->base_layer},
                       .depth = {z->image, zv, z->format, z->level, z->base_layer},
                       .mv = {e->mv, e->mv_view, VK_FORMAT_R16G16_SFLOAT, 0, 0},
                       .out = {e->out, e->out_view, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 0},
                       .w = w, .h = h, .ow = ow, .oh = oh, .jx = jx, .jy = jy, .reset = D.reset, .ldr = !hdr};
        int ok = vk_xess_evaluate(cb, &f);
        vk_exec_draw_new_cb();   // XeSS may have bound pipelines and sets of its own, even when it failed
        if (!ok) return 0;        // off for the rest of the run (vk_xess.c); this frame keeps the unresolved scene
    }
    vk_mvprev_swap();   // the objects' blocks of this frame become last frame's for the next
    D.reset = 0;
    D.last = e, e->used = D.evals + 1;
    full_barrier(cb);
    // AC_DLAA_MVDEBUG=1 (test): the motion vectors in the picture's place, x in red and y in green, render pixels as
    // scene light (the bloom, the composite and SMAA's blend still run over them in place): a character moving on its
    // own stands out from the scenery around it. The light target holds no negative values: motion left or up shows
    // black.
    static int mvdebug = -1;
    if (mvdebug < 0) mvdebug = rt_knob("DLAA_MVDEBUG", "0");
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{0, 0, 0}, {(int32_t)(mvdebug ? w : ow), (int32_t)(mvdebug ? h : oh), 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, up ? 0 : t->level, up ? 0 : t->base_layer, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}}};
    if (!up || mvdebug) {   // in place: over the light target; upscaling, only the debug view, into the output image
        vkCmdBlitImage(cb, mvdebug ? e->mv : e->out, VK_IMAGE_LAYOUT_GENERAL, up ? e->out : t->image,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
        full_barrier(cb);
    }
    if (up) {   // the scaling draw and anything wider after it read the upscaled picture in the composite's place
        VkImage pic = e->out;
        VkImageView view = e->out_view;
        if (!mvdebug) sharpen(cb, e, &pic, &view);
        vk_dlaa_swap_img = t->image, vk_dlaa_swap_view = view, vk_dlaa_swap_pic = pic, vk_dlaa_swap_w = w;
    }
    ex->recorded += 4;
    vk_exec_draw_new_cb();   // the backend bound pipelines and descriptor sets of its own: the next draw binds everything
    ex->dirty = VK_EXEC_DIRTY_ALL;
    D.jdepth = z->image;
    D.resolved = 1;
    next_jitter(w, ow);
    if (++D.evals == 1 || D.evals % 600 == 0)
        rt_log("dlaa: %s, %llu frames resolved (scene %ux%u, output %ux%u; %llu frames without the AA point, %llu with "
               "camera motion, %llu without a camera, %llu camera cuts, %llu jitter holds; resets: %llu size, %llu "
               "camera, %llu target, %llu gap)",
               backend_name(D.backend, up), (unsigned long long)D.evals, w, h, ow, oh,
               (unsigned long long)D.frames_missed, (unsigned long long)D.mv_frames, (unsigned long long)D.mv_none,
               (unsigned long long)D.cuts, (unsigned long long)D.holds, (unsigned long long)D.why[0],
               (unsigned long long)D.why[1], (unsigned long long)D.why[2], (unsigned long long)D.why[3]);
    return 1;
}

// A frame the upscaler did not resolve leaves no per-object list or block history behind for the next.
static void unresolved(VkExec *ex) {
    vk_mvobj_drop();
    vk_mvprev_forget(1);
    vk_exec_draw_new_cb();   // the motion pass or the backend may have bound pipelines and sets before it failed
    ex->dirty = VK_EXEC_DIRTY_ALL;
}

static int scene_color(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// A D32F(S8) depth at least 640 wide: the smallest scene resolution.c makes (640x360), and the narrowest of the game's
// own levels (800x720 in handheld).
static int scene_depth(const VkExecTarget *z) {
    return z->image && z->width >= 640 && (z->format == VK_FORMAT_D32_SFLOAT_S8_UINT || z->format == VK_FORMAT_D32_SFLOAT);
}

// The game resolution (resolution.c), the output an upscaled scene goes to.
static void game_size(uint32_t o[2]) {
    unsigned g[2];
    res_game_size(g);
    o[0] = g[0], o[1] = g[1];
}

// This frame's armed scene is the reduced one the scene set made (resolution.c), asked of the target itself: a frame
// at the game's own size (a level the set does not cover) runs in place as without upscaling.
static int target_scaled(const VkExecTarget *c) {
    unsigned s[2];
    res_scene_size(s);
    return res_scene_reduced() && c->width == s[0] && c->height == s[1];
}

void vk_dlaa_pass(VkExec *ex) {
    const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
    // the composite drawn into again after the upscaler ran: its swap would show a stale picture
    if (vk_dlaa_swap_img)
        for (uint32_t i = 0; i < ex->pass_ncolor; i++)
            if (ex->color[i].image == vk_dlaa_swap_img) {
                vk_dlaa_swap_img = VK_NULL_HANDLE, vk_dlaa_swap_view = VK_NULL_HANDLE, vk_dlaa_swap_pic = VK_NULL_HANDLE;
                break;
            }
    if (!dl_active() || D.done) return;
    // the depth prepass (or a shadow map pass: no colour target either, and no harm)
    if (ex->pass_ncolor == 0 && scene_depth(z)) {
        D.gbuf = 1;
        return;
    }
    // a light-target pass; a smaller pair never takes over an armed one
    if (D.gbuf && !D.ended && ex->pass_ncolor == 1 && c->image && scene_color(c->format) && scene_depth(z) &&
        c->width == z->width && c->height == z->height && c->samples <= 1 && !c->is3d &&
        (!D.armed || (c->width >= D.color.width && c->height >= D.color.height))) {
        D.color = *c, D.depth = *z, D.armed = 1;
        return;
    }
    if (!D.armed) return;
    if (!D.ended) {
        if (!ex->pass_w || ex->pass_w * 4 > D.color.width) return;   // the first pass at most a quarter as wide ends it
        D.ended = 1;
        D.scaled = target_scaled(&D.color);
        uint32_t o[2] = {D.color.width, D.color.height};
        if (D.scaled) game_size(o);
        if (!motion(ex, o[0], o[1])) {
            D.done = 1;
            unresolved(ex);
            return;
        }
        if (!D.scaled) {   // in place, at once
            D.done = 1;
            if (!resolve(ex, &D.color)) unresolved(ex);
            return;
        }
        D.resolved = 1;   // SMAA's edges and weights, which come first, are skipped for the upscaler
        return;
    }
    // upscaling: the composite, a pass at the light target's size into one HDR-format target with the scene depth (the
    // latest such pass before the scaling pass)
    if (ex->pass_ncolor == 1 && c->image && scene_color(c->format) && z->image == D.depth.image &&
        c->width == D.color.width && c->height == D.color.height && c->samples <= 1 && !c->is3d) {
        D.comp = *c, D.comp_ok = 1;
        return;
    }
    // the game's scaling pass: one colour target of the game resolution, wider than the scene, no depth (SMAA's blend,
    // which writes the RGBA8 picture the HUD is drawn over). A frame without a composite is left unresolved.
    uint32_t o[2];
    game_size(o);
    if (D.comp_ok && ex->pass_ncolor == 1 && c->image && !z->image && c->width == o[0] && c->height == o[1] &&
        c->samples <= 1 && !c->is3d) {
        D.done = 1;
        if (!resolve(ex, &D.comp)) unresolved(ex), D.resolved = 0;
    }
}

// What the overlay shows (vk_dlaa_running): the backend's name while it resolved one of the last GAP_PRESENTS frames,
// so a setting whose library is missing, a failed backend and screens without the AA point (the title, loading) show
// nothing. Written at each present on the replay thread, read by the menu's.
static const char *running;

// At each present: the backend for the next frame (pick), the texture bias that goes with upscaling, and the frame's
// state. Returns 1 when the frame that ends here was sharpened before its HUD (sharpen): the present copies it.
int vk_dlaa_frame(void) {
    int sharpened = D.sharp_done;
    D.sharp_done = 0;
    vk_dlaa_swap_img = VK_NULL_HANDLE, vk_dlaa_swap_view = VK_NULL_HANDLE, vk_dlaa_swap_pic = VK_NULL_HANDLE;
    int bk = pick();
    static uint64_t evals_seen;   // an evaluation that completed this frame, not only the AA point reached
    int resolved = D.evals != evals_seen;
    evals_seen = D.evals;
    const char *name = backend_name(D.backend, D.last && (D.last->ow != D.last->w || D.last->oh != D.last->h));
    __atomic_store_n(&running, !dl_active() ? NULL : resolved ? name : D.gap + 1 < GAP_PRESENTS ? running : NULL,
                     __ATOMIC_RELAXED);
    // What no longer runs gives its video memory back after OFF_PRESENTS presents (not at once: a click through the
    // menu's choices would make and free the work images per click). Without DLSS (XeSS, nothing, or DLSS failed) the
    // sizes' DLSS features go, without FSR 3 its internal images; with no temporal AA at all the sizes' work images too.
    // XeSS keeps its context until exit: making it again needs the device idle.
    static uint32_t no_dlss, no_fsr3, no_taa;
    no_dlss = D.backend == BK_DLSS && dl_active() ? 0 : no_dlss + 1;
    no_fsr3 = D.backend == BK_FSR3 && dl_active() ? 0 : no_fsr3 + 1;
    no_taa = dl_active() ? 0 : no_taa + 1;
    if (no_fsr3 == OFF_PRESENTS) vk_fsr3_release();
    if (no_dlss == OFF_PRESENTS)
        for (int i = 0; i < 3; i++)
            if (D.sizes[i].feat) retire_feature(D.sizes[i].feat), D.sizes[i].feat = NULL;
    if (no_taa == OFF_PRESENTS)
        for (int i = 0; i < 3; i++)
            if (D.sizes[i].out) free_size(&D.sizes[i]);
    if (N.loaded) features_sweep(0);
    if (bk != D.backend) {
        int reduced = res_scene_reduced(), want = chosen();
        if (bk != BK_NONE && bk != want)
            rt_log("dlaa: %s cannot run, and the scene is reduced: %s upscales it until a restart",
                   want == BK_NONE ? "no upscaler is chosen" : backend_name(want, 1), backend_name(bk, 1));
        else
            rt_log("dlaa: %s %s", reduced ? "upscaling" : "temporal anti-aliasing", backend_name(bk, reduced));
        D.backend = bk;
        // either way the next frame starts a new history from an unjittered frame
        D.reset = 1, D.cam_prev_ok = D.view_ok = 0, D.jit_ok = D.jit_used = D.armed = D.done = D.gbuf = D.ended = 0;
        D.resolved = D.scaled = D.comp_ok = 0, D.cur = NULL;
        D.hold = 0, D.gap = 0;
        memset(D.cams, 0, sizeof D.cams);
        vk_mvobj_drop(), vk_mvprev_forget(1);
        return sharpened;
    }
    if (!dl_active()) {   // off, or the backend failed during the run: no jitter may stay on without the AA to resolve it
        D.jit_ok = D.jit_used = D.armed = D.done = D.gbuf = D.ended = D.resolved = D.scaled = D.comp_ok = 0;
        D.cur = NULL;
        vk_mvobj_drop(), vk_mvprev_forget(1);
        return sharpened;
    }
    D.gap = D.done && D.resolved ? 0 : D.gap + 1;   // a new history after GAP_PRESENTS of these
    // no AA point: this frame's object list and copies go; the next frame is matched against the last one the AA ran
    // on, unless the gap is long enough for a new history
    if (!(D.done && D.resolved)) D.frames_missed++, vk_mvobj_drop(), vk_mvprev_forget(D.gap >= GAP_PRESENTS);
    if (D.jit_used && !(D.done && D.resolved)) D.hold = 60, D.holds++;   // jittered but not resolved: that frame shook
    else if (D.hold) D.hold--;
    if (!(D.done && D.resolved)) D.jit_ok = 0;
    D.armed = D.done = D.jit_used = D.gbuf = D.ended = D.resolved = D.scaled = D.comp_ok = 0;
    D.cur = NULL;
    memset(D.cams, 0, sizeof D.cams);
    return sharpened;
}

const char *vk_dlaa_running(void) { return __atomic_load_n(&running, __ATOMIC_RELAXED); }

// For the menu's upscaler row (any thread; it only reads flags): 0 the choice runs or can at once, 1 it waits for a
// restart (DLSS or XeSS chosen after the start, so its library was not loaded; or off while the scene is reduced, which
// only a restart gives back), 2 it cannot run here (its library or the GPU could not bring it up, or it failed), so the
// game's SMAA runs, or while the scene is reduced the first upscaler that can.
int vk_dlaa_choice_state(const char *up) {
    if (!strcmp(up, "off")) return res_scene_reduced() ? 1 : 0;
    if (!strcmp(up, "dlss")) return !N.tried ? 1 : N.available && !D.failed ? 0 : 2;
    if (!strcmp(up, "xess")) return !vk_xess_tried() ? 1 : vk_xess_available() ? 0 : 2;
    if (!strcmp(up, "fsr3") || !strcmp(up, "auto")) return vk_fsr3_failed() && !can(BK_DLSS) ? 2 : 0;
    return 0;
}

// The backend that runs or would at once for the setting (pick), by name for the menu: "DLSS", "XeSS", "FSR 3", or
// NULL for none.
const char *vk_dlaa_upscaler_name(void) {
    int bk = pick();
    return bk == BK_NONE ? NULL : backend_name(bk, 1);
}

// ---------------------------------------------------------------- the game's SMAA while an upscaler runs
// Astral Chain anti-aliases with SMAA 1x, three full-screen passes after the composite: edge detection into an RGBA8
// target, blend weights from those edges and SMAA's area (160x560) and search (64x16) textures, then neighbourhood
// blending of the composite by the weights, which also writes the 8-bit picture the HUD is drawn over (measured in the
// dump above, passes 44-46). While an upscaler runs the edge pass is skipped and the weights pass clears its target
// instead of drawing, so the blending pass, which still runs, passes the anti-aliased picture through unchanged: SMAA
// on a temporally resolved picture only softens it. Upscaling, the edge and weight passes come before the upscaler
// (vk_dlaa_pass marks the frame resolved where the scene ends) and the blending pass then copies the upscaled picture.
// With anti_aliasing = off and no upscaler, SMAA's passes are skipped the same way and the blend copies the composite.
// Shaders are known by the FNV-1a hash of their SPIR-V, as vk_dump.c names them (this runtime's translator's); a
// translator change gives other hashes and SMAA then stays on, which the log says once.
#define SMAA_EDGES 0xf1ab7ad6c8883a54ull
#define SMAA_WEIGHTS 0x62325fa84d2ad3cbull

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
    static int found, reported;
    static uint64_t evals_seen;
    // SMAA comes after the AA point; only a resolved frame skips it, or every frame with anti_aliasing = off and no
    // upscaler running
    if (!fs || !(D.resolved || (!dl_active() && !cfg_smaa()))) return 0;
    if (!found && !reported && D.evals - evals_seen > 3600) {
        reported = 1;
        rt_log("dlaa: no draw of the game's SMAA found in %llu frames (its fragment shaders have other hashes): SMAA "
               "stays on over the temporal AA", (unsigned long long)(D.evals - evals_seen));
    }
    if (!evals_seen) evals_seen = D.evals;
    uint64_t h = spirv_hash(fs);
    int edges = h == SMAA_EDGES, weights = h == SMAA_WEIGHTS;
    if ((edges || weights) && !found) {
        found = 1;
        rt_log("dlaa: the game's SMAA is skipped while the temporal AA runs (fragment shader %016llx)",
               (unsigned long long)h);
    }
    if (edges) return 1;
    if (weights) {   // zero weights: the blending pass passes the picture through
        VkClearAttachment ca = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0};
        VkClearRect r = {.rect = {{0, 0}, {ex->pass_w, ex->pass_h}}, .baseArrayLayer = 0, .layerCount = 1};
        vkCmdClearAttachments(ex->cb, 1, &ca, 1, &r);
        ex->recorded++;
        return 1;
    }
    return 0;
}

// Before the device goes away (vk_device.c destroy_all, the device idle): the work images, every DLSS feature and the
// parameter object are released and NGX is shut down, so the driver's NGX core does not outlive the device it was
// started on with features still alive; then XeSS's context and FSR 3's objects. The motion pass's pipeline and buffer are left to the
// process's end, like the back end's other pipelines.
void vk_dlaa_release(void) {
    const VkCtx *c = vk_ctx();
    for (int i = 0; i < 3; i++) {
        struct Size *e = &D.sizes[i];
        if (e->out_view) vkDestroyImageView(c->device, e->out_view, NULL);
        if (e->mv_view) vkDestroyImageView(c->device, e->mv_view, NULL);
        if (e->sharp_view) vkDestroyImageView(c->device, e->sharp_view, NULL);
        if (e->out) vmaDestroyImage(c->vma, e->out, e->out_mem);
        if (e->mv) vmaDestroyImage(c->vma, e->mv, e->mv_mem);
        if (e->sharp) vmaDestroyImage(c->vma, e->sharp, e->sharp_mem);
        if (e->feat) N.release(e->feat);
        memset(e, 0, sizeof *e);
    }
    D.last = D.cur = NULL;
    if (N.loaded) {
        features_sweep(1);
        if (D.params && N.destroy_params) N.destroy_params(D.params);
        D.params = NULL;
        if (N.inited && N.shutdown) N.shutdown(c->device);
        N.inited = N.available = 0;
    }
    vk_xess_release();
    vk_fsr3_shutdown();
}

