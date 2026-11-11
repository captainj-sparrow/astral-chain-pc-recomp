// XeSS: Intel XeSS Super Resolution (XeSS-SR 2.0, the library of SDK 3.0.2) as the scene's temporal anti-aliasing at
// its own size (its Native AA quality setting), behind AC_ANTI_ALIASING=xess (astralchain.ini anti_aliasing = xess).
// libxess.dll (from Intel's XeSS SDK, shipped beside astralchain.exe by the release; Intel Simplified Software License) is loaded at run time
// from the folder of the exe. No Intel header
// is compiled in: the SDK's headers are under Intel's own license, so the few types and entry points used are declared
// here from the SDK's documented C ABI (xess.h, xess_vk.h of SDK 3.0.2), as vk_dlaa.c does for NVIDIA's NGX.
//
// vk_dlaa.c does everything around XeSS: it finds the scene in the frame, jitters its draws, makes the camera motion
// vectors and calls vk_xess_evaluate where DLAA would run, then copies the result over the scene's light target. This
// file brings XeSS up and records its evaluation.
//
// What the device needs: XeSS names its instance and device extensions and a chain of feature structures
// (mutableDescriptorType, shaderStorageImageWriteWithoutFormat, and on most drivers shaderInt8 and
// shaderIntegerDotProduct for its faster path). vk_device.c enables every feature the device supports, so the device is
// fit for XeSS when it supports each feature XeSS asks for and the one extension structure among them that is not core
// (VK_EXT_mutable_descriptor_type) is chained; vk_xess_device_exts checks this before any extension is added, so a
// device that lacks something is made exactly as without XeSS.
//
// Threads: XeSS is not thread-safe. The context is made, built, initialised and executed on the replay thread (the first
// time the setting is on there: vk_xess_evaluate), and destroyed at shutdown once the device is idle.
#include "vk_xess.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- the SDK's ABI (xess.h, xess_vk.h)
// Structures are packed to 8 in the SDK, which on x64 is their natural layout.
typedef int XessResult;   // xess_result_t: 0 success, 1..2 warnings, negative errors
#define XESS_OK(r) ((r) >= 0)
typedef struct XessCtx *XessHandle;   // xess_context_handle_t
typedef struct { uint16_t major, minor, patch, reserved; } XessVersion;
typedef struct { uint32_t x, y; } Xess2d;
enum { XESS_QUALITY_AA = 106 };                     // XESS_QUALITY_SETTING_AA: input and output of one size
enum { XESS_FLAG_AUTOEXPOSURE = 1 << 8 };           // XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE
enum { XESS_LOG_DEBUG, XESS_LOG_INFO, XESS_LOG_WARNING, XESS_LOG_ERROR };
typedef struct {   // xess_vk_image_view_info
    VkImageView imageView;
    VkImage image;
    VkImageSubresourceRange subresourceRange;
    VkFormat format;
    unsigned int width, height;
} XessView;
typedef struct {   // xess_vk_execute_params_t
    XessView color, velocity, depth, exposure, responsive, output;
    float jitterX, jitterY, exposureScale;
    uint32_t resetHistory, inputWidth, inputHeight;
    Xess2d colorBase, mvBase, depthBase, responsiveBase, reserved0, outputBase;
} XessExec;
typedef struct {   // xess_vk_init_params_t
    Xess2d outputResolution;
    int qualitySetting;
    uint32_t initFlags, creationNodeMask, visibleNodeMask;
    VkDeviceMemory tempBufferHeap;
    uint64_t bufferHeapOffset;
    VkDeviceMemory tempTextureHeap;
    uint64_t textureHeapOffset;
    VkPipelineCache pipelineCache;
} XessInit;
_Static_assert(sizeof(XessView) == 48, "xess_vk_image_view_info layout");
_Static_assert(sizeof(XessExec) == 6 * 48 + 24 + 6 * 8, "xess_vk_execute_params_t layout");
_Static_assert(sizeof(XessInit) == 64, "xess_vk_init_params_t layout");

typedef void (*XessLogFn)(const char *message, int level);
typedef XessResult (*PFN_GetVersion)(XessVersion *);
typedef XessResult (*PFN_InstExts)(uint32_t *, const char *const **, uint32_t *);
typedef XessResult (*PFN_DevExts)(VkInstance, VkPhysicalDevice, uint32_t *, const char *const **);
typedef XessResult (*PFN_DevFeats)(VkInstance, VkPhysicalDevice, void **);
typedef XessResult (*PFN_Create)(VkInstance, VkPhysicalDevice, VkDevice, XessHandle *);
typedef XessResult (*PFN_Build)(XessHandle, VkPipelineCache, bool, uint32_t);
typedef XessResult (*PFN_Init)(XessHandle, const XessInit *);
typedef XessResult (*PFN_Execute)(XessHandle, VkCommandBuffer, const XessExec *);
typedef XessResult (*PFN_Destroy)(XessHandle);
typedef XessResult (*PFN_SetLog)(XessHandle, int, XessLogFn);
typedef XessResult (*PFN_SetScale)(XessHandle, float, float);

static struct {
    void *lib;
    PFN_GetVersion version;
    PFN_InstExts inst_exts;
    PFN_DevExts dev_exts;
    PFN_DevFeats dev_feats;
    PFN_Create create;
    PFN_Build build;
    PFN_Init init;
    PFN_Execute execute;
    PFN_Destroy destroy;
    PFN_SetLog set_log;
    PFN_SetScale jitter_scale, velocity_scale;
    int tried, loaded;
    int device_ok;        // vk_xess_device_exts: the device is made with what XeSS needs
    int failed;           // a call failed: off for the rest of the run
    XessHandle ctx;
    uint32_t w, h;        // the size the context is initialised for, 0: not yet
    uint64_t evals, inits;
} X;

int vk_xess_wanted(void) {
    static int wanted = -1;
    if (wanted < 0) wanted = !strcmp(cfg_value("AC_ANTI_ALIASING"), "xess");
    return wanted;
}

int vk_xess_tried(void) { return X.tried; }

static int lib_load(void) {
    if (X.tried) return X.loaded;
    X.tried = 1;
#ifdef _WIN32
    char path[PLAT_PATH_MAX];
    int len = plat_exe_path(path, sizeof path - 16);
    char *cut = len ? plat_last_sep(path) : NULL;
    if (!cut) return 0;
    strcpy(cut + 1, "libxess.dll");
    X.lib = plat_lib_open(path);
    if (!X.lib) {
        rt_log("xess: cannot load %s (missing, or the Microsoft Visual C++ 2015-2022 runtime is not installed): XeSS off",
               path);
        return 0;
    }
#define FN(f, name) X.f = (void *)plat_lib_sym(X.lib, name)
    FN(version, "xessGetVersion");
    FN(inst_exts, "xessVKGetRequiredInstanceExtensions");
    FN(dev_exts, "xessVKGetRequiredDeviceExtensions");
    FN(dev_feats, "xessVKGetRequiredDeviceFeatures");
    FN(create, "xessVKCreateContext");
    FN(build, "xessVKBuildPipelines");
    FN(init, "xessVKInit");
    FN(execute, "xessVKExecute");
    FN(destroy, "xessDestroyContext");
    FN(set_log, "xessSetLoggingCallback");
    FN(jitter_scale, "xessSetJitterScale");
    FN(velocity_scale, "xessSetVelocityScale");
#undef FN
    if (!X.version || !X.inst_exts || !X.dev_exts || !X.dev_feats || !X.create || !X.init || !X.execute || !X.destroy) {
        rt_log("xess: %s lacks Vulkan entry points: XeSS off", path);
        return 0;
    }
    XessVersion v = {0};
    X.version(&v);
    rt_log("xess: %s, XeSS %u.%u.%u", path, v.major, v.minor, v.patch);
    return X.loaded = 1;
#else
    rt_log("xess: XeSS is a Windows library: off on this system");
    return 0;
#endif
}

static int listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return 1;
    return 0;
}

void vk_xess_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    if (!vk_xess_wanted() || !lib_load()) return;
    uint32_t ni = 0, api = 0;
    const char *const *ie = NULL;
    if (!XESS_OK(X.inst_exts(&ni, &ie, &api))) {
        rt_log("xess: no instance extension list: XeSS off");
        X.loaded = 0;
        return;
    }
    if (api > VK_API_VERSION_1_3)
        rt_log("xess: XeSS asks for Vulkan %u.%u, the runtime makes a 1.3 instance", VK_API_VERSION_MAJOR(api),
               VK_API_VERSION_MINOR(api));
    uint32_t add = 0;
    for (uint32_t i = 0; i < ni; i++) add += !listed(ext, *n, ie[i]);
    if (*n + add > max) {   // an instance without one of them would fail XeSS later, in the middle of play
        rt_log("xess: no room for XeSS's %u instance extensions: XeSS off", add);
        X.loaded = 0;
        return;
    }
    for (uint32_t i = 0; i < ni; i++)
        if (!listed(ext, *n, ie[i])) ext[(*n)++] = ie[i];
}

// The feature structures XeSS may ask for, with their sizes: the core ones (vk_device.c enables every core feature the
// device supports through the 1.1, 1.2 and 1.3 structures) and the one extension structure create_device chains for
// XeSS. Anything else is a feature this runtime would not enable, so XeSS stays off.
#define FS(stype, type, core) {VK_STRUCTURE_TYPE_##stype, sizeof(type), core}
static const struct { VkStructureType s; size_t size; int core; } FEATS[] = {
    FS(PHYSICAL_DEVICE_FEATURES_2, VkPhysicalDeviceFeatures2, 1),
    FS(PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, VkPhysicalDeviceVulkan11Features, 1),
    FS(PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features, 1),
    FS(PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features, 1),
    FS(PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, VkPhysicalDevice16BitStorageFeatures, 1),
    FS(PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, VkPhysicalDevice8BitStorageFeatures, 1),
    FS(PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, VkPhysicalDeviceShaderFloat16Int8Features, 1),
    FS(PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, VkPhysicalDeviceShaderIntegerDotProductFeatures, 1),
    FS(PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, VkPhysicalDeviceSubgroupSizeControlFeatures, 1),
    FS(PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES, VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures, 1),
    FS(PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, VkPhysicalDeviceDescriptorIndexingFeatures, 1),
    FS(PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, VkPhysicalDeviceScalarBlockLayoutFeatures, 1),
    FS(PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, VkPhysicalDeviceTimelineSemaphoreFeatures, 1),
    FS(PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, VkPhysicalDeviceBufferDeviceAddressFeatures, 1),
    FS(PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, VkPhysicalDeviceVulkanMemoryModelFeatures, 1),
    FS(PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, VkPhysicalDeviceHostQueryResetFeatures, 1),
    FS(PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, VkPhysicalDeviceSynchronization2Features, 1),
    FS(PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, VkPhysicalDeviceDynamicRenderingFeatures, 1),
    FS(PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, VkPhysicalDeviceMaintenance4Features, 1),
    FS(PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES, VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures, 1),
    FS(PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT, VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT, 0),
};
#undef FS

// 1 when every VkBool32 XeSS sets in its structure is supported by the device (the structure queried on its own).
static int feature_supported(VkPhysicalDevice pd, const VkBaseOutStructure *want, size_t size) {
    _Alignas(16) uint8_t have[1024];
    if (size > sizeof have) return 0;
    memset(have, 0, size);
    ((VkBaseOutStructure *)have)->sType = want->sType;
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    int is_f2 = want->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (!is_f2) f2.pNext = have;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    const VkBool32 *w = (const VkBool32 *)((const uint8_t *)want + sizeof(VkBaseOutStructure));
    const VkBool32 *h = is_f2 ? (const VkBool32 *)&f2.features
                              : (const VkBool32 *)(have + sizeof(VkBaseOutStructure));
    size_t n = (size - sizeof(VkBaseOutStructure)) / sizeof(VkBool32);
    for (size_t i = 0; i < n; i++)
        if (w[i] && !h[i]) {
            rt_log("xess: the device lacks feature %zu of structure %d, which XeSS asks for", i, (int)want->sType);
            return 0;
        }
    return 1;
}

int vk_xess_device_exts(VkInstance inst, VkPhysicalDevice pd, const VkExtensionProperties *have, uint32_t nhave,
                        const char **ext, uint32_t *n, uint32_t max) {
    if (!vk_xess_wanted() || !lib_load()) return 0;
    uint32_t nd = 0;
    const char *const *de = NULL;
    if (!XESS_OK(X.dev_exts(inst, pd, &nd, &de))) {
        rt_log("xess: no device extension list for this GPU: XeSS off");
        return 0;
    }
    for (uint32_t i = 0; i < nd; i++) {
        int ok = 0;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) {
            rt_log("xess: the device lacks %s, which XeSS asks for: XeSS off", de[i]);
            return 0;
        }
    }
    void *chain = NULL;
    if (!XESS_OK(X.dev_feats(inst, pd, &chain))) {
        rt_log("xess: no device feature list for this GPU: XeSS off");
        return 0;
    }
    for (const VkBaseOutStructure *s = chain; s; s = s->pNext) {
        size_t k = 0;
        while (k < sizeof FEATS / sizeof *FEATS && FEATS[k].s != s->sType) k++;
        if (k == sizeof FEATS / sizeof *FEATS) {
            rt_log("xess: XeSS asks for features in structure %d, which this runtime does not enable: XeSS off",
                   (int)s->sType);
            return 0;
        }
        if (!feature_supported(pd, s, FEATS[k].size)) return 0;
    }
    if (*n + nd > max) {
        rt_log("xess: no room for XeSS's %u device extensions: XeSS off", nd);
        return 0;
    }
    for (uint32_t i = 0; i < nd; i++)
        if (!listed(ext, *n, de[i])) ext[(*n)++] = de[i];
    X.device_ok = 1;
    return 1;
}

int vk_xess_available(void) { return X.loaded && X.device_ok && !X.failed; }

static void on_log(const char *message, int level) {
    if (level >= XESS_LOG_WARNING) rt_log("xess: %s: %s", level >= XESS_LOG_ERROR ? "error" : "warning", message);
}

static int fail(const char *what, XessResult r) {
    rt_log("xess: %s failed (%d): XeSS off for this run", what, r);
    X.failed = 1;
    return 0;
}

// The context, made on the replay thread at the first evaluation; its pipelines are built right away (the JIT of
// XeSS's kernels: a hitch the first time a session turns it on, logged with its length).
static int context(void) {
    if (X.ctx) return 1;
    const VkCtx *c = vk_ctx();
    uint64_t t0 = kern_now_ns();
    XessResult r = X.create(c->instance, c->phys, c->device, &X.ctx);
    if (!XESS_OK(r) || !X.ctx) return fail("context creation", r);
    if (X.set_log) X.set_log(X.ctx, XESS_LOG_WARNING, on_log);
    if (X.build && !XESS_OK(r = X.build(X.ctx, VK_NULL_HANDLE, true, XESS_FLAG_AUTOEXPOSURE)))
        return fail("pipeline build", r);
    // Signs of the jitter and of the motion vectors (AC_XESS_JITTER_SCALE, AC_XESS_MV_SCALE: test knobs, "1" or "-1").
    // The defaults follow XeSS's documented conventions, which are DLSS's: the jitter is the shift of the picture in
    // pixels, the vectors point from this frame's pixel to its position in the last frame.
    const char *je = cfg_get("AC_XESS_JITTER_SCALE"), *me = cfg_get("AC_XESS_MV_SCALE");
    float js = je && *je ? (float)atof(je) : 1.0f, ms = me && *me ? (float)atof(me) : 1.0f;
    if (X.jitter_scale && js != 1.0f) X.jitter_scale(X.ctx, js, js);
    if (X.velocity_scale && ms != 1.0f) X.velocity_scale(X.ctx, ms, ms);
    rt_log("xess: context made and its pipelines built in %.0f ms", (double)(kern_now_ns() - t0) / 1e6);
    return 1;
}

// xessVKInit for the scene size. XeSS may submit copies of its own while it initialises and requires the GPU done with
// its previous work before a new size, so it runs with the queues locked and, for a new size, after the device is idle.
static int init_size(uint32_t w, uint32_t h) {
    if (X.w == w && X.h == h) return 1;
    uint64_t t0 = kern_now_ns();
    XessInit ip = {.outputResolution = {w, h}, .qualitySetting = XESS_QUALITY_AA, .initFlags = XESS_FLAG_AUTOEXPOSURE};
    vk_queue_lock();
    if (X.w) vkDeviceWaitIdle(vk_ctx()->device);
    XessResult r = X.init(X.ctx, &ip);
    vk_queue_unlock();
    if (!XESS_OK(r)) return fail("initialisation", r);
    X.w = w, X.h = h, X.inits++;
    rt_log("xess: Native AA at %ux%u (initialised in %.0f ms)", w, h, (double)(kern_now_ns() - t0) / 1e6);
    return 1;
}

static XessView view_of(const XessImage *m, VkImageAspectFlags aspect, uint32_t w, uint32_t h) {
    return (XessView){m->view, m->image, {aspect, m->level, 1, m->layer, 1}, m->format, w, h};
}

static int has_stencil(VkFormat f) {
    return f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT;
}

// XeSS reads its inputs in SHADER_READ_ONLY_OPTIMAL (the back end keeps every image in GENERAL): the three inputs go
// there and back around the evaluation, with the full ordering the back end uses between steps.
static void inputs_layout(VkCommandBuffer cb, const XessFrame *f, int to_read) {
    VkImageLayout a = to_read ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageLayout b = to_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    const XessImage *m[3] = {&f->color, &f->depth, &f->mv};
    VkImageMemoryBarrier2 ib[3];
    for (int i = 0; i < 3; i++) {
        VkImageAspectFlags asp = i != 1 ? VK_IMAGE_ASPECT_COLOR_BIT
                                        : VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil(m[i]->format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = a, .newLayout = b,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = m[i]->image,
                                        .subresourceRange = {asp, m[i]->level, 1, m[i]->layer, 1}};
    }
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 3, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
}

int vk_xess_evaluate(VkCommandBuffer cb, const XessFrame *f) {
    if (!vk_xess_available() || !context() || !init_size(f->w, f->h)) return 0;
    XessExec p;
    memset(&p, 0, sizeof p);
    p.color = view_of(&f->color, VK_IMAGE_ASPECT_COLOR_BIT, f->w, f->h);
    p.velocity = view_of(&f->mv, VK_IMAGE_ASPECT_COLOR_BIT, f->w, f->h);
    p.depth = view_of(&f->depth, VK_IMAGE_ASPECT_DEPTH_BIT, f->w, f->h);
    p.output = view_of(&f->out, VK_IMAGE_ASPECT_COLOR_BIT, f->w, f->h);
    p.jitterX = f->jx, p.jitterY = f->jy;
    p.exposureScale = 1.0f;
    p.resetHistory = f->reset ? 1u : 0u;
    p.inputWidth = f->w, p.inputHeight = f->h;
    // The barrier goes through the recorder thread (vk_rec.h) like the replay's own commands; XeSS records into cb
    // itself, so the recorder must have made it first. The barrier after is queued behind XeSS's commands, in order.
    inputs_layout(cb, f, 1);
    rec_drain();
    XessResult r = X.execute(X.ctx, cb, &p);
    inputs_layout(cb, f, 0);
    if (!XESS_OK(r)) return fail("evaluation", r);
    X.evals++;
    return 1;
}

void vk_xess_release(void) {
    if (X.ctx) X.destroy(X.ctx);   // the device is idle (vk_device.c destroy_all)
    X.ctx = NULL, X.w = X.h = 0;
}

// ---------------------------------------------------------------- --probe-xess
// XeSS on its own, with the device the game makes (anti_aliasing = xess for the run): a context, Native AA at 256x144, and one
// evaluation of a flat grey picture with zero motion, which must come out grey. Without libxess.dll, or on a device
// XeSS cannot run on, it says so and passes.
static int probe_image(uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags u, VkImageAspectFlags asp, VkImage *img,
                       VmaAllocation *mem, VkImageView *view) {
    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = u, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, mem, NULL) != VK_SUCCESS) return 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f, .subresourceRange = {asp, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, view) == VK_SUCCESS;
}

static double half_to_double(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? ldexp(m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexp(m + 1024, e - 25);
    return h & 0x8000 ? -v : v;
}

int probe_xess(void) {
    cfg_set("AC_ANTI_ALIASING", "xess");
    if (!vk_init()) {
        printf("probe-xess: FAILED (no Vulkan device)\n");
        return 1;
    }
    if (!vk_xess_available()) {
        printf("probe-xess: passed (XeSS not available here: %s)\n",
               !X.loaded ? "libxess.dll not loaded" : "the device lacks what XeSS needs");
        vk_shutdown();
        return 0;
    }
    const VkCtx *c = vk_ctx();
    enum { W = 256, H = 144 };
    VkImage im[4] = {0};
    VmaAllocation mem[4] = {0};
    VkImageView vw[4] = {0};
    const VkFormat fmt[4] = {VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_R16G16_SFLOAT,
                             VK_FORMAT_R16G16B16A16_SFLOAT};
    const VkImageUsageFlags base = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageUsageFlags use[4] = {base | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, base | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                      base | VK_IMAGE_USAGE_STORAGE_BIT, base | VK_IMAGE_USAGE_STORAGE_BIT};
    const VkImageAspectFlags asp[4] = {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                                       VK_IMAGE_ASPECT_COLOR_BIT};
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++) ok = probe_image(W, H, fmt[i], use[i], asp[i], &im[i], &mem[i], &vw[i]);
    VkBuffer rb = VK_NULL_HANDLE;
    VmaAllocation rm = NULL;
    VmaAllocationInfo rinfo = {0};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)W * H * 8,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (ok) ok = vmaCreateBuffer(c->vma, &bi, &bai, &rb, &rm, &rinfo) == VK_SUCCESS;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->gfx_family};
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (ok) ok = vkCreateCommandPool(c->device, &pi, NULL, &pool) == VK_SUCCESS &&
                 vkCreateFence(c->device, &fi, NULL, &fence) == VK_SUCCESS;
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    if (ok) ok = vkAllocateCommandBuffers(c->device, &ai, &cb) == VK_SUCCESS;
    int evaluated = 0;
    if (ok) {
        VkCommandBufferBeginInfo bb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bb);
        VkImageMemoryBarrier2 ib[4];
        for (int i = 0; i < 4; i++)
            ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                            .subresourceRange = {i == 1 ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                                                        : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 4,
                               .pImageMemoryBarriers = ib};
        vkCmdPipelineBarrier2(cb, &di);
        VkClearColorValue grey = {{0.5f, 0.5f, 0.5f, 1.0f}}, zero = {{0, 0, 0, 0}};
        VkClearDepthStencilValue far = {1.0f, 0};
        VkImageSubresourceRange cr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageSubresourceRange dr = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, im[0], VK_IMAGE_LAYOUT_GENERAL, &grey, 1, &cr);
        vkCmdClearDepthStencilImage(cb, im[1], VK_IMAGE_LAYOUT_GENERAL, &far, 1, &dr);
        vkCmdClearColorImage(cb, im[2], VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &cr);
        vkCmdClearColorImage(cb, im[3], VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &cr);
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
        VkDependencyInfo md = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(cb, &md);
        XessFrame f = {.color = {im[0], vw[0], fmt[0], 0, 0}, .depth = {im[1], vw[1], fmt[1], 0, 0},
                       .mv = {im[2], vw[2], fmt[2], 0, 0}, .out = {im[3], vw[3], fmt[3], 0, 0},
                       .w = W, .h = H, .reset = 1};
        evaluated = vk_xess_evaluate(cb, &f);
        vkCmdPipelineBarrier2(cb, &md);
        VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}};
        vkCmdCopyImageToBuffer(cb, im[3], VK_IMAGE_LAYOUT_GENERAL, rb, 1, &bc);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
        vk_queue_lock();
        ok = vkQueueSubmit(c->gfx_queue, 1, &si, fence) == VK_SUCCESS;
        vk_queue_unlock();
        ok = ok && vkWaitForFences(c->device, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    }
    // the centre pixel, RGBA16F: about 0.5 grey (auto exposure may move it a little; a black or NaN result is a failure)
    double grey = -1;
    if (ok && evaluated) {
        const uint16_t *px = (const uint16_t *)rinfo.pMappedData + ((size_t)(H / 2) * W + W / 2) * 4;
        grey = half_to_double(px[1]);
    }
    int pass = ok && evaluated && grey > 0.25 && grey < 0.75;
    printf("probe-xess: %s (evaluated %d, centre green %.3f)\n", pass ? "passed" : "FAILED", evaluated, grey);
    if (fence) vkDestroyFence(c->device, fence, NULL);
    if (pool) vkDestroyCommandPool(c->device, pool, NULL);
    if (rb) vmaDestroyBuffer(c->vma, rb, rm);
    for (int i = 0; i < 4; i++) {
        if (vw[i]) vkDestroyImageView(c->device, vw[i], NULL);
        if (im[i]) vmaDestroyImage(c->vma, im[i], mem[i]);
    }
    vk_shutdown();
    return pass ? 0 : 1;
}
