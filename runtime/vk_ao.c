// Screen-space ambient occlusion (AC_AO = off, low or high). Astral Chain renders forward and
// has no ambient occlusion of its own: every surface takes the same ambient light however enclosed it is. This pass
// darkens the scene light where the depth buffer shows nearby geometry hiding part of the sky above a surface:
// creases, corners, the ground under characters and objects.
//
// Where it runs: Astral Chain's frame (vk_dlaa.c describes it) first draws into its light target (R11G11B10F, RGBA16F
// in a few places) with the scene depth (D32F with stencil), then turns that depth into an R32F linear depth in a pass
// of its own, with no depth target, and some passes later draws the opaque world into the light target with the same
// depth. The first light-target pass with that depth after the linear depth pass is the opaque lighting; when the next
// pass with other targets begins, or a dispatch or copy comes first, the steps run (runtime/shaders): ao_prep.comp takes
// the scene depth to a view z at the occlusion's resolution, ao.comp computes the occlusion there, ao_meter.comp reads
// the light target's typical brightness, ao_blur.comp averages the noise pattern out and lets go of what glows, and ao_apply.frag scales the result up and multiplies the light target by it through a blend.
// Transparent surfaces, particles, the bloom, the tonemap and the HUD come after, untouched. The
// frame state starts over at each present (vk_ao_frame).
//
// The camera is the vertex stage's c3 block of the scene's draws (256 bytes, row-major vec4 rows: view, projection,
// view-projection, inverse view; vk_dlaa.c mv_pass): the one most draws with the scene depth bound this frame. View z
// comes from the depth through the inverse of its projection, made here in double precision, with the depth mode of
// DeviceSetDepthMode (0..1 or -1..1); view x and y come from z through the projection's scales. The scene targets are
// made at their own size whatever the output resolution or upscaling mode (resolution.c), so the pass works at the size
// of the targets it finds, ultrawide ones included: every length on screen is counted in heights of the target, and
// where aspect = wide stretches an ultrawide view over a 16:9 target, ao.comp steps by the texel's shape in view space.
// The occlusion is computed at half the target's resolution (AO_SCALE) with a pattern that stays put, with a temporal
// upscaler too. Measured with FSR 3 in its quality mode (a 1696x954 scene for 2560x1440, a street at rest, the mean
// change of the picture from one frame to the next): 2.3 without occlusion, 2.2 at half resolution with the pattern
// fixed, 2.5 when the pattern turns every frame, 2.4 and 3.4 for the same two at the scene's full resolution, which
// also costs 2.5 ms in place of 0.9. The upscaler does not average a turning pattern out, and the blur already does.
//
// How dark it gets is fixed (AO_OPEN, AO_POWER), the same for every player: no setting. The measured visibility of a
// surface in a street is seldom 1 (a wall 20 m away with pipes and ledges within the radius reads about 0.85, and a
// plain power of it darkened the whole picture by a fifth), so visibilities from AO_OPEN up count as open and stay
// untouched, and the rest falls off with AO_POWER: contacts and corners go dark, open walls and ground keep the game's
// light.
// Live: the setting is read again whenever cfg_version moves, so the menu switches it on and off at once.
// AC_AO_DEBUG=1 (no settings row) shows the occlusion factor in grey in place of the light. AC_GPU_PASSES times the
// pass on a row of its own ("ao").
#include "vk.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ao_apply_spv.h"
#include "ao_blur_spv.h"
#include "ao_meter_spv.h"
#include "ao_prep_spv.h"
#include "ao_spv.h"
#include "cfg.h"
#include "hdr_fullscreen_spv.h"
#include "host.h"
#include "nvn_api.h"
#include "vk_exec.h"

#define CAM_BYTES 256u        // the c3 camera block: vec4 rows 0-15
#define Z_FORMAT VK_FORMAT_R32_SFLOAT   // ao_prep.comp: view z (storage support is mandatory)
#define AO_RADIUS 1.5f        // world radius of the search, metres: reaches under cars and along walls
#define AO_SCALE 2u           // light-target pixels per occlusion texel, each way
#define AO_REACH 4.0f         // the search covers at most 1/AO_REACH of the picture's height: close surfaces, close-ups
#define AO_GLOW_LO 6.0f       // light this many times the picture's typical brightness starts to lose its occlusion
#define AO_GLOW_HI 24.0f      // and this many times has none: signs, lamps, fires (ao_blur.comp)
#define AO_OPEN 0.90f         // visibility from which a surface counts as open: no darkening (ao_apply.frag)
#define AO_POWER 2.0f         // the remaining visibility to this power is the factor on the light
#define AO_FADE 120.0f        // the occlusion fades out between half this view distance and this one
#define MAX_APPLY 4           // apply pipelines: light target format x debug

struct Cam {
    uint64_t addr;
    uint32_t n;
    int have;
    float block[CAM_BYTES / 4];
};

static struct {
    // settings
    int level;                 // 0 off, 1 low, 2 high
    int debug;
    unsigned cfg_seen;
    int cfg_read;
    // pipelines
    int tried, ok;
    VkSampler nearest, linear;
    VkDescriptorSetLayout dsl_c, dsl_b, dsl_a;   // compute steps, the blur, the apply
    VkPipelineLayout lay_c, lay_b, lay_a;
    VkPipeline comp[4];        // ao_prep, ao, ao_meter, ao_blur
    struct { VkFormat fmt; int debug; VkPipeline p; } apply[MAX_APPLY];
    int napply;
    // the images at the occlusion's resolution: view z, the occlusion, the blur's other side; and the 1x1 brightness
    struct { VkImage img; VmaAllocation mem; VkImageView view; } im[4];
    uint32_t iw, ih;
    VkFormat ao_format;        // ao.comp's output: RG16F, half the bytes its readers fetch, else RG32F (mandatory)
    // this frame
    VkExecTarget light, depth;  // the light target and scene depth of the last light-target pass
    int have, lin, armed, done;
    int wrote;                  // a draw of the armed pass writes colour: it is the opaque lighting (vk_ao_draw)
    struct Cam cams[4];
    int logged, no_cam_logs, fmt_logs;
} A;

// ---------------------------------------------------------------- settings
static void read_settings(void) {
    unsigned v = cfg_version();
    if (A.cfg_read && v == A.cfg_seen) return;
    A.cfg_read = 1, A.cfg_seen = v;
    const char *q = cfg_value("AC_AO");
    int level = !strcmp(q, "high") ? 2 : !strcmp(q, "low") ? 1 : 0;
    const char *e = cfg_get("AC_AO_DEBUG");   // diagnostic: no settings row
    int debug = e && *e && strcmp(e, "0");
    if (level != A.level || debug != A.debug)
        rt_log("vk: ambient occlusion %s%s", level == 2 ? "high" : level ? "low" : "off", debug ? " (debug view)" : "");
    A.level = level, A.debug = debug;
}

// ---------------------------------------------------------------- pipelines
static VkShaderModule module(VkDevice d, const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    return vkCreateShaderModule(d, &mi, NULL, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

static VkDescriptorSetLayout set_layout(VkDevice d, const VkDescriptorType *types, uint32_t n, VkShaderStageFlags st) {
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < n; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = types[i], .descriptorCount = 1,
                                              .stageFlags = st};
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                          .bindingCount = n, .pBindings = b};
    VkDescriptorSetLayout l = VK_NULL_HANDLE;
    return vkCreateDescriptorSetLayout(d, &ci, NULL, &l) == VK_SUCCESS ? l : VK_NULL_HANDLE;
}

static VkPipelineLayout pipe_layout(VkDevice d, VkDescriptorSetLayout *dsl, VkShaderStageFlags st, uint32_t pc) {
    VkPushConstantRange pr = {st, 0, pc};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkPipelineLayout l = VK_NULL_HANDLE;
    return vkCreatePipelineLayout(d, &li, NULL, &l) == VK_SUCCESS ? l : VK_NULL_HANDLE;
}

static int init(void) {
    A.tried = 1;
    VkCtx *c = vk_ctx();
    VkDevice d = c->device;
    if (!vk_pipeline_push_fn()) {
        rt_log("vk: ambient occlusion off: no push descriptors");
        return 0;
    }
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkSamplerCreateInfo sl = si;
    sl.magFilter = sl.minFilter = VK_FILTER_LINEAR;   // the light target, for the brightness (R11G11B10F and RGBA16F filter)
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkDescriptorType tc[2] = {S, I}, tb[4] = {S, I, S, S}, ta[2] = {S, S};
    if (vkCreateSampler(d, &si, NULL, &A.nearest) != VK_SUCCESS || vkCreateSampler(d, &sl, NULL, &A.linear) != VK_SUCCESS ||
        !(A.dsl_c = set_layout(d, tc, 2, VK_SHADER_STAGE_COMPUTE_BIT)) ||
        !(A.dsl_b = set_layout(d, tb, 4, VK_SHADER_STAGE_COMPUTE_BIT)) ||
        !(A.dsl_a = set_layout(d, ta, 2, VK_SHADER_STAGE_FRAGMENT_BIT)) ||
        !(A.lay_c = pipe_layout(d, &A.dsl_c, VK_SHADER_STAGE_COMPUTE_BIT, 64)) ||
        !(A.lay_b = pipe_layout(d, &A.dsl_b, VK_SHADER_STAGE_COMPUTE_BIT, 16)) ||
        !(A.lay_a = pipe_layout(d, &A.dsl_a, VK_SHADER_STAGE_FRAGMENT_BIT, 64))) {
        rt_log("vk: ambient occlusion off: its pipeline layouts could not be created");
        return 0;
    }
    const struct { const uint32_t *code; size_t size; VkPipelineLayout lay; } cs[4] = {
        {ao_prep_spv, sizeof ao_prep_spv, A.lay_c}, {ao_spv, sizeof ao_spv, A.lay_c},
        {ao_meter_spv, sizeof ao_meter_spv, A.lay_c}, {ao_blur_spv, sizeof ao_blur_spv, A.lay_b}};
    for (int i = 0; i < 4; i++) {
        VkShaderModule m = module(d, cs[i].code, cs[i].size);
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                          .layout = cs[i].lay};
        VkResult r = m ? vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &ci, NULL, &A.comp[i]) : VK_ERROR_UNKNOWN;
        if (m) vkDestroyShaderModule(d, m, NULL);
        if (r != VK_SUCCESS) {
            A.comp[i] = VK_NULL_HANDLE;
            rt_log("vk: ambient occlusion off: its compute pipelines could not be created");
            return 0;
        }
    }
    // ao.comp writes without a format qualifier (shaderStorageImageWriteWithoutFormat, which vk_device.c requires)
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c->phys, VK_FORMAT_R16G16_SFLOAT, &fp);
    VkFormatFeatureFlags want = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    A.ao_format = (fp.optimalTilingFeatures & want) == want ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R32G32_SFLOAT;
    return A.ok = 1;
}

// The apply pipeline for a light target format: a multiplying blend (target = target x output), or none in debug.
static VkPipeline apply_pipeline(VkFormat fmt, int debug) {
    for (int i = 0; i < A.napply; i++)
        if (A.apply[i].fmt == fmt && A.apply[i].debug == debug) return A.apply[i].p;
    if (A.napply == MAX_APPLY) return VK_NULL_HANDLE;
    VkCtx *c = vk_ctx();
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c->phys, fmt, &fp);
    if (!debug && !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)) {
        if (A.fmt_logs++ < 4) rt_log("vk: ambient occlusion: the device cannot blend into format %d: skipped", (int)fmt);
        return VK_NULL_HANDLE;
    }
    VkShaderModule vs = module(c->device, hdr_fullscreen_spv, sizeof hdr_fullscreen_spv),
                   fs = module(c->device, ao_apply_spv, sizeof ao_apply_spv);
    VkPipelineShaderStageCreateInfo st[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs,
         .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                            .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba = {
        .blendEnable = debug ? VK_FALSE : VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR, .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = 1, .pAttachments = &ba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                        .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
    VkGraphicsPipelineCreateInfo gi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = 2, .pStages = st, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
                                       .layout = A.lay_a};
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vs && fs ? vkCreateGraphicsPipelines(c->device, VK_NULL_HANDLE, 1, &gi, NULL, &p) : VK_ERROR_UNKNOWN;
    if (vs) vkDestroyShaderModule(c->device, vs, NULL);
    if (fs) vkDestroyShaderModule(c->device, fs, NULL);
    if (r != VK_SUCCESS) {
        if (A.fmt_logs++ < 4) rt_log("vk: ambient occlusion: no apply pipeline for format %d: skipped", (int)fmt);
        return VK_NULL_HANDLE;
    }
    A.apply[A.napply].fmt = fmt, A.apply[A.napply].debug = debug, A.apply[A.napply++].p = p;
    return p;
}

// deferred: through vk_memory.c's deferred destruction (submissions in flight may still use them); else at once (the
// device is idle).
static void free_images(int deferred) {
    VkCtx *c = vk_ctx();
    for (int i = 0; i < 4; i++) {
        if (deferred) {
            vk_defer_view(A.im[i].view);
            if (A.im[i].img) vk_defer_image(A.im[i].img, A.im[i].mem);
        } else {
            if (A.im[i].view) vkDestroyImageView(c->device, A.im[i].view, NULL);
            if (A.im[i].img) vmaDestroyImage(c->vma, A.im[i].img, A.im[i].mem);
        }
        A.im[i].img = VK_NULL_HANDLE, A.im[i].view = VK_NULL_HANDLE, A.im[i].mem = NULL;
    }
    A.iw = A.ih = 0;
}

// The occlusion's images at w x h and the 1x1 brightness; a new size retires the old ones. New images go to GENERAL,
// the layout every image of the back end is kept in.
static int ensure_images(VkCommandBuffer cb, uint32_t w, uint32_t h) {
    if (A.im[0].img && A.iw == w && A.ih == h) return 1;
    VkCtx *c = vk_ctx();
    if (A.im[0].img) free_images(1);
    VkImageMemoryBarrier2 ib[4];
    for (int i = 0; i < 4; i++) {
        VkFormat f = i == 0 || i == 3 ? Z_FORMAT : A.ao_format;
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {i == 3 ? 1 : w, i == 3 ? 1 : h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &A.im[i].img, &A.im[i].mem, NULL) != VK_SUCCESS) {
            A.im[i].img = VK_NULL_HANDLE;
            free_images(1);
            return 0;
        }
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = A.im[i].img,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &A.im[i].view) != VK_SUCCESS) {
            A.im[i].view = VK_NULL_HANDLE;
            free_images(1);
            return 0;
        }
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = A.im[i].img,
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    }
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 4,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    A.iw = w, A.ih = h;
    return 1;
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

// This frame's camera: the block most scene draws bound. NULL: none seen.
static const float *frame_camera(void) {
    const struct Cam *best = NULL;
    for (int i = 0; i < 4; i++)
        if (A.cams[i].have && (!best || A.cams[i].n > best->n)) best = &A.cams[i];
    return best ? best->block : NULL;
}

// A perspective projection (row-major) as ao.comp needs it: clip w a multiple of view z, and clip x and y depending on
// their own coordinate and z alone (an off-centre frustum included), so view x and y follow from z and the scales.
static int plain_perspective(const double p[16]) {
    double ws = p[14];
    for (int i = 0; i < 16; i++)
        if (!isfinite(p[i])) return 0;
    return fabs(p[0]) > 1e-6 && fabs(p[5]) > 1e-6 && fabs(ws) > 1e-9 && fabs(p[12]) <= 1e-6 * fabs(ws) &&
           fabs(p[13]) <= 1e-6 * fabs(ws) && fabs(p[15]) <= 1e-6 * fabs(ws) && fabs(p[1]) <= 1e-6 * fabs(p[0]) &&
           fabs(p[3]) <= 1e-6 && fabs(p[4]) <= 1e-6 * fabs(p[5]) && fabs(p[7]) <= 1e-6;
}

// ---------------------------------------------------------------- the pass
// One compute step: src sampled at binding 0, dst written at binding 1, gx x gy groups. AC_GPU_PASSES times the steps
// on rows of their own (ao, step n): a step with n 0 counts with the next.
static void step(VkExec *ex, int n, VkPipeline pipe, VkDescriptorImageInfo src, VkImageView dst, const void *pc,
                 uint32_t pc_size, uint32_t gx, uint32_t gy) {
    VkCommandBuffer cb = ex->cb;
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wc[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &src},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.lay_c, 0, 2, wc);
    if (pc_size) vkCmdPushConstants(cb, A.lay_c, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    vkCmdDispatch(cb, gx, gy, 1);
    full_barrier(cb);
    if (n && vk_gpuprof_on()) vk_gpuprof_mark(ex, 6, (uint64_t)n);
}

// The blur (ao_blur.comp), from one occlusion image into the other, with the light target and its typical brightness.
static void blur(VkExec *ex, int from, int to, VkImageView light, float scale) {
    VkCommandBuffer cb = ex->cb;
    VkDescriptorImageInfo in = {A.nearest, A.im[from].view, VK_IMAGE_LAYOUT_GENERAL},
                          out = {VK_NULL_HANDLE, A.im[to].view, VK_IMAGE_LAYOUT_GENERAL},
                          li = {A.linear, light, VK_IMAGE_LAYOUT_GENERAL},
                          mean = {A.nearest, A.im[3].view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wb[4] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &in},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &out},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &li},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &mean}};
    float pb[4] = {scale, AO_GLOW_LO, AO_GLOW_HI, 0};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.comp[3]);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.lay_b, 0, 4, wb);
    vkCmdPushConstants(cb, A.lay_b, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pb, pb);
    vkCmdDispatch(cb, (A.iw + 7) / 8, (A.ih + 7) / 8, 1);
    full_barrier(cb);
}

static void run(VkExec *ex) {
    A.armed = 0, A.done = 1;
    if (!A.tried && !init()) return;
    if (!A.ok) return;
    const float *cam = frame_camera();
    double pj[16], inv[16];
    for (int i = 0; i < 16; i++) pj[i] = cam ? cam[16 + i] : 0;
    if (!cam || !plain_perspective(pj) || !inv4(pj, inv)) {
        if (A.no_cam_logs++ < 4)
            rt_log("vk: ambient occlusion: %s this frame: skipped", cam ? "no perspective projection" : "no camera block");
        return;
    }
    const VkExecTarget *t = &A.light, *z = &A.depth;
    uint32_t w = t->width, h = t->height, sc = AO_SCALE;
    uint32_t hw = (w + sc - 1) / sc, hh = (h + sc - 1) / sc, gx = (hw + 7) / 8, gy = (hh + 7) / 8;
    VkPipeline ap = apply_pipeline(t->format, A.debug);
    if (!ap) return;
    vk_exec_barrier(ex);   // ends the open pass; what drew the scene is done, feedback-layout targets back in GENERAL
    VkCommandBuffer cb = ex->cb;
    if (!ensure_images(cb, hw, hh)) {
        if (A.fmt_logs++ < 4) rt_log("vk: ambient occlusion: its %ux%u images could not be created", hw, hh);
        return;
    }
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    VkImageView cv = vk_exec_target_view(t->img, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                                         t->base_layer, 1, 0);
    if (!zv || !cv) return;
    if (!A.logged) {
        A.logged = 1;
        rt_log("vk: ambient occlusion on a %ux%u scene (format %d), %ux%u occlusion, projection y scale %.3f", w, h,
               (int)t->format, hw, hh, pj[5]);
    }
    float mode = nvn_depth_minus_w() ? 1.0f : 0.0f;
    struct { float inv2[4], inv3[4], size[4]; } pz = {.size = {(float)w, (float)h, mode, (float)sc}};
    for (int i = 0; i < 4; i++) pz.inv2[i] = (float)inv[8 + i], pz.inv3[i] = (float)inv[12 + i];
    VkDescriptorImageInfo depth_in = {A.nearest, zv, VK_IMAGE_LAYOUT_GENERAL},
                          z_in = {A.nearest, A.im[0].view, VK_IMAGE_LAYOUT_GENERAL},
                          a2_in = {A.nearest, A.im[2].view, VK_IMAGE_LAYOUT_GENERAL},
                          light_in = {A.linear, cv, VK_IMAGE_LAYOUT_GENERAL};
    // 1: view z at the occlusion's resolution
    step(ex, 1, A.comp[0], depth_in, A.im[0].view, &pz, sizeof pz, gx, gy);
    // 2: the occlusion. Clip x = p00 x + p02 z and clip w = p32 z give view x = z (ndc.x p32 / p00 - p02 / p00); y alike.
    // Pixels per unit at view z: |p11| (half height / 2) / |p32 z|.
    double ws = pj[14];
    struct { float ndc_to_view[4], size[4], radius[4], steps[4]; } pa = {
        {(float)(ws / pj[0]), (float)(-pj[2] / pj[0]), (float)(ws / pj[5]), (float)(-pj[6] / pj[5])},
        {(float)hw, (float)hh, (float)(fabs(pj[5]) * hh / (fabs(pj[0]) * hw)), 0},
        {AO_RADIUS, (float)(fabs(pj[5]) * 0.5 * hh / fabs(ws)), (float)hh / AO_REACH, AO_FADE},
        {A.level == 2 ? 3.0f : 2.0f, A.level == 2 ? 5.0f : 4.0f, 0, 0}};
    step(ex, 2, A.comp[1], z_in, A.im[1].view, &pa, sizeof pa, gx, gy);
    // 3: the light's typical brightness, then the pattern averaged out (image 1 into 2)
    step(ex, 0, A.comp[2], light_in, A.im[3].view, NULL, 0, 1, 1);
    blur(ex, 1, 2, cv, (float)sc);
    if (vk_gpuprof_on()) vk_gpuprof_mark(ex, 6, 3);
    // 4: scaled up and onto the light target
    struct { float inv2[4], inv3[4], size[4], mode[4]; } pf;
    memcpy(&pf, &pz, sizeof pz);
    pf.size[3] = AO_POWER;
    pf.mode[0] = (float)A.debug, pf.mode[1] = AO_OPEN, pf.mode[2] = (float)sc, pf.mode[3] = 0;
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = cv,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, ap);
    VkViewport v = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D s = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &v);
    vkCmdSetScissor(cb, 0, 1, &s);
    VkWriteDescriptorSet wa[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &a2_in},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &depth_in}};
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_GRAPHICS, A.lay_a, 0, 2, wa);
    vkCmdPushConstants(cb, A.lay_a, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pf, &pf);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    full_barrier(cb);   // what follows sees the darkened light
    ex->recorded += 30;
    vk_texture_gpu_written(t->tex);
    if (vk_gpuprof_on()) vk_gpuprof_mark(ex, 6, 4);
    vk_exec_draw_new_cb();   // the game's pipelines, sets and dynamic state were replaced: the next draw binds them again
    ex->dirty = VK_EXEC_DIRTY_ALL;
}

// ---------------------------------------------------------------- the frame
static int scene_color(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// A D32F(S8) depth at least 640 wide (the smallest scene resolution.c makes), as vk_dlaa.c scene_depth.
static int scene_depth(const VkExecTarget *z) {
    return z->image && z->width >= 640 && (z->format == VK_FORMAT_D32_SFLOAT_S8_UINT || z->format == VK_FORMAT_D32_SFLOAT);
}

void vk_ao_pass(VkExec *ex) {
    if (!A.level || A.done) return;
    const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
    int light = ex->pass_ncolor == 1 && c->image && scene_color(c->format) && scene_depth(z) && c->width == z->width &&
                c->height == z->height && c->samples <= 1 && !c->is3d && !c->level && !z->level;
    if (A.armed) {
        if (light && c->image == A.light.image && z->image == A.depth.image) return;   // the opaque pass goes on
        if (A.wrote) {
            run(ex);
            return;
        }
        A.armed = 0;   // that pass drew no colour: not the opaque lighting, which the pass beginning here may be
    }
    if (light) {
        if (A.have && A.lin && z->image == A.depth.image) {   // the opaque lighting, after the linear depth
            A.light = *c, A.armed = 1, A.wrote = 0;
            return;
        }
        if (!A.have || z->image != A.depth.image) memset(A.cams, 0, sizeof A.cams);   // another scene's camera
        A.light = *c, A.depth = *z, A.have = 1, A.lin = 0;
        return;
    }
    // the linear depth: one R32F target of the scene's size and no depth target
    if (A.have && ex->pass_ncolor == 1 && c->image && c->format == VK_FORMAT_R32_SFLOAT && !z->image &&
        c->width == A.depth.width && c->height == A.depth.height)
        A.lin = 1;
}

void vk_ao_flush(VkExec *ex) {
    if (A.armed && A.wrote && !A.done) run(ex);
}

// A draw about to be recorded. Two things are taken from it. Whether the armed pass writes colour: in play, every other
// frame, the game draws nine shapes with the colour writes masked off into the light target and the scene depth between
// the linear depth and the shadow map (cutscenes have no such pass); the opaque lighting comes after the shadow map, so
// the occlusion applied after the masked pass was drawn over one frame in two. And the camera block of a draw with the
// scene depth (vk_dlaa.c vk_dlaa_draw does the same for the upscaler): copied when first seen this frame from the
// uniform snapshot of the submission that draws with it, what the GPU reads.
void vk_ao_draw(VkExec *ex) {
    if (!A.level || A.done || !A.have) return;
    if (A.armed && !A.wrote && (ex->pipe.chmask.mask & 0xf) && ex->color[0].image == A.light.image &&
        ex->depth.image == A.depth.image)
        A.wrote = 1;
    const p_bind_buf *b = &ex->ubo[0][0];   // constant bank c3
    if (b->size != CAM_BYTES || !b->addr || ex->depth.image != A.depth.image) return;
    int i = 0, low = 0;
    for (; i < 4 && A.cams[i].addr && A.cams[i].addr != b->addr; i++)
        if (A.cams[i].n < A.cams[low].n) low = i;
    if (i == 4) i = low, A.cams[i].addr = 0;   // full: the rarest makes room
    struct Cam *cm = &A.cams[i];
    if (cm->addr != b->addr) {
        if (!vk_usnap_copy(b->addr, CAM_BYTES, cm->block)) memcpy(cm->block, (const void *)(uintptr_t)b->addr, CAM_BYTES);
        cm->addr = b->addr, cm->n = 0, cm->have = 1;
    }
    cm->n++;
}

void vk_ao_frame(void) {
    read_settings();
    A.have = A.lin = A.armed = A.done = A.wrote = 0;
    memset(A.cams, 0, sizeof A.cams);
}

void vk_ao_release(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    VkDevice d = c->device;
    free_images(0);
    for (int i = 0; i < A.napply; i++) vkDestroyPipeline(d, A.apply[i].p, NULL);
    for (int i = 0; i < 4; i++)
        if (A.comp[i]) vkDestroyPipeline(d, A.comp[i], NULL);
    if (A.lay_c) vkDestroyPipelineLayout(d, A.lay_c, NULL);
    if (A.lay_b) vkDestroyPipelineLayout(d, A.lay_b, NULL);
    if (A.dsl_b) vkDestroyDescriptorSetLayout(d, A.dsl_b, NULL);
    if (A.linear) vkDestroySampler(d, A.linear, NULL);
    if (A.lay_a) vkDestroyPipelineLayout(d, A.lay_a, NULL);
    if (A.dsl_c) vkDestroyDescriptorSetLayout(d, A.dsl_c, NULL);
    if (A.dsl_a) vkDestroyDescriptorSetLayout(d, A.dsl_a, NULL);
    if (A.nearest) vkDestroySampler(d, A.nearest, NULL);
    memset(&A, 0, sizeof A);
}
