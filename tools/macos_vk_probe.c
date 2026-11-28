// Standalone capability probe for the macOS port. No game files or runtime required.
// Exit 0: current runtime requirements present; 1: missing requirements; 2: probe error.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static int has(const VkExtensionProperties *p, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(p[i].extensionName, name)) return 1;
    return 0;
}

static int check(const char *name, int present) {
    printf("  %-48s %s\n", name, present ? "yes" : "MISSING");
    return !present;
}

int main(void) {
    const char *ext = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Astral Chain macOS capability probe", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app, .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = &ext};
    VkInstance instance;
    VkResult r = vkCreateInstance(&ci, NULL, &instance);
    if (r != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance: %d\n", r); return 2; }
    uint32_t n = 0;
    r = vkEnumeratePhysicalDevices(instance, &n, NULL);
    if (r != VK_SUCCESS || !n) { fprintf(stderr, "No Vulkan devices (%d)\n", r); vkDestroyInstance(instance, NULL); return 2; }
    VkPhysicalDevice *devices = calloc(n, sizeof *devices);
    if (!devices) { vkDestroyInstance(instance, NULL); return 2; }
    r = vkEnumeratePhysicalDevices(instance, &n, devices);
    int result = 1;
    if (r != VK_SUCCESS) { result = 2; goto done; }
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        printf("%s: Vulkan %u.%u.%u\n", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
               VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        uint32_t ne = 0;
        r = vkEnumerateDeviceExtensionProperties(devices[i], NULL, &ne, NULL);
        if (r != VK_SUCCESS) { result = 2; break; }
        VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
        if (!ep) { result = 2; break; }
        r = vkEnumerateDeviceExtensionProperties(devices[i], NULL, &ne, ep);
        if (r != VK_SUCCESS) { free(ep); result = 2; break; }
        int missing = check("Vulkan >= 1.3", p.apiVersion >= VK_API_VERSION_1_3);
        const char *required[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
        for (size_t j = 0; j < sizeof required / sizeof *required; j++)
            missing += check(required[j], has(ep, ne, required[j]));
        VkPhysicalDeviceDepthClipControlFeaturesEXT clip = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
        VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        void **tail = &f.pNext;
        if (p.apiVersion >= VK_API_VERSION_1_2) { *tail = &f12; tail = &f12.pNext; }
        if (p.apiVersion >= VK_API_VERSION_1_3) { *tail = &f13; tail = &f13.pNext; }
        if (has(ep, ne, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME)) *tail = &clip;
        vkGetPhysicalDeviceFeatures2(devices[i], &f);
        free(ep);
#define FEATURE(obj, name) missing += check(#name, obj.name)
        FEATURE(f12, timelineSemaphore);
        FEATURE(f13, dynamicRendering);
        FEATURE(f13, synchronization2);
        FEATURE(clip, depthClipControl);
        FEATURE(f.features, shaderStorageImageWriteWithoutFormat);
        FEATURE(f.features, shaderInt64);
        FEATURE(f.features, samplerAnisotropy);
        FEATURE(f.features, independentBlend);
        FEATURE(f.features, geometryShader);
        FEATURE(f.features, textureCompressionBC);
#undef FEATURE
        printf("  Current runtime requirements missing: %d\n", missing);
        if (!missing) result = 0;
    }
done:
    free(devices);
    vkDestroyInstance(instance, NULL);
    return result;
}
