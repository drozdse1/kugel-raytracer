// Kugel-Raytracer auf der AMD-GPU.
// Die CPU füllt Kamera, Licht, Boden und Kugeln. Eine Compute-Pipeline rechnet
// das Mandelbrot der Tischplatte, eine zweite wirft pro Pixel Strahl, Schatten
// und Spiegelung.

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <X11/keysym.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <vulkan/vulkan.h>

#include "mandelbrot_spv.h"
#include "trace_spv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kGroupX = 16;
constexpr uint32_t kGroupY = 8;
constexpr uint32_t kMandelSize = 1536;
constexpr uint32_t kMandelGroup = 16;
constexpr uint32_t kMaxSpheres = 8;
// Ein Strahl deckt block×block Fensterpixel. Pfeil runter macht die Blöcke größer.
constexpr uint32_t kRayBlocks[] = {1, 2, 3, 4, 6, 8};
constexpr int kRayBlockCount = static_cast<int>(sizeof(kRayBlocks) / sizeof(kRayBlocks[0]));

uint32_t ray_extent(uint32_t window_extent, uint32_t block) {
    uint32_t rays = (window_extent + block - 1) / block;
    if (rays < 1) {
        rays = 1;
    }
    if (rays > window_extent) {
        rays = window_extent;
    }
    return rays;
}

struct Vec3 {
    float x, y, z;
};

Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3 normalize(Vec3 a) {
    float length = std::sqrt(dot(a, a));
    return a * (1.0f / length);
}

struct Params {
    float cam_pos[4];
    float cam_forward[4];
    float cam_right[4];
    float cam_up[4];
    float light_dir[4];
    float light_color[4];
    float plane_point[4];
    float plane_normal[4];
    float plane_color[4];
    uint32_t width;
    uint32_t height;
    uint32_t sphere_count;
    uint32_t pad;
};

struct GpuSphere {
    float geom[4];
    float color[4];
    float material[4];  // x Metall 0..1, y Rauheit 0..1
};

struct MandelParams {
    float focus_zoom[4];  // xy Brennpunkt, z Zoom, w Farbphase
    float table_iter[4];  // rgb Plattenfarbe, w Iterationen
};

static_assert(sizeof(Params) == 160, "Params muss dem std140-Block im Shader entsprechen");
static_assert(sizeof(GpuSphere) == 48, "GpuSphere muss dem std430-Struct im Shader entsprechen");
static_assert(sizeof(MandelParams) == 32, "MandelParams muss dem std140-Block im Mandelbrot-Shader entsprechen");

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    float metallic;
    float roughness;
};

struct Scene {
    Vec3 cam_pos;
    Vec3 cam_target;
    float fov_y_degrees;
    Vec3 light_dir;
    Vec3 light_color;
    float ambient;
    Vec3 plane_point;
    Vec3 plane_normal;
    Vec3 plane_color;
    std::vector<Sphere> spheres;
};

Scene make_scene() {
    Scene scene;
    scene.cam_pos = {0.0f, 2.05f, 8.2f};
    scene.cam_target = {0.0f, 0.75f, 0.2f};
    scene.fov_y_degrees = 48.0f;
    scene.light_dir = normalize({0.75f, 1.0f, 0.35f});
    scene.light_color = {1.0f, 0.98f, 0.94f};
    scene.ambient = 0.08f;
    scene.plane_point = {0.0f, 0.0f, 0.0f};
    scene.plane_normal = {0.0f, 1.0f, 0.0f};
    scene.plane_color = {0.74f, 0.71f, 0.66f};
    // Von hartem Kunststoff bis fast Chrom. Metall 0 ist Kunststoff, 1 ist Metall.
    // Kleine Rauheit gibt einen scharfen Spiegel, große Rauheit einen breiten Glanz.
    scene.spheres = {
        {{0.0f, 1.0f, 0.0f}, 1.0f, {0.78f, 0.03f, 0.02f}, 0.0f, 0.05f},
        {{-1.75f, 0.55f, 0.85f}, 0.55f, {0.08f, 0.22f, 0.62f}, 0.0f, 0.28f},
        {{1.55f, 0.42f, 0.15f}, 0.42f, {1.0f, 0.76f, 0.30f}, 1.0f, 0.10f},
        {{0.65f, 0.28f, 1.85f}, 0.28f, {0.93f, 0.94f, 0.95f}, 1.0f, 0.02f},
    };
    return scene;
}

void put(float out[4], Vec3 v, float w) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = w;
}

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void vk_check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        fail(std::string(what) + " fehlgeschlagen (VkResult " + std::to_string(result) + ")");
    }
}

uint32_t find_memory(VkPhysicalDevice device, uint32_t type_bits, VkMemoryPropertyFlags need,
                      VkMemoryPropertyFlags avoid = 0) {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) {
            continue;
        }
        VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
        if ((flags & need) != need) {
            continue;
        }
        if (avoid != 0 && (flags & avoid) != 0) {
            if (fallback == UINT32_MAX) {
                fallback = i;
            }
            continue;
        }
        return i;
    }
    if (fallback != UINT32_MAX) {
        return fallback;
    }
    fail("kein passender Vulkan-Speichertyp");
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

Buffer make_buffer(VkDevice device, VkPhysicalDevice physical, VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags memory_flags, VkMemoryPropertyFlags avoid = 0) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &info, nullptr, &buffer.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = find_memory(physical, requirements.memoryTypeBits, memory_flags, avoid);
    vk_check(vkAllocateMemory(device, &alloc, nullptr, &buffer.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");
    return buffer;
}

void destroy_buffer(VkDevice device, Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

void upload(VkDevice device, const Buffer& buffer, const void* data, size_t bytes) {
    void* mapped = nullptr;
    vk_check(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped), "vkMapMemory");
    std::memset(mapped, 0, static_cast<size_t>(buffer.size));
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(device, buffer.memory);
}

struct Gpu {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buffer pixels;
    Buffer staging;
    Buffer params;
    Buffer spheres;
    VkDescriptorSetLayout mandel_layout = VK_NULL_HANDLE;
    VkPipelineLayout mandel_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline mandel_pipeline = VK_NULL_HANDLE;
    VkShaderModule mandel_shader = VK_NULL_HANDLE;
    VkDescriptorSet mandel_set = VK_NULL_HANDLE;
    VkImage mandel_image = VK_NULL_HANDLE;
    VkDeviceMemory mandel_memory = VK_NULL_HANDLE;
    VkImageView mandel_view = VK_NULL_HANDLE;
    VkSampler mandel_sampler = VK_NULL_HANDLE;
    Buffer mandel_params;
    bool mandel_ready = false;

    ~Gpu() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDescriptorPool(device, pool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, shader, nullptr);
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            vkDestroyPipeline(device, mandel_pipeline, nullptr);
            vkDestroyShaderModule(device, mandel_shader, nullptr);
            vkDestroyPipelineLayout(device, mandel_pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, mandel_layout, nullptr);
            vkDestroySampler(device, mandel_sampler, nullptr);
            vkDestroyImageView(device, mandel_view, nullptr);
            vkDestroyImage(device, mandel_image, nullptr);
            vkFreeMemory(device, mandel_memory, nullptr);
            destroy_buffer(device, pixels);
            destroy_buffer(device, staging);
            destroy_buffer(device, params);
            destroy_buffer(device, spheres);
            destroy_buffer(device, mandel_params);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;
    Gpu() = default;
};

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "diskret";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integriert";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtuell";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
    default: return "andere";
    }
}

VkDeviceSize local_memory(VkPhysicalDevice device) {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    VkDeviceSize bytes = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            bytes += memory.memoryHeaps[i].size;
        }
    }
    return bytes;
}

void create_device(Gpu& gpu) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "kugel-raytracer";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    vk_check(vkCreateInstance(&instance_info, nullptr, &gpu.instance), "vkCreateInstance");

    uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(gpu.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    if (count == 0) {
        fail("kein Vulkan-Gerät gefunden");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vk_check(vkEnumeratePhysicalDevices(gpu.instance, &count, devices.data()), "vkEnumeratePhysicalDevices");

    struct Choice {
        VkPhysicalDevice device;
        VkPhysicalDeviceProperties props;
        VkDeviceSize local;
    };
    std::vector<Choice> choices;
    std::cout << "Vulkan-Geräte:\n";
    for (VkPhysicalDevice device : devices) {
        Choice choice;
        choice.device = device;
        vkGetPhysicalDeviceProperties(device, &choice.props);
        choice.local = local_memory(device);
        choices.push_back(choice);
        std::cout << "  " << choice.props.deviceName << " (" << device_type_name(choice.props.deviceType)
                  << ", " << (choice.local / (1024 * 1024)) << " MiB)\n";
    }

    auto better = [](const Choice& candidate, const Choice& current) {
        std::string name = candidate.props.deviceName;
        std::string current_name = current.props.deviceName;
        bool candidate_is_target = name.find("9070") != std::string::npos;
        bool current_is_target = current_name.find("9070") != std::string::npos;
        if (candidate_is_target != current_is_target) {
            return candidate_is_target;
        }
        bool candidate_discrete = candidate.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        bool current_discrete = current.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (candidate_discrete != current_discrete) {
            return candidate_discrete;
        }
        return candidate.local > current.local;
    };

    Choice selected = choices[0];
    for (const Choice& choice : choices) {
        if (better(choice, selected)) {
            selected = choice;
        }
    }
    gpu.physical = selected.device;
    std::cout << "Gewählt: " << selected.props.deviceName << "\n";

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &family_count, families.data());
    gpu.queue_family = family_count;
    for (uint32_t i = 0; i < family_count; ++i) {
        VkQueueFlags flags = families[i].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) && (flags & VK_QUEUE_TRANSFER_BIT)) {
            gpu.queue_family = i;
            break;
        }
    }
    if (gpu.queue_family == family_count) {
        fail("keine Compute-Queue");
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = gpu.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    vk_check(vkCreateDevice(gpu.physical, &device_info, nullptr, &gpu.device), "vkCreateDevice");
    vkGetDeviceQueue(gpu.device, gpu.queue_family, 0, &gpu.queue);
}

void create_mandel(Gpu& gpu) {
    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormatProperties format_props{};
    vkGetPhysicalDeviceFormatProperties(gpu.physical, format, &format_props);
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((format_props.optimalTilingFeatures & need) != need) {
        fail("das Gerät kann die Mandelbrot-Textur nicht als Bild speichern");
    }

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {kMandelSize, kMandelSize, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vk_check(vkCreateImage(gpu.device, &image_info, nullptr, &gpu.mandel_image), "vkCreateImage");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(gpu.device, gpu.mandel_image, &requirements);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = find_memory(gpu.physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_check(vkAllocateMemory(gpu.device, &alloc, nullptr, &gpu.mandel_memory), "vkAllocateMemory");
    vk_check(vkBindImageMemory(gpu.device, gpu.mandel_image, gpu.mandel_memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = gpu.mandel_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(gpu.device, &view_info, nullptr, &gpu.mandel_view), "vkCreateImageView");

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(gpu.device, &sampler_info, nullptr, &gpu.mandel_sampler), "vkCreateSampler");

    gpu.mandel_params = make_buffer(gpu.device, gpu.physical, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;
    vk_check(vkCreateDescriptorSetLayout(gpu.device, &layout_info, nullptr, &gpu.mandel_layout),
             "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &gpu.mandel_layout;
    vk_check(vkCreatePipelineLayout(gpu.device, &pipeline_layout_info, nullptr, &gpu.mandel_pipeline_layout),
             "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = kMandelSpvWords * sizeof(uint32_t);
    shader_info.pCode = kMandelSpv;
    vk_check(vkCreateShaderModule(gpu.device, &shader_info, nullptr, &gpu.mandel_shader), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = gpu.mandel_shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = gpu.mandel_pipeline_layout;
    vk_check(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &gpu.mandel_pipeline),
             "vkCreateComputePipelines");
    std::cout << "Mandelbrot-Pipeline: " << kMandelSize << "x" << kMandelSize << ", jede Frame.\n";
}

void create_pipeline(Gpu& gpu, const std::vector<GpuSphere>& spheres, const Params& params) {
    VkDescriptorSetLayoutBinding bindings[4]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 4;
    layout_info.pBindings = bindings;
    vk_check(vkCreateDescriptorSetLayout(gpu.device, &layout_info, nullptr, &gpu.set_layout),
             "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &gpu.set_layout;
    vk_check(vkCreatePipelineLayout(gpu.device, &pipeline_layout_info, nullptr, &gpu.pipeline_layout),
             "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = kTraceSpvWords * sizeof(uint32_t);
    shader_info.pCode = kTraceSpv;
    vk_check(vkCreateShaderModule(gpu.device, &shader_info, nullptr, &gpu.shader), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = gpu.shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = gpu.pipeline_layout;
    vk_check(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &gpu.pipeline),
             "vkCreateComputePipelines");

    VkDeviceSize pixel_bytes = static_cast<VkDeviceSize>(params.width) * params.height * sizeof(float) * 4;
    gpu.pixels = make_buffer(gpu.device, gpu.physical, pixel_bytes,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Gecachter Systemspeicher. Lesen aus VRAM oder ungecachetem Speicher kostet hier sonst den Frame.
    gpu.staging = make_buffer(gpu.device, gpu.physical, pixel_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu.params = make_buffer(gpu.device, gpu.physical, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    gpu.spheres = make_buffer(gpu.device, gpu.physical, sizeof(GpuSphere) * spheres.size(),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    upload(gpu.device, gpu.params, &params, sizeof(params));
    upload(gpu.device, gpu.spheres, spheres.data(), sizeof(GpuSphere) * spheres.size());
    create_mandel(gpu);

    VkDescriptorPoolSize pool_sizes[4]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 2;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 2;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[2].descriptorCount = 1;
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[3].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 2;
    pool_info.poolSizeCount = 4;
    pool_info.pPoolSizes = pool_sizes;
    vk_check(vkCreateDescriptorPool(gpu.device, &pool_info, nullptr, &gpu.pool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = gpu.pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &gpu.set_layout;
    vk_check(vkAllocateDescriptorSets(gpu.device, &alloc, &gpu.set), "vkAllocateDescriptorSets");

    alloc.pSetLayouts = &gpu.mandel_layout;
    vk_check(vkAllocateDescriptorSets(gpu.device, &alloc, &gpu.mandel_set), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo infos[3]{};
    infos[0] = {gpu.pixels.buffer, 0, VK_WHOLE_SIZE};
    infos[1] = {gpu.params.buffer, 0, 256};
    infos[2] = {gpu.spheres.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo floor_info{};
    floor_info.sampler = gpu.mandel_sampler;
    floor_info.imageView = gpu.mandel_view;
    floor_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = gpu.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = gpu.set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &floor_info;

    VkDescriptorImageInfo storage_info{};
    storage_info.imageView = gpu.mandel_view;
    storage_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo mandel_uniform{gpu.mandel_params.buffer, 0, 256};
    VkWriteDescriptorSet mandel_writes[2]{};
    mandel_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    mandel_writes[0].dstSet = gpu.mandel_set;
    mandel_writes[0].dstBinding = 0;
    mandel_writes[0].descriptorCount = 1;
    mandel_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    mandel_writes[0].pImageInfo = &storage_info;
    mandel_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    mandel_writes[1].dstSet = gpu.mandel_set;
    mandel_writes[1].dstBinding = 1;
    mandel_writes[1].descriptorCount = 1;
    mandel_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    mandel_writes[1].pBufferInfo = &mandel_uniform;
    vkUpdateDescriptorSets(gpu.device, 4, writes, 0, nullptr);
    vkUpdateDescriptorSets(gpu.device, 2, mandel_writes, 0, nullptr);

    VkCommandPoolCreateInfo pool_create{};
    pool_create.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_create.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_create.queueFamilyIndex = gpu.queue_family;
    vk_check(vkCreateCommandPool(gpu.device, &pool_create, nullptr, &gpu.command_pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = gpu.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(gpu.device, &command_info, &gpu.command), "vkAllocateCommandBuffers");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(vkCreateFence(gpu.device, &fence_info, nullptr, &gpu.fence), "vkCreateFence");
}

void replace_pixel_targets(Gpu& gpu, uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(gpu.device);
    destroy_buffer(gpu.device, gpu.pixels);
    destroy_buffer(gpu.device, gpu.staging);
    VkDeviceSize pixel_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(float) * 4;
    gpu.pixels = make_buffer(gpu.device, gpu.physical, pixel_bytes,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu.staging = make_buffer(gpu.device, gpu.physical, pixel_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDescriptorBufferInfo info{gpu.pixels.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = gpu.set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(gpu.device, 1, &write, 0, nullptr);
}

void upload_mandel(Gpu& gpu, Vec3 table, float time) {
    // Seepferdchen-Tal, knapp außerhalb des Randes. Tief genug, dass das
    // schwarze Innere aus dem Bild ist und die Filamente die Platte füllen.
    constexpr float kFocusX = -0.7464f;
    constexpr float kFocusY = 0.1342f;
    float pulse = 0.5f - 0.5f * std::cos(time * 0.18f);
    float zoom = 140.0f + 420.0f * pulse;
    MandelParams mandel{};
    mandel.focus_zoom[0] = kFocusX;
    mandel.focus_zoom[1] = kFocusY;
    mandel.focus_zoom[2] = zoom;
    mandel.focus_zoom[3] = std::fmod(time * 0.35f, 6.2831853f);
    mandel.table_iter[0] = table.x;
    mandel.table_iter[1] = table.y;
    mandel.table_iter[2] = table.z;
    mandel.table_iter[3] = 160.0f + 80.0f * std::log2(zoom);
    upload(gpu.device, gpu.mandel_params, &mandel, sizeof(mandel));
}

void dispatch(Gpu& gpu, uint32_t width, uint32_t height) {
    vk_check(vkResetFences(gpu.device, 1, &gpu.fence), "vkResetFences");
    vk_check(vkResetCommandBuffer(gpu.command, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(gpu.command, &begin), "vkBeginCommandBuffer");

    VkBufferMemoryBarrier host[3]{};
    host[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    host[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host[0].dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    host[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host[0].buffer = gpu.params.buffer;
    host[0].size = VK_WHOLE_SIZE;
    host[1] = host[0];
    host[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    host[1].buffer = gpu.spheres.buffer;
    host[2] = host[0];
    host[2].buffer = gpu.mandel_params.buffer;
    VkImageMemoryBarrier image_barrier{};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = gpu.mandel_image;
    image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_barrier.subresourceRange.levelCount = 1;
    image_barrier.subresourceRange.layerCount = 1;
    if (!gpu.mandel_ready) {
        image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(gpu.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &image_barrier);
        gpu.mandel_ready = true;
    }
    vkCmdPipelineBarrier(gpu.command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         3, host, 0, nullptr);

    vkCmdBindPipeline(gpu.command, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.mandel_pipeline);
    vkCmdBindDescriptorSets(gpu.command, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.mandel_pipeline_layout, 0, 1,
                            &gpu.mandel_set, 0, nullptr);
    vkCmdDispatch(gpu.command, kMandelSize / kMandelGroup, kMandelSize / kMandelGroup, 1);

    image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(gpu.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_barrier);

    vkCmdBindPipeline(gpu.command, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline);
    vkCmdBindDescriptorSets(gpu.command, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline_layout, 0, 1, &gpu.set, 0,
                            nullptr);
    uint32_t groups_x = (width + kGroupX - 1) / kGroupX;
    uint32_t groups_y = (height + kGroupY - 1) / kGroupY;
    vkCmdDispatch(gpu.command, groups_x, groups_y, 1);

    VkBufferMemoryBarrier to_copy{};
    to_copy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.buffer = gpu.pixels.buffer;
    to_copy.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(gpu.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &to_copy, 0, nullptr);

    VkBufferCopy region{};
    region.size = gpu.pixels.size;
    vkCmdCopyBuffer(gpu.command, gpu.pixels.buffer, gpu.staging.buffer, 1, &region);

    VkBufferMemoryBarrier to_host{};
    to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer = gpu.staging.buffer;
    to_host.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(gpu.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                         1, &to_host, 0, nullptr);
    vk_check(vkEndCommandBuffer(gpu.command), "vkEndCommandBuffer");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &gpu.command;
    vk_check(vkQueueSubmit(gpu.queue, 1, &submit, gpu.fence), "vkQueueSubmit");
    VkResult wait = vkWaitForFences(gpu.device, 1, &gpu.fence, VK_TRUE, 10'000'000'000ull);
    if (wait == VK_TIMEOUT) {
        fail("GPU-Dispatch hat nach 10 Sekunden nicht geendet");
    }
    vk_check(wait, "vkWaitForFences");
}

struct RgbImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;
};

RgbImage read_image(Gpu& gpu, uint32_t width, uint32_t height) {
    void* mapped = nullptr;
    vk_check(vkMapMemory(gpu.device, gpu.staging.memory, 0, gpu.staging.size, 0, &mapped), "vkMapMemory");
    const auto* pixels = static_cast<const float*>(mapped);
    RgbImage image;
    image.width = width;
    image.height = height;
    image.rgb.resize(static_cast<size_t>(width) * height * 3);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t src = (static_cast<size_t>(y) * width + x) * 4;
            size_t dst = (static_cast<size_t>(y) * width + x) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                float value = std::clamp(pixels[src + channel], 0.0f, 1.0f);
                image.rgb[dst + channel] = static_cast<uint8_t>(std::lround(value * 255.0f));
            }
        }
    }
    vkUnmapMemory(gpu.device, gpu.staging.memory);
    return image;
}

void write_ppm(const RgbImage& image, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("kann " + path + " nicht schreiben");
    }
    out << "P6\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.rgb.data()), static_cast<std::streamsize>(image.rgb.size()));
    if (!out) {
        fail("Schreiben von " + path + " unvollständig");
    }
}

void report(const RgbImage& image) {
    int sky = 0;
    int floor_lit = 0;
    int shadow = 0;
    int red = 0;
    int blue = 0;
    int gold = 0;
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            size_t i = (static_cast<size_t>(y) * image.width + x) * 3;
            int r = image.rgb[i];
            int g = image.rgb[i + 1];
            int b = image.rgb[i + 2];
            int max_channel = std::max(r, std::max(g, b));
            int min_channel = std::min(r, std::min(g, b));
            bool gray = max_channel - min_channel < 18;
            if (y < image.height / 3 && b > r + 12 && b > 90) {
                ++sky;
            }
            if (gray && y > image.height / 2 && max_channel > 150) {
                ++floor_lit;
            }
            if (gray && y > image.height / 2 && max_channel < 110 && max_channel > 25) {
                ++shadow;
            }
            if (r > 150 && r > g + 50 && r > b + 50) {
                ++red;
            }
            if (y > image.height / 2 && b > 140 && b > r + 40 && b > g + 20) {
                ++blue;
            }
            if (r > 170 && g > 120 && b < 90) {
                ++gold;
            }
        }
    }
    std::cout << "Kontrolle: Himmel " << sky << ", Boden hell " << floor_lit << ", Schatten " << shadow
              << ", rot " << red << ", blau " << blue << ", gold " << gold << "\n";
}

Vec3 group_pivot(const std::vector<Sphere>& spheres) {
    Vec3 pivot{0.0f, 0.0f, 0.0f};
    for (const Sphere& sphere : spheres) {
        pivot.x += sphere.center.x;
        pivot.z += sphere.center.z;
    }
    float count = static_cast<float>(spheres.size());
    pivot.x /= count;
    pivot.z /= count;
    return pivot;
}

Vec3 light_direction(float time) {
    // Azimut und Höhe auf zwei überlagerten Ellipsen. Die Höhe bleibt positiv,
    // der Schatten wandert dadurch deutlich um die Kugeln.
    float turn = time * 0.80f;
    float azimuth = 0.50f + 1.15f * std::sin(turn) + 0.40f * std::sin(turn * 2.0f);
    float elevation = 0.98f + 0.34f * std::cos(turn) + 0.14f * std::cos(turn * 2.0f);
    return {std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)};
}

void rotate_spheres(const std::vector<Sphere>& base, Vec3 pivot, float angle, std::vector<GpuSphere>& out) {
    float cosine = std::cos(angle);
    float sine = std::sin(angle);
    for (size_t i = 0; i < base.size(); ++i) {
        float x = base[i].center.x - pivot.x;
        float z = base[i].center.z - pivot.z;
        Vec3 center{x * cosine - z * sine + pivot.x, base[i].center.y, x * sine + z * cosine + pivot.z};
        put(out[i].geom, center, base[i].radius);
        put(out[i].color, base[i].color, 0.0f);
        out[i].material[0] = base[i].metallic;
        out[i].material[1] = base[i].roughness;
        out[i].material[2] = 0.0f;
        out[i].material[3] = 0.0f;
    }
}

int mask_shift(unsigned long mask) {
    int shift = 0;
    while (mask != 0 && (mask & 1ul) == 0) {
        mask >>= 1;
        ++shift;
    }
    return shift;
}

struct LiveView {
    Display* display = nullptr;
    Window window = 0;
    GC gc = nullptr;
    XImage* image = nullptr;
    XShmSegmentInfo shm{};
    bool use_shm = false;
    Atom wm_delete = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t requested_width = 0;
    uint32_t requested_height = 0;
    bool size_dirty = false;
    int ray_block_index = 0;
    bool quality_dirty = false;
    unsigned long last_quality_ms = 0;
    int red_shift = 0;
    int green_shift = 0;
    int blue_shift = 0;

    LiveView(const LiveView&) = delete;
    LiveView& operator=(const LiveView&) = delete;

    explicit LiveView(uint32_t view_width, uint32_t view_height) : width(view_width), height(view_height) {
        display = XOpenDisplay(nullptr);
        if (display == nullptr) {
            fail("kein X11-Display. Unter Wayland muss XWayland laufen");
        }
        int screen = DefaultScreen(display);
        unsigned long black = BlackPixel(display, screen);
        window = XCreateSimpleWindow(display, RootWindow(display, screen), 80, 80, width, height, 0, black, black);
        show_rays(width, height);
        XSelectInput(display, window, ExposureMask | KeyPressMask | StructureNotifyMask);
        wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display, window, &wm_delete, 1);

        XSizeHints* hints = XAllocSizeHints();
        if (hints != nullptr) {
            hints->flags = PMinSize | PMaxSize;
            hints->min_width = 16;
            hints->min_height = 16;
            hints->max_width = 8192;
            hints->max_height = 8192;
            XSetWMNormalHints(display, window, hints);
            XFree(hints);
        }
        XMapWindow(display, window);
        gc = XCreateGC(display, window, 0, nullptr);
        XSync(display, False);
        create_image();
        std::cout << (use_shm ? "Ausgabe über Shared Memory." : "Ausgabe über XPutImage.") << std::endl;
    }

    ~LiveView() {
        if (display == nullptr) {
            return;
        }
        destroy_image();
        if (gc != nullptr) {
            XFreeGC(display, gc);
        }
        if (window != 0) {
            XDestroyWindow(display, window);
        }
        XCloseDisplay(display);
    }

    void resize(uint32_t next_width, uint32_t next_height) {
        width = next_width;
        height = next_height;
        destroy_image();
        create_image();
    }

    bool pump() {
        while (XPending(display) > 0) {
            XEvent event;
            XNextEvent(display, &event);
            if (event.type == ClientMessage && static_cast<Atom>(event.xclient.data.l[0]) == wm_delete) {
                return false;
            }
            if (event.type == ConfigureNotify && event.xconfigure.window == window) {
                int next_width = std::clamp(event.xconfigure.width, 16, 8192);
                int next_height = std::clamp(event.xconfigure.height, 16, 8192);
                requested_width = static_cast<uint32_t>(next_width);
                requested_height = static_cast<uint32_t>(next_height);
                size_dirty = requested_width != width || requested_height != height;
            }
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_Escape || key == XK_q || key == XK_Q) {
                    return false;
                }
                bool up = key == XK_Up || key == XK_KP_Up;
                bool down = key == XK_Down || key == XK_KP_Down;
                if ((up || down) && event.xkey.time - last_quality_ms >= 160) {
                    last_quality_ms = event.xkey.time;
                    if (up && ray_block_index > 0) {
                        --ray_block_index;
                        quality_dirty = true;
                    } else if (down && ray_block_index + 1 < kRayBlockCount) {
                        ++ray_block_index;
                        quality_dirty = true;
                    }
                }
            }
        }
        return true;
    }

    bool take_resize(uint32_t& out_width, uint32_t& out_height) {
        if (!size_dirty) {
            return false;
        }
        size_dirty = false;
        if (requested_width == width && requested_height == height) {
            return false;
        }
        out_width = requested_width;
        out_height = requested_height;
        return true;
    }

    uint32_t ray_block() const {
        return kRayBlocks[ray_block_index];
    }

    bool take_quality() {
        if (!quality_dirty) {
            return false;
        }
        quality_dirty = false;
        return true;
    }

    void show_rays(uint32_t render_width, uint32_t render_height) {
        uint32_t block = ray_block();
        std::string title = "Kugel-Raytracer — " + std::to_string(render_width) + "x" +
                            std::to_string(render_height) + " Strahlen, 1 pro " + std::to_string(block) + "x" +
                            std::to_string(block) + " Pixel";
        XStoreName(display, window, title.c_str());
    }

    void destroy_image() {
        if (image == nullptr) {
            return;
        }
        if (use_shm) {
            XShmDetach(display, &shm);
            XSync(display, False);
            char* address = image->data;
            image->data = nullptr;
            XDestroyImage(image);
            if (address != nullptr && address != reinterpret_cast<char*>(-1)) {
                shmdt(address);
            }
        } else {
            XDestroyImage(image);
        }
        image = nullptr;
        use_shm = false;
        shm = {};
        shm.shmid = -1;
    }

    void create_image() {
        int screen = DefaultScreen(display);
        int depth = DefaultDepth(display, screen);
        Visual* visual = DefaultVisual(display, screen);
        shm = {};
        shm.shmid = -1;
        use_shm = false;
        if (XShmQueryExtension(display) != 0) {
            image = XShmCreateImage(display, visual, static_cast<unsigned int>(depth), ZPixmap, nullptr, &shm, width,
                                    height);
            if (image != nullptr && image->bits_per_pixel == 32) {
                size_t bytes = static_cast<size_t>(image->bytes_per_line) * height;
                shm.shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
                if (shm.shmid >= 0) {
                    shm.shmaddr = static_cast<char*>(shmat(shm.shmid, nullptr, 0));
                    shm.readOnly = False;
                    image->data = shm.shmaddr;
                    if (shm.shmaddr != reinterpret_cast<char*>(-1) && XShmAttach(display, &shm) != 0) {
                        shmctl(shm.shmid, IPC_RMID, nullptr);
                        use_shm = true;
                    }
                }
            }
        }
        if (!use_shm) {
            if (image != nullptr) {
                image->data = nullptr;
                XDestroyImage(image);
                if (shm.shmaddr != nullptr && shm.shmaddr != reinterpret_cast<char*>(-1)) {
                    shmdt(shm.shmaddr);
                }
                if (shm.shmid >= 0) {
                    shmctl(shm.shmid, IPC_RMID, nullptr);
                }
                image = nullptr;
                shm = {};
                shm.shmid = -1;
            }
            image = XCreateImage(display, visual, static_cast<unsigned int>(depth), ZPixmap, 0, nullptr, width, height,
                                 32, 0);
            if (image == nullptr || image->bits_per_pixel != 32) {
                fail("das Display erwartet kein 32-Bit-Bild");
            }
            image->data = static_cast<char*>(std::malloc(static_cast<size_t>(image->bytes_per_line) * height));
            if (image->data == nullptr) {
                fail("kein Speicher für das Fensterbild");
            }
        }
        std::memset(image->data, 0, static_cast<size_t>(image->bytes_per_line) * height);
        red_shift = mask_shift(image->red_mask);
        green_shift = mask_shift(image->green_mask);
        blue_shift = mask_shift(image->blue_mask);
    }

    void present(const float* pixels, uint32_t source_width, uint32_t source_height) {
        auto* rows = reinterpret_cast<uint8_t*>(image->data);
        auto pack = [this](const float* sample) {
            auto channel = [](float value) {
                value = std::clamp(value, 0.0f, 1.0f);
                return static_cast<uint32_t>(value * 255.0f + 0.5f);
            };
            return (channel(sample[0]) << red_shift) | (channel(sample[1]) << green_shift) |
                   (channel(sample[2]) << blue_shift);
        };
        if (source_width == width && source_height == height) {
            for (uint32_t y = 0; y < height; ++y) {
                auto* row = reinterpret_cast<uint32_t*>(rows + static_cast<size_t>(y) * image->bytes_per_line);
                const float* sample = pixels + static_cast<size_t>(y) * width * 4;
                for (uint32_t x = 0; x < width; ++x, sample += 4) {
                    row[x] = pack(sample);
                }
            }
        } else {
            // Jeder gerechnete Strahl füllt sein Pixelrechteck. Weniger Strahlen, gleiche Fenstergröße.
            for (uint32_t sy = 0; sy < source_height; ++sy) {
                uint32_t y0 = sy * height / source_height;
                uint32_t y1 = (sy + 1) * height / source_height;
                const float* sample = pixels + static_cast<size_t>(sy) * source_width * 4;
                for (uint32_t sx = 0; sx < source_width; ++sx, sample += 4) {
                    uint32_t packed = pack(sample);
                    uint32_t x0 = sx * width / source_width;
                    uint32_t x1 = (sx + 1) * width / source_width;
                    for (uint32_t y = y0; y < y1; ++y) {
                        auto* row = reinterpret_cast<uint32_t*>(rows + static_cast<size_t>(y) * image->bytes_per_line);
                        for (uint32_t x = x0; x < x1; ++x) {
                            row[x] = packed;
                        }
                    }
                }
            }
        }
        if (use_shm) {
            XShmPutImage(display, window, gc, image, 0, 0, 0, 0, width, height, False);
            XSync(display, False);
        } else {
            XPutImage(display, window, gc, image, 0, 0, 0, 0, width, height);
            XFlush(display);
        }
    }
};

// Kreis um den Blickpunkt, die Höhe ist eine Sinuswelle mit fünf Bögen: eine Krone.
void place_camera(Params& params, float angle) {
    constexpr float kRadius = 8.0f;
    constexpr float kHeight = 2.05f;
    constexpr float kCrown = 1.0f;
    constexpr float kPoints = 5.0f;
    Vec3 target{0.0f, 0.75f, 0.2f};
    float height = kHeight + kCrown * std::sin(kPoints * angle);
    Vec3 pos{target.x + kRadius * std::sin(angle), height, target.z + kRadius * std::cos(angle)};
    Vec3 forward = normalize(target - pos);
    Vec3 right = normalize(cross(forward, Vec3{0.0f, 1.0f, 0.0f}));
    Vec3 up = cross(right, forward);
    put(params.cam_pos, pos, 1.0f);
    put(params.cam_forward, forward, 0.0f);
    put(params.cam_right, right, 0.0f);
    put(params.cam_up, up, 0.0f);
}

void usage() {
    std::cout << "Verwendung: kugel-raytracer [breite] [hoehe]\n"
              << "           kugel-raytracer --bild ausgabe.ppm [breite] [hoehe]\n"
              << "Ohne --bild öffnet sich ein Fenster. Die Größe lässt sich ändern. "
                 "Pfeil hoch und runter ändert die Strahlzahl. Escape oder Q beendet es.\n";
}

Params make_params(const Scene& scene, uint32_t width, uint32_t height) {
    Vec3 forward = normalize(scene.cam_target - scene.cam_pos);
    Vec3 right = normalize(cross(forward, Vec3{0.0f, 1.0f, 0.0f}));
    Vec3 up = cross(right, forward);
    float tan_half = std::tan(scene.fov_y_degrees * 0.5f * 3.14159265f / 180.0f);

    Params params{};
    put(params.cam_pos, scene.cam_pos, 1.0f);
    put(params.cam_forward, forward, 0.0f);
    put(params.cam_right, right, 0.0f);
    put(params.cam_up, up, 0.0f);
    put(params.light_dir, scene.light_dir, tan_half);
    put(params.light_color, scene.light_color, scene.ambient);
    put(params.plane_point, scene.plane_point, 0.0f);
    put(params.plane_normal, scene.plane_normal, 0.0f);
    put(params.plane_color, scene.plane_color, 1.0f);
    params.width = width;
    params.height = height;
    params.sphere_count = static_cast<uint32_t>(scene.spheres.size());
    return params;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        uint32_t width = 1280;
        uint32_t height = 720;
        bool still = false;
        std::string still_path;
        int index = 1;
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            usage();
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--bild") {
            still = true;
            if (argc < 3) {
                usage();
                return 1;
            }
            still_path = argv[2];
            index = 3;
        }
        if (argc > index) {
            width = static_cast<uint32_t>(std::stoul(argv[index]));
        }
        if (argc > index + 1) {
            height = static_cast<uint32_t>(std::stoul(argv[index + 1]));
        }
        if (width < 16 || height < 16 || width > 8192 || height > 8192) {
            fail("Auflösung muss zwischen 16 und 8192 liegen");
        }

        Scene scene = make_scene();
        if (scene.spheres.empty() || scene.spheres.size() > kMaxSpheres) {
            fail("eine bis acht Kugeln");
        }
        Params params = make_params(scene, width, height);
        Vec3 pivot = group_pivot(scene.spheres);
        std::vector<GpuSphere> spheres(scene.spheres.size());
        rotate_spheres(scene.spheres, pivot, 0.0f, spheres);

        Gpu gpu;
        create_device(gpu);
        create_pipeline(gpu, spheres, params);

        if (still) {
            upload_mandel(gpu, scene.plane_color, 0.0f);
            dispatch(gpu, width, height);
            RgbImage image = read_image(gpu, width, height);
            write_ppm(image, still_path);
            report(image);
            std::cout << "Geschrieben: " << still_path << "\n";
            return 0;
        }

        LiveView view(width, height);
        uint32_t render_width = width;
        uint32_t render_height = height;
        void* mapped = nullptr;
        vk_check(vkMapMemory(gpu.device, gpu.staging.memory, 0, gpu.staging.size, 0, &mapped), "vkMapMemory");
        const auto* pixels = static_cast<const float*>(mapped);

        auto retarget = [&]() {
            uint32_t rw = ray_extent(width, view.ray_block());
            uint32_t rh = ray_extent(height, view.ray_block());
            if (rw != render_width || rh != render_height) {
                vkUnmapMemory(gpu.device, gpu.staging.memory);
                replace_pixel_targets(gpu, rw, rh);
                mapped = nullptr;
                vk_check(vkMapMemory(gpu.device, gpu.staging.memory, 0, gpu.staging.size, 0, &mapped), "vkMapMemory");
                pixels = static_cast<const float*>(mapped);
                render_width = rw;
                render_height = rh;
                params.width = rw;
                params.height = rh;
            }
            view.show_rays(render_width, render_height);
        };

        std::cout << "Fenster offen, " << width << "x" << height
                  << ". Die Größe lässt sich ändern. Pfeil runter rechnet weniger Strahlen, "
                     "das Bild wird gröber und schneller, Pfeil hoch macht es wieder feiner. "
                     "Die Kamera kreist auf einer Krone, die Kugeln drehen und "
                     "spiegeln, das Mandelbrot wandert, das Licht wandert auf Ellipsen. "
                     "Beenden mit Escape oder Q."
                  << std::endl;
        using clock = std::chrono::steady_clock;
        auto previous = clock::now();
        auto fps_mark = previous;
        float angle = 0.0f;
        float light_time = 0.0f;
        int frames = 0;
        constexpr float kRadiansPerSecond = 0.7f;
        constexpr auto kFrame = std::chrono::milliseconds(16);

        while (view.pump()) {
            uint32_t next_width = 0;
            uint32_t next_height = 0;
            bool resized = view.take_resize(next_width, next_height);
            bool quality = view.take_quality();
            if (resized) {
                view.resize(next_width, next_height);
                width = next_width;
                height = next_height;
            }
            if (resized || quality) {
                retarget();
                if (quality) {
                    std::cout << "Strahlen: 1 pro " << view.ray_block() << "x" << view.ray_block()
                              << " Pixel, gerechnet " << render_width << "x" << render_height << std::endl;
                }
            }
            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - previous).count();
            previous = now;
            if (dt > 0.1f) {
                dt = 0.1f;
            }
            angle += dt * kRadiansPerSecond;
            light_time += dt;
            rotate_spheres(scene.spheres, pivot, angle, spheres);
            place_camera(params, light_time * 0.32f);
            float tan_half = params.light_dir[3];
            put(params.light_dir, light_direction(light_time), tan_half);
            upload(gpu.device, gpu.params, &params, sizeof(params));
            upload(gpu.device, gpu.spheres, spheres.data(), sizeof(GpuSphere) * spheres.size());
            upload_mandel(gpu, scene.plane_color, light_time);
            dispatch(gpu, render_width, render_height);
            view.present(pixels, render_width, render_height);
            ++frames;
            float elapsed = std::chrono::duration<float>(now - fps_mark).count();
            if (elapsed >= 1.0f) {
                std::cout << "Bildrate: " << (static_cast<float>(frames) / elapsed) << " fps, " << render_width
                          << "x" << render_height << " Strahlen" << std::endl;
                frames = 0;
                fps_mark = now;
            }
            auto spent = clock::now() - now;
            if (spent < kFrame) {
                std::this_thread::sleep_for(kFrame - spent);
            }
        }
        vkUnmapMemory(gpu.device, gpu.staging.memory);
        std::cout << "Fenster geschlossen.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
