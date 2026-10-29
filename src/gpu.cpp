#include "log.h"

#include "error.h"
#include "gpu.h"
#include "mandelbrot_spv.h"
#include "trace_spv.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kGroupX = 16;
constexpr uint32_t kGroupY = 8;
constexpr uint32_t kMandelSize = 1536;
constexpr uint32_t kMandelGroup = 16;

struct MandelParams {
    float focus_zoom[4];  // xy Brennpunkt, z Zoom, w Farbphase
    float table_iter[4];  // rgb Plattenfarbe, w Iterationen
};

static_assert(sizeof(MandelParams) == 32, "MandelParams muss dem std140-Block im Mandelbrot-Shader entsprechen");

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

}  // namespace

Gpu::Gpu() = default;

Gpu::~Gpu() {
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

void Gpu::create_device() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "kugel-raytracer";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    vk_check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    if (count == 0) {
        fail("kein Vulkan-Gerät gefunden");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vk_check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");

    struct Choice {
        VkPhysicalDevice device;
        VkPhysicalDeviceProperties props;
        VkDeviceSize local;
    };
    std::vector<Choice> choices;
    spdlog::info("Vulkan-Geräte:");
    for (VkPhysicalDevice found : devices) {
        Choice choice;
        choice.device = found;
        vkGetPhysicalDeviceProperties(found, &choice.props);
        choice.local = local_memory(found);
        choices.push_back(choice);
        spdlog::info("  {} ({}, {} MiB)", choice.props.deviceName, device_type_name(choice.props.deviceType),
                     choice.local / (1024 * 1024));
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
    physical = selected.device;
    spdlog::info("Gewählt: {}", selected.props.deviceName);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    queue_family = family_count;
    for (uint32_t i = 0; i < family_count; ++i) {
        VkQueueFlags flags = families[i].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) && (flags & VK_QUEUE_TRANSFER_BIT)) {
            queue_family = i;
            break;
        }
    }
    if (queue_family == family_count) {
        fail("keine Compute-Queue");
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    vk_check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, queue_family, 0, &queue);
}

void Gpu::create_mandel() {
    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormatProperties format_props{};
    vkGetPhysicalDeviceFormatProperties(physical, format, &format_props);
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
    vk_check(vkCreateImage(device, &image_info, nullptr, &mandel_image), "vkCreateImage");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, mandel_image, &requirements);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = find_memory(physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_check(vkAllocateMemory(device, &alloc, nullptr, &mandel_memory), "vkAllocateMemory");
    vk_check(vkBindImageMemory(device, mandel_image, mandel_memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = mandel_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device, &view_info, nullptr, &mandel_view), "vkCreateImageView");

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device, &sampler_info, nullptr, &mandel_sampler), "vkCreateSampler");

    mandel_params = make_buffer(device, physical, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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
    vk_check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &mandel_layout),
             "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &mandel_layout;
    vk_check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &mandel_pipeline_layout),
             "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = kMandelSpvWords * sizeof(uint32_t);
    shader_info.pCode = kMandelSpv;
    vk_check(vkCreateShaderModule(device, &shader_info, nullptr, &mandel_shader), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mandel_shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = mandel_pipeline_layout;
    vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &mandel_pipeline),
             "vkCreateComputePipelines");
    spdlog::info("Mandelbrot-Pipeline: {}x{}, jede Frame.", kMandelSize, kMandelSize);
}

void Gpu::create_pipeline(const std::vector<GpuSphere>& sphere_data, const Params& params_data) {
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
    vk_check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout),
             "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    vk_check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
             "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = kTraceSpvWords * sizeof(uint32_t);
    shader_info.pCode = kTraceSpv;
    vk_check(vkCreateShaderModule(device, &shader_info, nullptr, &shader), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = pipeline_layout;
    vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
             "vkCreateComputePipelines");

    VkDeviceSize pixel_bytes = static_cast<VkDeviceSize>(params_data.width) * params_data.height * sizeof(float) * 4;
    pixels = make_buffer(device, physical, pixel_bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Gecachter Systemspeicher. Lesen aus VRAM oder ungecachetem Speicher kostet hier sonst den Frame.
    staging = make_buffer(device, physical, pixel_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    params = make_buffer(device, physical, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    spheres = make_buffer(device, physical, sizeof(GpuSphere) * sphere_data.size(),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    upload(params, &params_data, sizeof(params_data));
    upload(spheres, sphere_data.data(), sizeof(GpuSphere) * sphere_data.size());
    create_mandel();

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
    vk_check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &set_layout;
    vk_check(vkAllocateDescriptorSets(device, &alloc, &set), "vkAllocateDescriptorSets");

    alloc.pSetLayouts = &mandel_layout;
    vk_check(vkAllocateDescriptorSets(device, &alloc, &mandel_set), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo infos[3]{};
    infos[0] = {pixels.buffer, 0, VK_WHOLE_SIZE};
    infos[1] = {params.buffer, 0, 256};
    infos[2] = {spheres.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo floor_info{};
    floor_info.sampler = mandel_sampler;
    floor_info.imageView = mandel_view;
    floor_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &floor_info;

    VkDescriptorImageInfo storage_info{};
    storage_info.imageView = mandel_view;
    storage_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo mandel_uniform{mandel_params.buffer, 0, 256};
    VkWriteDescriptorSet mandel_writes[2]{};
    mandel_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    mandel_writes[0].dstSet = mandel_set;
    mandel_writes[0].dstBinding = 0;
    mandel_writes[0].descriptorCount = 1;
    mandel_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    mandel_writes[0].pImageInfo = &storage_info;
    mandel_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    mandel_writes[1].dstSet = mandel_set;
    mandel_writes[1].dstBinding = 1;
    mandel_writes[1].descriptorCount = 1;
    mandel_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    mandel_writes[1].pBufferInfo = &mandel_uniform;
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    vkUpdateDescriptorSets(device, 2, mandel_writes, 0, nullptr);

    VkCommandPoolCreateInfo pool_create{};
    pool_create.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_create.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_create.queueFamilyIndex = queue_family;
    vk_check(vkCreateCommandPool(device, &pool_create, nullptr, &command_pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(device, &command_info, &command), "vkAllocateCommandBuffers");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");
}

void Gpu::replace_pixel_targets(uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(device);
    destroy_buffer(device, pixels);
    destroy_buffer(device, staging);
    VkDeviceSize pixel_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(float) * 4;
    pixels = make_buffer(device, physical, pixel_bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    staging = make_buffer(device, physical, pixel_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDescriptorBufferInfo info{pixels.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void Gpu::upload(const Buffer& buffer, const void* data, size_t bytes) {
    void* mapped = nullptr;
    vk_check(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped), "vkMapMemory");
    std::memset(mapped, 0, static_cast<size_t>(buffer.size));
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(device, buffer.memory);
}

void Gpu::upload_params(const Params& value) {
    upload(params, &value, sizeof(value));
}

void Gpu::upload_spheres(const std::vector<GpuSphere>& value) {
    upload(spheres, value.data(), sizeof(GpuSphere) * value.size());
}

void Gpu::upload_mandel(Vec3 table, float time) {
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
    upload(mandel_params, &mandel, sizeof(mandel));
}

void Gpu::dispatch(uint32_t width, uint32_t height) {
    vk_check(vkResetFences(device, 1, &fence), "vkResetFences");
    vk_check(vkResetCommandBuffer(command, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");

    VkBufferMemoryBarrier host[3]{};
    host[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    host[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host[0].dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    host[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host[0].buffer = params.buffer;
    host[0].size = VK_WHOLE_SIZE;
    host[1] = host[0];
    host[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    host[1].buffer = spheres.buffer;
    host[2] = host[0];
    host[2].buffer = mandel_params.buffer;
    VkImageMemoryBarrier image_barrier{};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = mandel_image;
    image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_barrier.subresourceRange.levelCount = 1;
    image_barrier.subresourceRange.layerCount = 1;
    if (!mandel_ready) {
        image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &image_barrier);
        mandel_ready = true;
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         3, host, 0, nullptr);

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, mandel_pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, mandel_pipeline_layout, 0, 1,
                            &mandel_set, 0, nullptr);
    vkCmdDispatch(command, kMandelSize / kMandelGroup, kMandelSize / kMandelGroup, 1);

    image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_barrier);

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    uint32_t groups_x = (width + kGroupX - 1) / kGroupX;
    uint32_t groups_y = (height + kGroupY - 1) / kGroupY;
    vkCmdDispatch(command, groups_x, groups_y, 1);

    VkBufferMemoryBarrier to_copy{};
    to_copy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.buffer = pixels.buffer;
    to_copy.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &to_copy, 0, nullptr);

    VkBufferCopy region{};
    region.size = pixels.size;
    vkCmdCopyBuffer(command, pixels.buffer, staging.buffer, 1, &region);

    VkBufferMemoryBarrier to_host{};
    to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer = staging.buffer;
    to_host.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                         1, &to_host, 0, nullptr);
    vk_check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    vk_check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
    VkResult wait = vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ull);
    if (wait == VK_TIMEOUT) {
        fail("GPU-Dispatch hat nach 10 Sekunden nicht geendet");
    }
    vk_check(wait, "vkWaitForFences");
}

const float* Gpu::map_staging() {
    void* mapped = nullptr;
    vk_check(vkMapMemory(device, staging.memory, 0, staging.size, 0, &mapped), "vkMapMemory");
    return static_cast<const float*>(mapped);
}

void Gpu::unmap_staging() {
    vkUnmapMemory(device, staging.memory);
}
