#pragma once

#include "scene.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

// Compute-Gerät: Mandelbrot-Textur und der Strahlwurf pro Pixel.
class Gpu {
public:
    Gpu();
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    void create_device();
    void create_pipeline(const std::vector<GpuSphere>& spheres, const Params& params);
    void replace_pixel_targets(uint32_t width, uint32_t height);
    void upload_params(const Params& params);
    void upload_spheres(const std::vector<GpuSphere>& spheres);
    void upload_mandel(Vec3 table, float time);
    void dispatch(uint32_t width, uint32_t height);
    const float* map_staging();
    void unmap_staging();

private:
    void upload(const Buffer& buffer, const void* data, size_t bytes);
    void create_mandel();

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
};
