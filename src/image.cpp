#include "image.h"

#include "error.h"
#include "gpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

RgbImage RgbImage::read(Gpu& gpu, uint32_t width, uint32_t height) {
    const float* pixels = gpu.map_staging();
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
    gpu.unmap_staging();
    return image;
}

void RgbImage::write_ppm(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("kann " + path + " nicht schreiben");
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (!out) {
        fail("Schreiben von " + path + " unvollständig");
    }
}

void RgbImage::report() const {
    int sky = 0;
    int floor_lit = 0;
    int shadow = 0;
    int red = 0;
    int blue = 0;
    int gold = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t i = (static_cast<size_t>(y) * width + x) * 3;
            int r = rgb[i];
            int g = rgb[i + 1];
            int b = rgb[i + 2];
            int max_channel = std::max(r, std::max(g, b));
            int min_channel = std::min(r, std::min(g, b));
            bool gray = max_channel - min_channel < 18;
            if (y < height / 3 && b > r + 12 && b > 90) {
                ++sky;
            }
            if (gray && y > height / 2 && max_channel > 150) {
                ++floor_lit;
            }
            if (gray && y > height / 2 && max_channel < 110 && max_channel > 25) {
                ++shadow;
            }
            if (r > 150 && r > g + 50 && r > b + 50) {
                ++red;
            }
            if (y > height / 2 && b > 140 && b > r + 40 && b > g + 20) {
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
