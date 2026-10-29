#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Gpu;

// Gelesene Pixel des Standbilds, als 8-Bit-RGB.
class RgbImage {
public:
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;

    static RgbImage read(Gpu& gpu, uint32_t width, uint32_t height);
    void write_ppm(const std::string& path) const;
    void report() const;
};
