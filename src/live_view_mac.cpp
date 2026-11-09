#include "log.h"

#include "error.h"
#include "live_view.h"
#include "scene.h"

#include <SDL.h>

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr uint32_t kRayBlocks[] = {1, 2, 3, 4, 6, 8};
constexpr int kRayBlockCount = static_cast<int>(sizeof(kRayBlocks) / sizeof(kRayBlocks[0]));
constexpr uint32_t kBounceCounts[] = {1, 2, 3, 4, 5, 6, 7, 8};
constexpr int kBounceCountCount = static_cast<int>(sizeof(kBounceCounts) / sizeof(kBounceCounts[0]));
static_assert(kBounceCounts[3] == kDefaultBounces, "Startindex 3 muss die voreingestellte Sprungtiefe sein");

bool quality_key(SDL_Keycode key, bool& up, bool& down) {
    up = key == SDLK_UP;
    down = key == SDLK_DOWN;
    return up || down;
}

bool depth_key(SDL_Keycode key, bool& left, bool& right) {
    left = key == SDLK_LEFT;
    right = key == SDLK_RIGHT;
    return left || right;
}

}  // namespace

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

LiveView::LiveView(uint32_t view_width, uint32_t view_height) : width(view_width), height(view_height) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fail(std::string("SDL startet nicht: ") + SDL_GetError());
    }
    window = SDL_CreateWindow("Kugel-Raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              static_cast<int>(width), static_cast<int>(height), SDL_WINDOW_RESIZABLE);
    if (window == nullptr) {
        std::string message = SDL_GetError();
        SDL_Quit();
        fail("kein Fenster: " + message);
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        std::string message = SDL_GetError();
        SDL_DestroyWindow(window);
        window = nullptr;
        SDL_Quit();
        fail("kein Renderer: " + message);
    }
    SDL_StartTextInput();
    show_rays(width, height);
    create_image();
    spdlog::info("Ausgabe über SDL.");
}

LiveView::~LiveView() {
    destroy_image();
    if (renderer != nullptr) {
        SDL_DestroyRenderer(renderer);
    }
    if (window != nullptr) {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
}

void LiveView::resize(uint32_t next_width, uint32_t next_height) {
    width = next_width;
    height = next_height;
    destroy_image();
    create_image();
}

bool LiveView::pump() {
    SDL_Event event;
    while (SDL_PollEvent(&event) != 0) {
        if (event.type == SDL_QUIT) {
            return false;
        }
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
            return false;
        }
        if (event.type == SDL_WINDOWEVENT && (event.window.event == SDL_WINDOWEVENT_RESIZED ||
                                              event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)) {
            int next_width = std::clamp(event.window.data1, 16, 8192);
            int next_height = std::clamp(event.window.data2, 16, 8192);
            requested_width = static_cast<uint32_t>(next_width);
            requested_height = static_cast<uint32_t>(next_height);
            size_dirty = requested_width != width || requested_height != height;
        }
        if (event.type == SDL_KEYDOWN && event.key.repeat == 0) {
            SDL_Keycode key = event.key.keysym.sym;
            if (key == SDLK_ESCAPE || key == SDLK_q) {
                return false;
            }
        }
        if (event.type == SDL_KEYDOWN) {
            SDL_Keycode key = event.key.keysym.sym;
            unsigned long now = SDL_GetTicks();
            bool up = false;
            bool down = false;
            if (quality_key(key, up, down) && now - last_quality_ms >= 160) {
                last_quality_ms = now;
                if (up && ray_block_index > 0) {
                    --ray_block_index;
                    quality_dirty = true;
                } else if (down && ray_block_index + 1 < kRayBlockCount) {
                    ++ray_block_index;
                    quality_dirty = true;
                }
            }
            bool left = false;
            bool right = false;
            if (depth_key(key, left, right) && now - last_depth_ms >= 160) {
                last_depth_ms = now;
                if (left && bounce_index > 0) {
                    --bounce_index;
                    depth_dirty = true;
                } else if (right && bounce_index + 1 < kBounceCountCount) {
                    ++bounce_index;
                    depth_dirty = true;
                }
            }
        }
        // SDL_TEXTINPUT trifft das Fragezeichen auch auf dem deutschen Layout.
        if (event.type == SDL_TEXTINPUT && std::strcmp(event.text.text, "?") == 0) {
            unsigned long now = SDL_GetTicks();
            if (now - last_overlay_ms >= 160) {
                last_overlay_ms = now;
                overlay = !overlay;
                spdlog::info(overlay ? "Debug-Overlay an." : "Debug-Overlay aus.");
            }
        }
    }
    return true;
}

bool LiveView::take_resize(uint32_t& out_width, uint32_t& out_height) {
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

uint32_t LiveView::ray_block() const {
    return kRayBlocks[ray_block_index];
}

bool LiveView::take_quality() {
    if (!quality_dirty) {
        return false;
    }
    quality_dirty = false;
    return true;
}

uint32_t LiveView::bounce_count() const {
    return kBounceCounts[bounce_index];
}

bool LiveView::take_depth() {
    if (!depth_dirty) {
        return false;
    }
    depth_dirty = false;
    return true;
}

void LiveView::show_rays(uint32_t render_width, uint32_t render_height) {
    uint32_t block = ray_block();
    std::string title = "Kugel-Raytracer — " + std::to_string(render_width) + "x" +
                        std::to_string(render_height) + " Strahlen, 1 pro " + std::to_string(block) + "x" +
                        std::to_string(block) + " Pixel, " + std::to_string(bounce_count()) + " Sprünge";
    SDL_SetWindowTitle(window, title.c_str());
}

void LiveView::destroy_image() {
    if (texture != nullptr) {
        SDL_DestroyTexture(texture);
        texture = nullptr;
    }
    frame.clear();
}

void LiveView::create_image() {
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
                                static_cast<int>(width), static_cast<int>(height));
    if (texture == nullptr) {
        fail(std::string("kein Fensterbild: ") + SDL_GetError());
    }
    SDL_SetTextureScaleMode(texture, SDL_ScaleModeNearest);
    frame.assign(static_cast<size_t>(width) * height, 0);
}

void LiveView::present(const float* pixels, uint32_t source_width, uint32_t source_height) {
    auto pack = [](const float* sample) {
        auto channel = [](float value) {
            value = std::clamp(value, 0.0f, 1.0f);
            return static_cast<uint32_t>(value * 255.0f + 0.5f);
        };
        return channel(sample[0]) | (channel(sample[1]) << 8) | (channel(sample[2]) << 16) | (255u << 24);
    };
    if (source_width == width && source_height == height) {
        for (uint32_t y = 0; y < height; ++y) {
            uint32_t* row = frame.data() + static_cast<size_t>(y) * width;
            const float* sample = pixels + static_cast<size_t>(y) * width * 4;
            for (uint32_t x = 0; x < width; ++x, sample += 4) {
                row[x] = pack(sample);
            }
        }
    } else {
        for (uint32_t sy = 0; sy < source_height; ++sy) {
            uint32_t y0 = sy * height / source_height;
            uint32_t y1 = (sy + 1) * height / source_height;
            const float* sample = pixels + static_cast<size_t>(sy) * source_width * 4;
            for (uint32_t sx = 0; sx < source_width; ++sx, sample += 4) {
                uint32_t packed = pack(sample);
                uint32_t x0 = sx * width / source_width;
                uint32_t x1 = (sx + 1) * width / source_width;
                for (uint32_t y = y0; y < y1; ++y) {
                    uint32_t* row = frame.data() + static_cast<size_t>(y) * width;
                    for (uint32_t x = x0; x < x1; ++x) {
                        row[x] = packed;
                    }
                }
            }
        }
    }
    note_frame();
    if (overlay) {
        draw_overlay();
    }
    SDL_UpdateTexture(texture, nullptr, frame.data(), static_cast<int>(width * sizeof(uint32_t)));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
}

void LiveView::note_frame() {
    using clock = std::chrono::steady_clock;
    auto now = clock::now();
    if (overlay_frames == 0 && overlay_fps == 0.0f) {
        overlay_mark = now;
    }
    ++overlay_frames;
    float elapsed = std::chrono::duration<float>(now - overlay_mark).count();
    if (elapsed >= 0.5f) {
        overlay_fps = static_cast<float>(overlay_frames) / elapsed;
        overlay_frames = 0;
        overlay_mark = now;
    }
}

void LiveView::draw_overlay() {
    if (frame.empty()) {
        return;
    }
    char lines[3][64];
    if (overlay_fps <= 0.0f) {
        std::snprintf(lines[0], sizeof(lines[0]), "fps ...");
    } else {
        std::snprintf(lines[0], sizeof(lines[0]), "%.1f fps", static_cast<double>(overlay_fps));
    }
    std::snprintf(lines[1], sizeof(lines[1]), "Tiefe %u", bounce_count());
    std::snprintf(lines[2], sizeof(lines[2]), "%ux%u Pixel pro Strahl", ray_block(), ray_block());

    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(frame.data(), width, height, 8, static_cast<size_t>(width) * 4,
                                                 space, kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(space);
    if (context == nullptr) {
        return;
    }
    CTFontRef font = CTFontCreateWithName(CFSTR("Menlo"), 15.0, nullptr);
    if (font == nullptr) {
        CGContextRelease(context);
        return;
    }
    CGColorRef white = CGColorCreateGenericRGB(1.0, 1.0, 1.0, 1.0);
    CTLineRef drawn[3] = {};
    double widest = 0.0;
    double ascent = 0.0;
    double descent = 0.0;
    for (int i = 0; i < 3; ++i) {
        CFStringRef text = CFStringCreateWithCString(nullptr, lines[i], kCFStringEncodingUTF8);
        const void* keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
        const void* values[] = {font, white};
        CFDictionaryRef attributes = CFDictionaryCreate(nullptr, keys, values, 2, &kCFTypeDictionaryKeyCallBacks,
                                                         &kCFTypeDictionaryValueCallBacks);
        CFAttributedStringRef attributed = CFAttributedStringCreate(nullptr, text, attributes);
        drawn[i] = CTLineCreateWithAttributedString(attributed);
        double line_ascent = 0.0;
        double line_descent = 0.0;
        double text_width = CTLineGetTypographicBounds(drawn[i], &line_ascent, &line_descent, nullptr);
        widest = std::max(widest, text_width);
        ascent = std::max(ascent, line_ascent);
        descent = std::max(descent, line_descent);
        CFRelease(attributed);
        CFRelease(attributes);
        CFRelease(text);
    }

    double line_height = ascent + descent + 4.0;
    double pad = 10.0;
    double box_width = widest + pad * 2.0;
    double box_height = line_height * 3.0 + pad * 2.0;
    double origin_x = 16.0;
    double box_y = static_cast<double>(height) - 16.0 - box_height;
    CGContextSetRGBFillColor(context, 0.0, 0.0, 0.0, 1.0);
    CGContextFillRect(context, CGRectMake(origin_x, box_y, box_width, box_height));
    CGContextSetTextMatrix(context, CGAffineTransformIdentity);
    double baseline = box_y + box_height - pad - ascent;
    for (int i = 0; i < 3; ++i) {
        CGContextSetTextPosition(context, origin_x + pad, baseline);
        CTLineDraw(drawn[i], context);
        CFRelease(drawn[i]);
        baseline -= line_height;
    }
    CGColorRelease(white);
    CFRelease(font);
    CGContextRelease(context);
}
