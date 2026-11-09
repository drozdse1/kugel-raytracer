#pragma once

#include <chrono>
#include <cstdint>

// Linux, auch unter Wayland über XWayland, bleibt bei X11.
// macOS hat kein X11 und zeigt dasselbe Bild über SDL.
#if defined(__APPLE__)
#include <vector>
struct SDL_Window;
struct SDL_Renderer;
struct SDL_Texture;
#else
#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#endif

// Ein Strahl deckt block×block Fensterpixel. Pfeil runter macht die Blöcke größer.
uint32_t ray_extent(uint32_t window_extent, uint32_t block);

// Fenster. Die GPU rechnet eine kleinere Auflösung, present streckt sie auf.
class LiveView {
public:
    explicit LiveView(uint32_t view_width, uint32_t view_height);
    ~LiveView();
    LiveView(const LiveView&) = delete;
    LiveView& operator=(const LiveView&) = delete;

    void resize(uint32_t next_width, uint32_t next_height);
    bool pump();
    bool take_resize(uint32_t& out_width, uint32_t& out_height);
    uint32_t ray_block() const;
    bool take_quality();
    uint32_t bounce_count() const;
    bool take_depth();
    void show_rays(uint32_t render_width, uint32_t render_height);
    void present(const float* pixels, uint32_t source_width, uint32_t source_height);

private:
    void destroy_image();
    void create_image();
    void note_frame();
    void draw_overlay();

#if defined(__APPLE__)
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    std::vector<uint32_t> frame;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t requested_width = 0;
    uint32_t requested_height = 0;
    bool size_dirty = false;
    int ray_block_index = 0;
    bool quality_dirty = false;
    unsigned long last_quality_ms = 0;
    int bounce_index = 3;
    bool depth_dirty = false;
    unsigned long last_depth_ms = 0;
    bool overlay = false;
    unsigned long last_overlay_ms = 0;
    float overlay_fps = 0.0f;
    int overlay_frames = 0;
    std::chrono::steady_clock::time_point overlay_mark{};
#else
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
    int bounce_index = 3;
    bool depth_dirty = false;
    unsigned long last_depth_ms = 0;
    int red_shift = 0;
    int green_shift = 0;
    int blue_shift = 0;
    XFontStruct* font = nullptr;
    bool overlay = false;
    unsigned long last_overlay_ms = 0;
    float overlay_fps = 0.0f;
    int overlay_frames = 0;
    std::chrono::steady_clock::time_point overlay_mark{};
#endif
};
