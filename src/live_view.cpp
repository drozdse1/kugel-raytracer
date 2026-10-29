#include "log.h"

#include "error.h"
#include "live_view.h"
#include "scene.h"

#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/ipc.h>
#include <sys/shm.h>

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

int mask_shift(unsigned long mask) {
    int shift = 0;
    while (mask != 0 && (mask & 1ul) == 0) {
        mask >>= 1;
        ++shift;
    }
    return shift;
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
    font = XLoadQueryFont(display, "9x15");
    if (font == nullptr) {
        font = XLoadQueryFont(display, "fixed");
    }
    if (font == nullptr) {
        spdlog::warn("Keine X11-Schrift für das Debug-Overlay.");
    }
    XSync(display, False);
    create_image();
    if (use_shm) {
        spdlog::info("Ausgabe über Shared Memory.");
    } else {
        spdlog::warn("Ausgabe über XPutImage.");
    }
}

LiveView::~LiveView() {
    if (display == nullptr) {
        return;
    }
    destroy_image();
    if (font != nullptr) {
        XFreeFont(display, font);
    }
    if (gc != nullptr) {
        XFreeGC(display, gc);
    }
    if (window != 0) {
        XDestroyWindow(display, window);
    }
    XCloseDisplay(display);
}

void LiveView::resize(uint32_t next_width, uint32_t next_height) {
    width = next_width;
    height = next_height;
    destroy_image();
    create_image();
}

bool LiveView::pump() {
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
            KeySym shifted = XLookupKeysym(&event.xkey, 1);
            if ((key == XK_question || shifted == XK_question) && event.xkey.time - last_overlay_ms >= 160) {
                last_overlay_ms = event.xkey.time;
                overlay = !overlay;
                spdlog::info(overlay ? "Debug-Overlay an." : "Debug-Overlay aus.");
            }
            bool left = key == XK_Left || key == XK_KP_Left;
            bool right = key == XK_Right || key == XK_KP_Right;
            if ((left || right) && event.xkey.time - last_depth_ms >= 160) {
                last_depth_ms = event.xkey.time;
                if (left && bounce_index > 0) {
                    --bounce_index;
                    depth_dirty = true;
                } else if (right && bounce_index + 1 < kBounceCountCount) {
                    ++bounce_index;
                    depth_dirty = true;
                }
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
    XStoreName(display, window, title.c_str());
}

void LiveView::destroy_image() {
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

void LiveView::create_image() {
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

void LiveView::present(const float* pixels, uint32_t source_width, uint32_t source_height) {
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
    }
    note_frame();
    if (overlay) {
        draw_overlay();
    }
    XFlush(display);
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
    if (font == nullptr || gc == nullptr) {
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

    int line_height = font->ascent + font->descent + 4;
    int widest = 0;
    for (const char* line : lines) {
        int text_width = XTextWidth(font, line, static_cast<int>(std::strlen(line)));
        if (text_width > widest) {
            widest = text_width;
        }
    }
    int pad = 10;
    int origin_x = 16;
    int origin_y = 16;
    unsigned int box_width = static_cast<unsigned int>(widest + pad * 2);
    unsigned int box_height = static_cast<unsigned int>(line_height * 3 + pad * 2);
    int screen = DefaultScreen(display);
    XSetFont(display, gc, font->fid);
    XSetForeground(display, gc, BlackPixel(display, screen));
    XFillRectangle(display, window, gc, origin_x, origin_y, box_width, box_height);
    XSetForeground(display, gc, WhitePixel(display, screen));
    int baseline = origin_y + pad + font->ascent;
    for (const char* line : lines) {
        XDrawString(display, window, gc, origin_x + pad, baseline, line, static_cast<int>(std::strlen(line)));
        baseline += line_height;
    }
}
