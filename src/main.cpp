// Kugel-Raytracer auf der AMD-GPU.
// Scene hält Kugeln, Kamera und Licht. Gpu rechnet Mandelbrot und Strahlen.
// LiveView zeigt das Bild, RgbImage schreibt ein Standbild.

#include "log.h"

#include "error.h"
#include "gpu.h"
#include "image.h"
#include "live_view.h"
#include "scene.h"

#include <chrono>
#include <string>
#include <thread>

namespace {

void usage() {
    spdlog::info(
        "Verwendung: kugel-raytracer [breite] [hoehe]\n"
        "            kugel-raytracer --bild ausgabe.ppm [breite] [hoehe]\n"
        "Ohne --bild öffnet sich ein Fenster. Die Größe lässt sich ändern. "
        "Pfeil hoch und runter ändert die Strahlzahl. Pfeil links und rechts ändert die Sprungtiefe. "
        "Fragezeichen schaltet das Debug-Overlay ein. Escape oder Q beendet es.");
}

}  // namespace

int main(int argc, char** argv) {
    setup_log();
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
                spdlog::error("Bei --bild fehlt der Dateiname");
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

        Scene scene = Scene::make();
        if (scene.spheres.empty() || scene.spheres.size() > kMaxSpheres) {
            fail("eine bis acht Kugeln");
        }
        Params params = scene.make_params(width, height);
        Vec3 pivot = scene.pivot();
        std::vector<GpuSphere> spheres(scene.spheres.size());
        scene.rotate(pivot, 0.0f, spheres);

        Gpu gpu;
        gpu.create_device();
        gpu.create_pipeline(spheres, params);

        if (still) {
            gpu.upload_mandel(scene.plane_color, 0.0f);
            gpu.dispatch(width, height);
            RgbImage image = RgbImage::read(gpu, width, height);
            image.write_ppm(still_path);
            image.report();
            spdlog::info("Geschrieben: {}", still_path);
            return 0;
        }

        LiveView view(width, height);
        uint32_t render_width = width;
        uint32_t render_height = height;
        const float* pixels = gpu.map_staging();

        auto retarget = [&]() {
            uint32_t rw = ray_extent(width, view.ray_block());
            uint32_t rh = ray_extent(height, view.ray_block());
            if (rw != render_width || rh != render_height) {
                gpu.unmap_staging();
                gpu.replace_pixel_targets(rw, rh);
                pixels = gpu.map_staging();
                render_width = rw;
                render_height = rh;
                params.width = rw;
                params.height = rh;
            }
            view.show_rays(render_width, render_height);
        };

        spdlog::info(
            "Fenster offen, {}x{}. Die Größe lässt sich ändern. Pfeil runter rechnet weniger Strahlen, "
            "das Bild wird gröber und schneller, Pfeil hoch macht es wieder feiner. "
            "Pfeil links verringert die Sprünge, Pfeil rechts erhöht sie: 1 bis 8. "
            "Fragezeichen schaltet ein Debug-Overlay mit Bildrate, Tiefe und Pixel pro Strahl ein. "
            "Die Kamera kreist auf einer Krone, die Kugeln drehen und "
            "spiegeln, das Mandelbrot wandert, das Licht wandert auf Ellipsen. "
            "Beenden mit Escape oder Q.",
            width, height);
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
                    spdlog::info("Strahlen: 1 pro {}x{} Pixel, gerechnet {}x{}", view.ray_block(),
                                 view.ray_block(), render_width, render_height);
                }
            }
            if (view.take_depth()) {
                spdlog::info("Sprünge: {}", view.bounce_count());
                view.show_rays(render_width, render_height);
            }
            params.bounces = view.bounce_count();
            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - previous).count();
            previous = now;
            if (dt > 0.1f) {
                dt = 0.1f;
            }
            angle += dt * kRadiansPerSecond;
            light_time += dt;
            scene.rotate(pivot, angle, spheres);
            Scene::place_camera(params, light_time * 0.32f);
            float tan_half = params.light_dir[3];
            put(params.light_dir, Scene::light_direction(light_time), tan_half);
            gpu.upload_params(params);
            gpu.upload_spheres(spheres);
            gpu.upload_mandel(scene.plane_color, light_time);
            gpu.dispatch(render_width, render_height);
            view.present(pixels, render_width, render_height);
            ++frames;
            float elapsed = std::chrono::duration<float>(now - fps_mark).count();
            if (elapsed >= 1.0f) {
                spdlog::info("Bildrate: {:.1f} fps, {}x{} Strahlen, {} Sprünge",
                             static_cast<float>(frames) / elapsed, render_width, render_height,
                             view.bounce_count());
                frames = 0;
                fps_mark = now;
            }
            auto spent = clock::now() - now;
            if (spent < kFrame) {
                std::this_thread::sleep_for(kFrame - spent);
            }
        }
        gpu.unmap_staging();
        spdlog::info("Fenster geschlossen.");
        return 0;
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return 1;
    }
}
