#include "scene.h"

#include <cmath>

float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3 normalize(Vec3 a) {
    float length = std::sqrt(dot(a, a));
    return a * (1.0f / length);
}

void put(float out[4], Vec3 v, float w) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = w;
}

Scene Scene::make() {
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

Params Scene::make_params(uint32_t width, uint32_t height) const {
    Vec3 forward = normalize(cam_target - cam_pos);
    Vec3 right = normalize(cross(forward, Vec3{0.0f, 1.0f, 0.0f}));
    Vec3 up = cross(right, forward);
    float tan_half = std::tan(fov_y_degrees * 0.5f * 3.14159265f / 180.0f);

    Params params{};
    put(params.cam_pos, cam_pos, 1.0f);
    put(params.cam_forward, forward, 0.0f);
    put(params.cam_right, right, 0.0f);
    put(params.cam_up, up, 0.0f);
    put(params.light_dir, light_dir, tan_half);
    put(params.light_color, light_color, ambient);
    put(params.plane_point, plane_point, 0.0f);
    put(params.plane_normal, plane_normal, 0.0f);
    put(params.plane_color, plane_color, 1.0f);
    params.width = width;
    params.height = height;
    params.sphere_count = static_cast<uint32_t>(spheres.size());
    return params;
}

Vec3 Scene::pivot() const {
    Vec3 center{0.0f, 0.0f, 0.0f};
    for (const Sphere& sphere : spheres) {
        center.x += sphere.center.x;
        center.z += sphere.center.z;
    }
    float count = static_cast<float>(spheres.size());
    center.x /= count;
    center.z /= count;
    return center;
}

Vec3 Scene::light_direction(float time) {
    // Azimut und Höhe auf zwei überlagerten Ellipsen. Die Höhe bleibt positiv,
    // der Schatten wandert dadurch deutlich um die Kugeln.
    float turn = time * 0.80f;
    float azimuth = 0.50f + 1.15f * std::sin(turn) + 0.40f * std::sin(turn * 2.0f);
    float elevation = 0.98f + 0.34f * std::cos(turn) + 0.14f * std::cos(turn * 2.0f);
    return {std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)};
}

void Scene::rotate(Vec3 pivot, float angle, std::vector<GpuSphere>& out) const {
    float cosine = std::cos(angle);
    float sine = std::sin(angle);
    for (size_t i = 0; i < spheres.size(); ++i) {
        float x = spheres[i].center.x - pivot.x;
        float z = spheres[i].center.z - pivot.z;
        Vec3 center{x * cosine - z * sine + pivot.x, spheres[i].center.y, x * sine + z * cosine + pivot.z};
        put(out[i].geom, center, spheres[i].radius);
        put(out[i].color, spheres[i].color, 0.0f);
        out[i].material[0] = spheres[i].metallic;
        out[i].material[1] = spheres[i].roughness;
        out[i].material[2] = 0.0f;
        out[i].material[3] = 0.0f;
    }
}

void Scene::place_camera(Params& params, float angle) {
    // Kreis um den Blickpunkt, die Höhe ist eine Sinuswelle mit fünf Bögen: eine Krone.
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
