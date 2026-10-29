#pragma once

#include <cstdint>
#include <vector>

struct Vec3 {
    float x, y, z;
};

inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

float dot(Vec3 a, Vec3 b);
Vec3 cross(Vec3 a, Vec3 b);
Vec3 normalize(Vec3 a);
void put(float out[4], Vec3 v, float w);

// std140-Block der Trace-Pipeline. light_dir.w hält tan(fov/2).
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

// std430-Struct im Shader. material.x ist Metall, material.y ist Rauheit.
struct GpuSphere {
    float geom[4];
    float color[4];
    float material[4];
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    float metallic;
    float roughness;
};

constexpr uint32_t kMaxSpheres = 8;

static_assert(sizeof(Params) == 160, "Params muss dem std140-Block im Shader entsprechen");
static_assert(sizeof(GpuSphere) == 48, "GpuSphere muss dem std430-Struct im Shader entsprechen");

// Kugeln, Kamera und Licht. Die GPU-Daten baut make_params und rotate.
class Scene {
public:
    Vec3 cam_pos{};
    Vec3 cam_target{};
    float fov_y_degrees = 48.0f;
    Vec3 light_dir{};
    Vec3 light_color{};
    float ambient = 0.08f;
    Vec3 plane_point{};
    Vec3 plane_normal{};
    Vec3 plane_color{};
    std::vector<Sphere> spheres;

    static Scene make();
    Params make_params(uint32_t width, uint32_t height) const;
    Vec3 pivot() const;
    void rotate(Vec3 pivot, float angle, std::vector<GpuSphere>& out) const;
    static Vec3 light_direction(float time);
    static void place_camera(Params& params, float angle);
};
