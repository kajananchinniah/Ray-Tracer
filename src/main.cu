#include "camera/camera.hpp"
#include "common/common_types.hpp"
#include "common/cuda_memory_utils.hpp"
#include "image/cuda_image_utils.hpp"
#include "image/image.hpp"
#include "image/image_utils.hpp"
#include "material/material.hpp"
#include "ray/ray.hpp"
#include "ray/ray_utils.hpp"
#include "render/render.hpp"
#include "vector3/vector3.hpp"

#include <chrono>
#include <iostream>
#include <random>

namespace
{

RayTracer::f32 randomF32()
{
    static std::uniform_real_distribution<RayTracer::f32> distribution(0.0f,
                                                                       1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

RayTracer::f32 randomF32(RayTracer::f32 min, RayTracer::f32 max)
{
    return min + (max - min) * randomF32();
}

RayTracer::SphereArray randomScene()
{
    RayTracer::SphereArray world{488};
    RayTracer::Lambertian ground_material{RayTracer::Colour{0.5f, 0.5f, 0.5f}};
    bool result{true};
    result = world.add(
        RayTracer::Sphere(RayTracer::Point3f(0.0f, -1000.0f, 0.0f), 1000,
                          RayTracer::Material(ground_material)));
    if (!result) {
        std::cerr << "Could not add sphere!";
        std::exit(1);
    }

    for (RayTracer::s32 a = -11; a < 11; ++a) {
        for (RayTracer::s32 b = -11; b < 11; ++b) {
            RayTracer::f32 choose_mat{randomF32()};
            RayTracer::Point3f center{
                static_cast<RayTracer::f32>(a) + 0.9f * randomF32(), 0.2f,
                static_cast<RayTracer::f32>(b) + 0.9f * randomF32()};

            if ((center - RayTracer::Point3f{4.0f, 0.2f, 0.0f})
                    .magnitude_host() > 0.9f) {
                RayTracer::Material sphere_material{};

                if (choose_mat < 0.9f) {
                    RayTracer::Colour albedo{randomF32(), randomF32(),
                                             randomF32()};
                    sphere_material =
                        RayTracer::Material{RayTracer::Lambertian{albedo}};
                    result = world.add(
                        RayTracer::Sphere{center, 0.2f, sphere_material});
                } else if (choose_mat < 0.95f) {
                    RayTracer::Colour albedo{randomF32(0.5f, 1.0f),
                                             randomF32(0.5f, 1.0f),
                                             randomF32(0.5f, 1.0f)};
                    RayTracer::f32 fuzz{randomF32(0.0f, 0.5f)};
                    sphere_material =
                        RayTracer::Material{RayTracer::Metal{albedo, fuzz}};
                    result = world.add(
                        RayTracer::Sphere{center, 0.2f, sphere_material});
                } else {
                    sphere_material =
                        RayTracer::Material{RayTracer::Dielectric{1.5f}};
                    result = world.add(
                        RayTracer::Sphere{center, 0.2f, sphere_material});
                }
                if (!result) {
                    std::cerr << "Could not add sphere!";
                    std::exit(1);
                }
            }
        }
    }

    RayTracer::Material material1{RayTracer::Dielectric{1.5f}};
    RayTracer::Material material2{
        RayTracer::Lambertian{RayTracer::Colour{0.4f, 0.2f, 0.1f}}};
    RayTracer::Material material3{
        RayTracer::Metal{RayTracer::Colour{0.7f, 0.6f, 0.5f}, 0.0f}};

    result = result &&
             world.add(RayTracer::Sphere{RayTracer::Point3f{0.0f, 1.0f, 0.0f},
                                         1.0f, material1});
    result = result &&
             world.add(RayTracer::Sphere{RayTracer::Point3f{-4.0f, 1.0f, 0.0f},
                                         1.0f, material2});
    result = result &&
             world.add(RayTracer::Sphere{RayTracer::Point3f{4.0f, 1.0f, 0.0f},
                                         1.0f, material3});
    if (!result) {
        std::cerr << "Could not add sphere!";
        std::exit(1);
    }

    return world;
}
} // namespace

int main()
{
    const RayTracer::f32 aspect_ratio{3.0f / 2.0f};
    const RayTracer::s64 image_width{1200};
    const RayTracer::s64 image_height{
        static_cast<RayTracer::s64>(image_width / aspect_ratio)};
    const RayTracer::s32 samples_per_pixel{10};
    const RayTracer::s32 max_depth{50};

    RayTracer::SphereArray world = randomScene();

    RayTracer::Point3f look_from{13, 2, 3};
    RayTracer::Point3f look_at{0, 0, 0};
    RayTracer::Vector3f v_up{0, 1, 0};
    RayTracer::f32 dist_to_focus{10.0f};
    RayTracer::f32 aperture{0.1f};

    RayTracer::Camera camera{look_from,    look_at,  v_up,         20,
                             aspect_ratio, aperture, dist_to_focus};

    RayTracer::Image image{image_width, image_height};
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    RayTracer::cuda::render(camera, world, samples_per_pixel, max_depth, image);
    RayTracer::ImageUtils::saveImage("result.png", image);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;
}
