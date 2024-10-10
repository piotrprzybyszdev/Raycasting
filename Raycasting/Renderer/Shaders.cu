#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "Shaders.h"

namespace Raycasting
{

// Sets the color of the pixel at (x, y)
__device__ void SurfaceSetPixelColor(cudaSurfaceObject_t surface, size_t x, size_t y, vec4<float> color)
{
    vec4<uint8_t> rgba = clamp(color, 0.0f, 1.0f) * 255.0f;

    uint32_t bytes = rgba.w << 24 | rgba.z << 16 | rgba.y << 8 | rgba.x;

    surf2Dwrite(bytes, surface, x * sizeof(uint32_t), y, cudaBoundaryModeZero);
}

// Computes the color of the pixel at (x, y)
__device__ __host__ vec3<float> TraceRay(uint32_t x, uint32_t y, uint32_t width, uint32_t height,
    const mat4<float> &view, const mat4<float> &proj, vec3<float> position, size_t sphereCount,
    const SphereRef &spheres, size_t lightCount, const LightRef &lights)
{
    // compute pixel coordinates in [-1, 1] x [-1, 1] space
    vec2 ndc = vec2(x / (float)width, y / (float)height) * 2.0f - 1.0f;

    // ray direction in camera space
    vec3 camera(ndc, 1.0f);

    // multiply ray direction vector in camera space by projection and view matrices
    // to get ray direction in world space
    vec4 target = proj * vec4(camera, 1.0f);
    target = normalize(target / target.w);
    target.w = 0.0f;

    vec3 dir = view * target;

    float closestHit = FLT_MAX;
    int sphereIndex = -1;

    // Find closest sphere intersected by the ray
    for (int i = 0; i < sphereCount; i++)
    {
        vec3 spherePosition(spheres.PositionX[i], spheres.PositionY[i], spheres.PositionZ[i]);
        float radius = spheres.Radius[i];

        // transform camera and sphere positions so that the sphere center is at point (0, 0, 0)
        vec3 origin = position - spherePosition;

        // solve quadratic equation to find the intersection
        float a = dot(dir, dir);
        float b = 2.0f * dot(origin, dir);
        float c = dot(origin, origin) - radius * radius;

        float discriminant = b * b - 4.0f * a * c;

        // case where the ray doesn't intersect the sphere
        if (discriminant < 0.0f) continue;

        float hitDistance = (-b - sqrtf(discriminant)) / (2.0f * a);

        if (hitDistance < 0.0f || hitDistance > closestHit) continue;

        // save the sphere index only if ther sphere is not behind the camera
        // and it is the closest one to the camera yet
        closestHit = hitDistance;
        sphereIndex = i;
    }

    // if the ray didn't hit any spheres - return black as the background
    if (sphereIndex == -1)
        return vec3(0.0f, 0.0f, 0.0f);

    vec3 color(0.0f, 0.0f, 0.0f);
    vec3 spherePosition(spheres.PositionX[sphereIndex], spheres.PositionY[sphereIndex], spheres.PositionZ[sphereIndex]);
    vec3 origin = position - spherePosition;
    vec3 sphereColor(spheres.ColorR[sphereIndex], spheres.ColorG[sphereIndex], spheres.ColorB[sphereIndex]);

    // implementation of the Phong reflection model
    for (int i = 0; i < lightCount; i++)
    {
        vec3 lightDir(lights.DirectionX[i], lights.DirectionY[i], lights.DirectionZ[i]);
        vec3 lightColor(lights.ColorR[i], lights.ColorG[i], lights.ColorB[i]);

        vec3 hitpoint = normalize(origin + dir * closestHit);
        vec3 reflection = normalize(2.0f * dot(-lightDir, hitpoint) * hitpoint + lightDir);

        float diffuse = fmaxf(0.0f, dot(hitpoint, -lightDir));
        float specular = fmaxf(0.0f, dot(reflection, -normalize(dir)));
        specular = powf(specular, spheres.Shininess[sphereIndex]);

        color += lightColor * sphereColor * (diffuse * lights.Diffuse[i] + specular * lights.Specular[i]);
    }

    return color;
}

__global__ void TraceRays(cudaSurfaceObject_t surface, uint32_t width, uint32_t height,
    mat4<float> view, mat4<float> proj, vec3<float> position, size_t sphereCount,
    SphereRef spheres, size_t lightCount, LightRef lights)
{
    vec2 pixel(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y);

    vec3<float> color = TraceRay(pixel.x, pixel.y, width, height, view, proj, position, sphereCount, spheres, lightCount, lights);

    SurfaceSetPixelColor(surface, pixel.x, pixel.y, vec4(color, 1.0f));
}

}