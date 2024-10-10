#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "Core/Vector.h"

#include "Models.h"

namespace Raycasting
{

__device__ __host__ vec3<float> TraceRay(uint32_t x, uint32_t y, uint32_t width, uint32_t height,
	const mat4<float> &view, const mat4<float> &proj, vec3<float> position, size_t sphereCount,
	const SphereRef &spheres, size_t lightCount, const LightRef &lights);

__global__ void TraceRays(cudaSurfaceObject_t surface, uint32_t width, uint32_t height,
	mat4<float> view, mat4<float> proj, vec3<float> position, size_t sphereCount,
	SphereRef spheres, size_t lightCount, LightRef lights);

}