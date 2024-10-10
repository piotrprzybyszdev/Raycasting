#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "Core/Camera.h"

#include "Models.h"
#include "Renderer.h"
#include "Scene.h"
#include "Shaders.h"

namespace Raycasting
{

class DeviceRenderer : public Renderer
{
public:
	DeviceRenderer();
	~DeviceRenderer() override;

private:
	void InitImpl() override;
	void ShutdownImpl() override;

	void ResizeImpl(uint32_t width, uint32_t height) override;
	void RenderImpl() override;

	cudaGraphicsResource *m_TextureResource;
	cudaSurfaceObject_t m_Surface;

	Spheres m_DeviceSpheres;
	Lights m_DeviceLights;
};

}
