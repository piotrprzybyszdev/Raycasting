#include <glad/glad.h>

#include <cuda_gl_interop.h>

#include <exception>
#include <string>

#include "Core/Core.h"
#include "Core/Vector.h"

#include "DeviceRenderer.h"
#include "Shaders.h"

namespace Raycasting
{

using CudaAssert = Assert<cudaError_t, cudaGetLastError, cudaSuccess, cudaGetErrorString>;

DeviceRenderer::DeviceRenderer()
{
}

DeviceRenderer::~DeviceRenderer()
{
}

void DeviceRenderer::InitImpl()
{
	Renderer::InitImpl();

	cudaSetDevice(0); CudaAssert();

	const size_t arraySize = MaxSphereCount * sizeof(float);

	cudaMalloc(&m_DeviceSpheres.PositionX, arraySize);
	cudaMalloc(&m_DeviceSpheres.PositionY, arraySize);
	cudaMalloc(&m_DeviceSpheres.PositionZ, arraySize);
	cudaMalloc(&m_DeviceSpheres.ColorR, arraySize);
	cudaMalloc(&m_DeviceSpheres.ColorG, arraySize);
	cudaMalloc(&m_DeviceSpheres.ColorB, arraySize);
	cudaMalloc(&m_DeviceSpheres.Radius, arraySize);
	cudaMalloc(&m_DeviceSpheres.Shininess, arraySize);
	CudaAssert();

	const size_t arraySize2 = MaxLightCount * sizeof(float);

	cudaMalloc(&m_DeviceLights.DirectionX, arraySize2);
	cudaMalloc(&m_DeviceLights.DirectionY, arraySize2);
	cudaMalloc(&m_DeviceLights.DirectionZ, arraySize2);
	cudaMalloc(&m_DeviceLights.ColorR, arraySize2);
	cudaMalloc(&m_DeviceLights.ColorG, arraySize2);
	cudaMalloc(&m_DeviceLights.ColorB, arraySize2);
	cudaMalloc(&m_DeviceLights.Diffuse, arraySize2);
	cudaMalloc(&m_DeviceLights.Specular, arraySize2);
	CudaAssert();
}

void DeviceRenderer::ShutdownImpl()
{
	cudaFree(m_DeviceLights.DirectionX);
	cudaFree(m_DeviceLights.DirectionY);
	cudaFree(m_DeviceLights.DirectionZ);
	cudaFree(m_DeviceLights.ColorR);
	cudaFree(m_DeviceLights.ColorG);
	cudaFree(m_DeviceLights.ColorB);
	cudaFree(m_DeviceLights.Diffuse);
	cudaFree(m_DeviceLights.Specular);

	cudaFree(m_DeviceSpheres.PositionX);
	cudaFree(m_DeviceSpheres.PositionY);
	cudaFree(m_DeviceSpheres.PositionZ);
	cudaFree(m_DeviceSpheres.ColorR);
	cudaFree(m_DeviceSpheres.ColorG);
	cudaFree(m_DeviceSpheres.ColorB);
	cudaFree(m_DeviceSpheres.Radius);
	cudaFree(m_DeviceSpheres.Shininess);

	cudaDestroySurfaceObject(m_Surface); CudaAssert();
	cudaGraphicsUnmapResources(1, &m_TextureResource); CudaAssert();
	cudaGraphicsUnregisterResource(m_TextureResource); CudaAssert();
	m_TextureResource = nullptr;

	cudaDeviceReset(); CudaAssert();

	Renderer::ShutdownImpl();
}

void DeviceRenderer::ResizeImpl(uint32_t width, uint32_t height)
{
	bool noresize = m_ViewportWidth == width && m_ViewportHeight == height;

	Renderer::ResizeImpl(width, height);

	if (noresize) return;

	// drop cuda's lock on the buffer
	if (m_TextureResource != nullptr)
	{
		cudaDestroySurfaceObject(m_Surface); CudaAssert();
		cudaGraphicsUnmapResources(1, &m_TextureResource); CudaAssert();
		cudaGraphicsUnregisterResource(m_TextureResource); CudaAssert();
	}

	// resize the buffer with opengl
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_ViewportWidth, m_ViewportHeight,
		0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr); GlAssert();

	// lock the buffer again to cuda
	cudaGraphicsGLRegisterImage(&m_TextureResource, m_TextureId, GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsSurfaceLoadStore); CudaAssert();
	cudaGraphicsMapResources(1, &m_TextureResource); CudaAssert();

	cudaArray *array;
	cudaGraphicsSubResourceGetMappedArray(&array, m_TextureResource, 0, 0); CudaAssert();

	cudaResourceDesc desc;
	memset(&desc, 0, sizeof(cudaResourceDesc));
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = array;

	cudaCreateSurfaceObject(&m_Surface, &desc);
	CudaAssert();
}

void DeviceRenderer::RenderImpl()
{
	Renderer::RenderImpl();

	{
		Timer timer("Scene Upload");

		const SphereRef &spheresRef = s_ActiveScene->GetSpheres();
		const LightRef &lightsRef = s_ActiveScene->GetLights();

		const size_t arraySize = s_ActiveScene->GetSphereCount() * sizeof(float);
		
		cudaMemcpy(m_DeviceSpheres.PositionX, spheresRef.PositionX, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.PositionY, spheresRef.PositionY, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.PositionZ, spheresRef.PositionZ, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.ColorR, spheresRef.ColorR, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.ColorG, spheresRef.ColorG, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.ColorB, spheresRef.ColorB, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.Radius, spheresRef.Radius, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceSpheres.Shininess, spheresRef.Shininess, arraySize, cudaMemcpyHostToDevice);
		CudaAssert();

		const size_t arraySize2 = s_ActiveScene->GetLightCount() * sizeof(float);

		cudaMemcpy(m_DeviceLights.DirectionX, lightsRef.DirectionX, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.DirectionY, lightsRef.DirectionY, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.DirectionZ, lightsRef.DirectionZ, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.ColorR, lightsRef.ColorR, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.ColorG, lightsRef.ColorG, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.ColorB, lightsRef.ColorB, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.Diffuse, lightsRef.Diffuse, arraySize2, cudaMemcpyHostToDevice);
		cudaMemcpy(m_DeviceLights.Specular, lightsRef.Specular, arraySize2, cudaMemcpyHostToDevice);
		CudaAssert();
	}

	Camera &camera = s_ActiveScene->GetCamera();
	mat4<float> view = camera.GetInvViewMatrix();
	mat4<float> proj = camera.GetInvProjectionMatrix();
	vec3<float> position = camera.GetPosition();

	int tx = 8, ty = 8;
	dim3 blocks(m_ViewportWidth / tx + 1, m_ViewportHeight / ty + 1);
	dim3 threads(tx, ty);

	{
		Timer timer("Pixel Shader");
		TraceRays<<<blocks, threads>>>(m_Surface, m_ViewportWidth, m_ViewportHeight, view,  proj, position,
			s_ActiveScene->GetSphereCount(), m_DeviceSpheres.GetRef(), s_ActiveScene->GetLightCount(), m_DeviceLights.GetRef());

		cudaDeviceSynchronize();
		CudaAssert();
	}
}

}