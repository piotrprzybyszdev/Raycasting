#include <glad/glad.h>

#include "HostRenderer.h"
#include "Shaders.h"

namespace Raycasting
{

HostRenderer::HostRenderer()
{
}

HostRenderer::~HostRenderer()
{
}

void HostRenderer::InitImpl()
{
	Renderer::InitImpl();
}

void HostRenderer::ShutdownImpl()
{
	delete[] m_ColorBuffer;
	m_ColorBuffer = nullptr;

	Renderer::ShutdownImpl();
}

void HostRenderer::ResizeImpl(uint32_t width, uint32_t height)
{
	bool noresize = m_ViewportWidth == width && m_ViewportHeight == height;

	Renderer::ResizeImpl(width, height);

	if (noresize) return;

	delete[] m_ColorBuffer;

	m_ColorBuffer = new uint32_t[m_ViewportHeight * m_ViewportWidth];
}

void HostRenderer::RenderImpl()
{
	Renderer::RenderImpl();

	Camera &camera = s_ActiveScene->GetCamera();
	vec3 position = camera.GetPosition();
	mat4 view = camera.GetInvViewMatrix();
	mat4 proj = camera.GetInvProjectionMatrix();

	const SphereRef &sphereRef = s_ActiveScene->GetSpheres();
	const LightRef &lightRef = s_ActiveScene->GetLights();

	{
		Timer timer("Pixel Shader");

		for (int y = 0; y < m_ViewportHeight; y++)
		{
			for (int x = 0; x < m_ViewportWidth; x++)
			{
				vec3 color = TraceRay(x, y, m_ViewportWidth, m_ViewportHeight, view, proj, position,
					s_ActiveScene->GetSphereCount(), sphereRef, s_ActiveScene->GetLightCount(), lightRef);

				vec4<uint8_t> rgba = clamp(vec4(color, 1.0f), 0.0f, 1.0f) * 255.0f;

				uint32_t bytes = rgba.w << 24 | rgba.z << 16 | rgba.y << 8 | rgba.x;

				m_ColorBuffer[y * m_ViewportWidth + x] = bytes;
			}
		}
	}

	{
		Timer timer("Image Upload");
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_ViewportWidth, m_ViewportHeight,
			0, GL_RGBA, GL_UNSIGNED_BYTE, m_ColorBuffer); GlAssert();
	}
}

}
