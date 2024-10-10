#include <glad/glad.h>

#include <cassert>

#include "Renderer.h"
#include "DeviceRenderer.h"
#include "HostRenderer.h"

namespace Raycasting
{

Scene *Renderer::s_ActiveScene = nullptr;

std::vector<Scene*> Renderer::s_Scenes;

RendererType Renderer::s_ActiveRendererType = RendererType::NoRenderer;
Renderer *Renderer::s_ActiveRenderer = nullptr;

void Renderer::SelectRenderer(RendererType type)
{
	static HostRenderer HostRenderer;
	static DeviceRenderer DeviceRenderer;

	if (s_ActiveRendererType == type) return;

	if (s_ActiveRenderer)
		s_ActiveRenderer->ShutdownImpl();

	switch (type)
	{
	case RendererType::HostRenderer:
		s_ActiveRenderer = &HostRenderer;
		break;
	case RendererType::DeviceRenderer:
		s_ActiveRenderer = &DeviceRenderer;
		break;
	}

	s_ActiveRenderer->InitImpl();

	s_ActiveRendererType = type;
	Stats::Clear();
}

RendererType Renderer::GetActiveRendererType()
{
	return s_ActiveRendererType;
}

void Renderer::Init(RendererType type, Scene *scene)
{
	assert(type != RendererType::NoRenderer);

	s_Scenes.push_back(scene);
	s_ActiveScene = scene;

	SelectRenderer(type);
}

void Renderer::Shutdown()
{
	if (s_ActiveRenderer) 
		s_ActiveRenderer->ShutdownImpl();
	s_ActiveRenderer = nullptr;
}

void Renderer::Resize(uint32_t width, uint32_t height)
{
	assert(s_ActiveRendererType != RendererType::NoRenderer);
	s_ActiveRenderer->ResizeImpl(width, height);
}

void Renderer::Render()
{
	assert(s_ActiveRendererType != RendererType::NoRenderer);
	s_ActiveRenderer->RenderImpl();
}

uint32_t Renderer::GetTextureId()
{
	assert(s_ActiveRendererType != RendererType::NoRenderer);
	return s_ActiveRenderer->m_TextureId;
}

void Renderer::AddScene(Scene *scene)
{
	s_Scenes.push_back(scene);
}

std::vector<Scene*> &Renderer::GetScenes()
{
	return s_Scenes;
}

Scene *Renderer::GetActiveScene()
{
	return s_ActiveScene;
}

void Renderer::SelectScene(Scene *scene)
{
	s_ActiveScene = scene;
}

void Renderer::InitImpl()
{
	m_ViewportWidth = 0; m_ViewportHeight = 0;

	glGenTextures(1, &m_TextureId); GlAssert();
	glBindTexture(GL_TEXTURE_2D, m_TextureId); GlAssert();

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); GlAssert();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); GlAssert();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); GlAssert();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); GlAssert();
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0); GlAssert();
}

void Renderer::ShutdownImpl()
{
	glDeleteTextures(1, &m_TextureId);
}

void Renderer::ResizeImpl(uint32_t width, uint32_t height)
{
	s_ActiveScene->GetCamera().OnResize(width, height);

	if (m_ViewportWidth == width && m_ViewportHeight == height)
		return;

	m_ViewportWidth = width; m_ViewportHeight = height;
}

void Renderer::RenderImpl()
{
	assert(s_ActiveScene->GetSphereCount() <= MaxSphereCount);
	assert(s_ActiveScene->GetLightCount() <= MaxLightCount);

	GlAssert();
}

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
}

}
