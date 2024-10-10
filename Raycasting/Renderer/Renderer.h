#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Core/Core.h"
#include "Core/Camera.h"

#include "Scene.h"

namespace Raycasting
{

static inline GLenum GlGetError()
{
	return glGetError();
}

using GlAssert = Assert<GLenum, GlGetError, GL_NO_ERROR>;

enum class RendererType
{
	NoRenderer, HostRenderer, DeviceRenderer
};

class Renderer
{
public:
	static void SelectRenderer(RendererType type);
	static RendererType GetActiveRendererType();

	static void Init(RendererType type, Scene *scene);
	static void Shutdown();

	static void Resize(uint32_t width, uint32_t height);
	static void Render();

	static uint32_t GetTextureId();

	static void AddScene(Scene *scene);
	static std::vector<Scene*> &GetScenes();
	static Scene *GetActiveScene();
	static void SelectScene(Scene *scene);

	static constexpr size_t MaxSphereCount = 1000;
	static constexpr size_t MaxLightCount = 10;

protected:
	Renderer();
	virtual ~Renderer();

	virtual void InitImpl();
	virtual void ShutdownImpl();

	virtual void ResizeImpl(uint32_t width, uint32_t height);
	virtual void RenderImpl();

	static Scene *s_ActiveScene;

	uint32_t m_ViewportHeight = 0;
	uint32_t m_ViewportWidth = 0;

	uint32_t m_TextureId = 0;

private:
	static std::vector<Scene*> s_Scenes;

	static RendererType s_ActiveRendererType;
	static Renderer *s_ActiveRenderer;
};

}
