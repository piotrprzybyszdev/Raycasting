#pragma once

#include "Core/Camera.h"

#include "Renderer.h"
#include "Scene.h"

namespace Raycasting
{

class HostRenderer : public Renderer
{
public:
	HostRenderer();
	~HostRenderer() override;

private:
	void InitImpl() override;
	void ShutdownImpl() override;

	void ResizeImpl(uint32_t width, uint32_t height) override;
	void RenderImpl() override;

	uint32_t *m_ColorBuffer = nullptr;
};

}
