#pragma once

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <cstdint>

#include "Renderer/Renderer.h"

namespace Raycasting
{

class Window
{
public:
	Window(int width, int height, const char *title, bool vsync);
	virtual ~Window();

	bool ShouldClose();
	void OnUpdate(float timeStep);
	void OnRender();

	GLFWwindow *GetHandle() const;

private:
	int m_Width, m_Height;
	bool m_Vsync;

	GLFWwindow *m_Handle;
	ImGuiIO *m_Io;

	ImVec2 m_ViewportSize;

	bool m_VsyncChecked;
	int m_DesiredFontSize;
	int m_FontSize;

	void SetupDocking(ImGuiID dockspaceId);
};

}
