#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <imgui.h>

#include <iostream>
#include <string>

#include "Core/Core.h"
#include "Renderer/Renderer.h"
#include "Window.h"

namespace Raycasting
{

static void GlfwErrorCallback(int error, const char *description)
{
	throw std::exception(std::format("GLFW error {} {}", error, description).c_str());
}

Window::Window(int width, int height, const char *title, bool vsync)
	: m_Width(width), m_Height(height), m_Vsync(vsync), m_VsyncChecked(vsync),
	m_FontSize(13), m_DesiredFontSize(13)
{
	int result = glfwInit();

#ifndef NDEBUG
	if (result == GLFW_FALSE)
		throw std::exception("Glfw initialization failed!");
#endif

	glfwSetErrorCallback(GlfwErrorCallback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

	m_Handle = glfwCreateWindow(width, height, title, nullptr, nullptr);

#ifndef NDEBUG
	if (m_Handle == nullptr)
		throw std::exception("Window creation failed!");
#endif

	glfwMakeContextCurrent(m_Handle);
	glfwSwapInterval(vsync);

	result = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

#ifndef NDEBUG
	if (result == GLFW_FALSE)
		throw std::exception("glad initalization failed!");
#endif

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	m_Io = &ImGui::GetIO();
	m_Io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	m_Io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	m_Io->IniFilename = nullptr;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(m_Handle, true);
	ImGui_ImplOpenGL3_Init("#version 460");
}

Window::~Window()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_Handle);
	glfwTerminate();
}

GLFWwindow *Window::GetHandle() const
{
	return m_Handle;
}

bool Window::ShouldClose()
{
	return glfwWindowShouldClose(m_Handle);
}

void Window::OnUpdate(float timeStep)
{
	if (m_FontSize != m_DesiredFontSize)
	{
		m_FontSize = m_DesiredFontSize;

		ImFontConfig cfg;
		cfg.SizePixels = m_FontSize;
		m_Io->Fonts->Clear();
		m_Io->Fonts->AddFontDefault(&cfg);
		m_Io->Fonts->Build();

		ImGui_ImplOpenGL3_DestroyFontsTexture();
	}

	if (m_Vsync != m_VsyncChecked)
	{
		m_Vsync = m_VsyncChecked;
		glfwSwapInterval(m_Vsync);
	}

	glfwPollEvents();

	if (glfwGetWindowAttrib(m_Handle, GLFW_ICONIFIED) != 0)
	{
		ImGui_ImplGlfw_Sleep(10);
		return;
	}
}

void Window::OnRender()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGuiID dockspaceId = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_AutoHideTabBar);

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0.0f, 0.0f });
	ImGui::Begin("Viewport");
	m_ViewportSize = ImGui::GetContentRegionAvail();
	ImGui::Image((ImTextureID)(uintptr_t)Renderer::GetTextureId(), m_ViewportSize, { 0, 1 }, { 1, 0 });
	ImGui::End();
	ImGui::PopStyleVar();

	ImGui::Begin("Settings");

	ImGui::Dummy(ImVec2(0.0f, 2.0f));
	ImGui::Checkbox("vsync", &m_VsyncChecked);

	ImGui::Dummy(ImVec2(0.0f, 5.0f));
	ImGui::InputInt("font size", &m_DesiredFontSize);

	ImGui::Dummy(ImVec2(0.0f, 5.0f));
	ImGui::Text("Execution:");
	if (ImGui::RadioButton("sequential (host)", Renderer::GetActiveRendererType() == RendererType::HostRenderer))
		Renderer::SelectRenderer(RendererType::HostRenderer);
	if (ImGui::RadioButton("parallel (device)", Renderer::GetActiveRendererType() == RendererType::DeviceRenderer))
		Renderer::SelectRenderer(RendererType::DeviceRenderer);

	ImGui::Dummy(ImVec2(0.0f, 5.0f));
	ImGui::Text("Active scene:");
	for (Scene *scene : Renderer::GetScenes())
	{
		if (ImGui::RadioButton(scene->GetName().c_str(), Renderer::GetActiveScene() == scene))
			Renderer::SelectScene(scene);
	}

	ImGui::End();

	ImGui::Begin("Info");
	Stats::AddStat("Framerate", "Framerate: {:.3f} ms/frame ({:.1f} FPS)", 1000.0f / m_Io->Framerate, m_Io->Framerate);

	for (const auto &stat : Stats::GetStats())
		ImGui::Text(stat.second.c_str());

	ImGui::End();

	static bool setup = true;
	if (setup)
	{
		setup = false;
		SetupDocking(dockspaceId);
	}

	Renderer::Resize((uint32_t)m_ViewportSize.x, (uint32_t)m_ViewportSize.y);
	Renderer::Render();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	glfwSwapBuffers(m_Handle);
}

void Window::SetupDocking(ImGuiID dockspaceId)
{
	ImGuiID viewportId, settingsId;
	ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Right, 0.2f, &settingsId, &viewportId);
	ImGui::DockBuilderSetNodeSize(settingsId, { std::min(400.0f, m_Width / 4.0f), m_Height / 2.0f });
	ImGui::DockBuilderDockWindow("Settings", settingsId);
	ImGui::DockBuilderDockWindow("Viewport", viewportId);

	ImGuiID infoId = ImGui::DockBuilderSplitNode(settingsId, ImGuiDir_Down, 0.2f, nullptr, nullptr);
	ImGui::DockBuilderSetNodeSize(infoId, { std::min(400.0f, m_Width / 4.0f), m_Height / 2.0f });
	ImGui::DockBuilderDockWindow("Info", infoId);

	ImGui::DockBuilderFinish(dockspaceId);
}

}
