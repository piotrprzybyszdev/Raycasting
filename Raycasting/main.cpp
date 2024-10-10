#include <iostream>

#include "Core/Camera.h"
#include "Core/Core.h"
#include "Core/Input.h"
#include "Renderer/Renderer.h"

#include "Scene.h"
#include "Window.h"

using namespace Raycasting;

float random(float a, float b)
{
	return a + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (b - a)));
}

int main()
{
	try
	{
		Scene demoScene("Demo", Camera(45.0f, 100, 0.1f));
		Scene stressTestScene("Stress test", Camera(45.0f, 100, 0.1f));

		demoScene.AddSphere(Sphere{ glm::vec3(0.5f, 0, 0), glm::vec3(0, 1, 1), 0.5f, 32.0f });
		demoScene.AddSphere(Sphere{ glm::vec3(0.5f, 0.5f, -2), glm::vec3(1, 1, 0), 1, 64.0f });

		demoScene.AddLight(Light{ glm::vec3(-1, -1, -1), glm::vec3(1, 1, 1), 0.5f, 0.5f });

		for (int i = 0; i < Renderer::MaxSphereCount; i++)
		{
			stressTestScene.AddSphere(Sphere {
				glm::vec3(random(-30, 30), random(-15, 15), random(0, -50)),
				glm::vec3(random(0, 1), random(0, 1), random(0, 1)),
				random(0.3f, 1.0f), random(0.1f, 100.0f)
			});
		}

		for (int i = 0; i < Renderer::MaxLightCount; i++)
		{
			stressTestScene.AddLight(Light{
				glm::normalize(glm::vec3(random(-1, 1), random(-1, 1), random(-1, 1))),
				glm::vec3(random(0, 1), random(0, 1), random(0, 1)),
				random(0, 1), random(0, 1)
			});
		}

		Window window(1280, 720, "Sphere Raycasting", true);

		Renderer::Init(RendererType::DeviceRenderer, &stressTestScene);
		Renderer::AddScene(&demoScene);

		Input::SetWindow(window.GetHandle());

		float lastFrameTime = 0.0f;
		while (!window.ShouldClose())
		{
			Timer timer("Frame total");

			float time = glfwGetTime();

			float timeStep = time - lastFrameTime;
			lastFrameTime = time;

			{
				Timer timer("Update");
				Renderer::GetActiveScene()->OnUpdate(timeStep);
				window.OnUpdate(timeStep);
			}

			{
				Timer timer("Render");
				window.OnRender();
			}
		}

		Renderer::Shutdown();
	}
	catch (std::exception exception)
	{
		std::cerr << exception.what() << std::endl;

		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}
