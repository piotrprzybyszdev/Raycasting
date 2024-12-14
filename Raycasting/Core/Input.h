#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace Raycasting
{
	
enum MouseButton : uint8_t
{
	Left = GLFW_MOUSE_BUTTON_LEFT,
	Right = GLFW_MOUSE_BUTTON_RIGHT
};

enum Key : uint16_t
{
	A = GLFW_KEY_A,
	D = GLFW_KEY_D,
	E = GLFW_KEY_E,
	Q = GLFW_KEY_Q,
	S = GLFW_KEY_S,
	W = GLFW_KEY_W
};

class Input
{
public:
	static void SetWindow(GLFWwindow* window);

	static void LockCursor();
	static void UnlockCursor();

	static bool IsKeyPressed(Key key);
	static bool IsMouseButtonPressed(MouseButton mouseButton);

	static glm::vec2 GetMousePosition();

private:
	static GLFWwindow* s_Window;
};

}