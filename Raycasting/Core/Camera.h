#pragma once

#include <GLFW/glfw3.h>

#include <Core/Vector.h>

namespace Raycasting
{

class Camera
{
public:
	Camera(float verticalFOV, float nearClip, float farClip);
	~Camera();

	void OnUpdate(float timeStep);
	void OnResize(uint32_t width, uint32_t height);

	mat4<float> GetInvViewMatrix() const;
	mat4<float> GetInvProjectionMatrix() const;

	vec3<float> GetPosition() const;
	vec3<float> GetDirection() const;

private:
	static constexpr float CameraSpeed = 5.0f;
	static constexpr float MouseSensitivity = 0.05f;

	const float m_VerticalFOV;
	const float m_NearClip;
	const float m_FarClip;

	uint32_t m_Width = 0;
	uint32_t m_Height = 0;

	vec2<float> m_PreviousMousePos;

	float m_Yaw;
	float m_Pitch;

	vec3<float> m_Position;
	vec3<float> m_Direction;

	mat4<float> m_InvView;
	mat4<float> m_InvProjection;
};

}