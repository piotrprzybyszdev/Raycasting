#pragma once

#include <string>
#include <vector>

#include "Core/Camera.h"
#include "Core/Vector.h"

#include "Models.h"

namespace Raycasting
{

class Scene
{
public:
	Scene(std::string &&name, Camera &&camera);
	~Scene();

	const std::string &GetName() const;
	Camera &GetCamera();

	void AddSphere(Sphere &&sphere);
	void AddLight(Light &&light);

	void OnUpdate(float timeStep);

	size_t GetSphereCount() const;
	SphereRef GetSpheres() const;
	size_t GetLightCount() const;
	LightRef GetLights() const;

private:
	const std::string m_Name;
	Camera m_Camera;

	std::vector<float> m_SpherePositionX;
	std::vector<float> m_SpherePositionY;
	std::vector<float> m_SpherePositionZ;
	std::vector<float> m_SphereColorR;
	std::vector<float> m_SphereColorG;
	std::vector<float> m_SphereColorB;
	std::vector<float> m_SphereRadius;
	std::vector<float> m_SphereShininess;

	std::vector<float> m_LightDirectionX;
	std::vector<float> m_LightDirectionY;
	std::vector<float> m_LightDirectionZ;
	std::vector<float> m_LightColorR;
	std::vector<float> m_LightColorG;
	std::vector<float> m_LightColorB;
	std::vector<float> m_LightDiffuse;
	std::vector<float> m_LightSpecular;
};

}
