#include <glm/glm.hpp>

#include "Scene.h"

namespace Raycasting
{

Scene::Scene(std::string &&name, Camera &&camera) : m_Name(name), m_Camera(camera)
{
}

Scene::~Scene()
{
}

const std::string &Scene::GetName() const
{
	return m_Name;
}

Camera &Scene::GetCamera()
{
	return m_Camera;
}

void Scene::AddSphere(Sphere &&sphere)
{
	m_SpherePositionX.push_back(sphere.Position.x);
	m_SpherePositionY.push_back(sphere.Position.y);
	m_SpherePositionZ.push_back(sphere.Position.z);
	m_SphereColorR.push_back(sphere.Color.x);
	m_SphereColorG.push_back(sphere.Color.y);
	m_SphereColorB.push_back(sphere.Color.z);
	m_SphereRadius.push_back(sphere.Radius);
	m_SphereShininess.push_back(sphere.Shininess);
}

void Scene::AddLight(Light &&light)
{
	m_LightDirectionX.push_back(light.Direction.x);
	m_LightDirectionY.push_back(light.Direction.y);
	m_LightDirectionZ.push_back(light.Direction.z);
	m_LightColorR.push_back(light.Color.x);
	m_LightColorG.push_back(light.Color.y);
	m_LightColorB.push_back(light.Color.z);
	m_LightDiffuse.push_back(light.Diffuse);
	m_LightSpecular.push_back(light.Specular);
}

void Scene::OnUpdate(float timeStep)
{
	m_Camera.OnUpdate(timeStep);

	constexpr float angularVelocity = 0.8f;
	const float angle = timeStep * angularVelocity;

	glm::mat3 lightRotation {
		glm::cos(angle), 0, glm::sin(angle),
		0, 1, 0,
		-glm::sin(angle), 0, glm::cos(angle)
	};

	for (int i = 0; i < GetLightCount(); i++)
	{
		glm::vec3 lightDir { m_LightDirectionX[i], m_LightDirectionY[i], m_LightDirectionZ[i] };
		glm::vec3 lightDirNew = lightRotation * lightDir;
		m_LightDirectionX[i] = lightDirNew.x;
		m_LightDirectionY[i] = lightDirNew.y;
		m_LightDirectionZ[i] = lightDirNew.z;
	}
}

size_t Scene::GetSphereCount() const
{
	return m_SpherePositionX.size();
}

SphereRef Scene::GetSpheres() const
{
	return SphereRef{
		.PositionX = m_SpherePositionX.data(),
		.PositionY = m_SpherePositionY.data(),
		.PositionZ = m_SpherePositionZ.data(),
		.ColorR = m_SphereColorR.data(),
		.ColorG = m_SphereColorG.data(),
		.ColorB = m_SphereColorB.data(),
		.Radius = m_SphereRadius.data(),
		.Shininess = m_SphereShininess.data()
	};
}

size_t Scene::GetLightCount() const
{
	return m_LightDirectionX.size();
}

LightRef Scene::GetLights() const
{
	return LightRef{
		.DirectionX = m_LightDirectionX.data(),
		.DirectionY = m_LightDirectionY.data(),
		.DirectionZ = m_LightDirectionZ.data(),
		.ColorR = m_LightColorR.data(),
		.ColorG = m_LightColorG.data(),
		.ColorB = m_LightColorB.data(),
		.Diffuse = m_LightDiffuse.data(),
		.Specular = m_LightSpecular.data()
	};
}

}
