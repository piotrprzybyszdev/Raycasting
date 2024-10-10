#pragma once

#include <Core/Vector.h>

namespace Raycasting
{

struct Sphere
{
	vec3<float> Position;
	vec3<float> Color;
	float Radius;
	float Shininess;
};

struct Light
{
	vec3<float> Direction;
	vec3<float> Color;
	float Diffuse;
	float Specular;
};

struct SphereRef
{
	const float *PositionX;
	const float *PositionY;
	const float *PositionZ;
	const float *ColorR;
	const float *ColorG;
	const float *ColorB;
	const float *Radius;
	const float *Shininess;
};

struct LightRef
{
	const float *DirectionX;
	const float *DirectionY;
	const float *DirectionZ;
	const float *ColorR;
	const float *ColorG;
	const float *ColorB;
	const float *Diffuse;
	const float *Specular;
};

struct Spheres
{
	float *PositionX;
	float *PositionY;
	float *PositionZ;
	float *ColorR;
	float *ColorG;
	float *ColorB;
	float *Radius;
	float *Shininess;

	SphereRef GetRef()
	{
		return SphereRef{
			.PositionX = PositionX,
			.PositionY = PositionY,
			.PositionZ = PositionZ,
			.ColorR = ColorR,
			.ColorG = ColorG,
			.ColorB = ColorB,
			.Radius = Radius,
			.Shininess = Shininess
		};
	}
};

struct Lights
{
	float *DirectionX;
	float *DirectionY;
	float *DirectionZ;
	float *ColorR;
	float *ColorG;
	float *ColorB;
	float *Diffuse;
	float *Specular;

	LightRef GetRef()
	{
		return LightRef{
			.DirectionX = DirectionX,
			.DirectionY = DirectionY,
			.DirectionZ = DirectionZ,
			.ColorR = ColorR,
			.ColorG = ColorG,
			.ColorB = ColorB,
			.Diffuse = Diffuse,
			.Specular = Specular
		};
	}
};

}
