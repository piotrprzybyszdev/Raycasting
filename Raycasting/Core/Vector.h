#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace Raycasting
{

template<typename T = float>
struct vec2
{
	T x, y;

	__host__ __device__ __inline__ constexpr vec2() : x(T()), y(T()) {}
	__host__ __device__ __inline__ constexpr vec2(T x, T y) : x(x), y(y) {}

	template<typename V>
	__host__ __device__ __inline__ constexpr vec2<T>(V v) : x(v.x), y(v.y) {}

	__host__ __device__ __inline__ constexpr vec2<T> operator +(const vec2<T> &other) const
	{
		return vec2<T>(x + other.x, y + other.y);
	}

	__host__ __device__ __inline__ constexpr vec2<T> operator +(T other) const
	{
		return vec2<T>(x + other, y + other);
	}

	__host__ __device__ __inline__ constexpr vec2<T> operator -(T other) const
	{
		return vec2<T>(x - other, y - other);
	}

	__host__ __device__ __inline__ constexpr vec2<T> operator *(T scale) const
	{
		return vec2<T>(x * scale, y * scale);
	}

	__host__ __device__ __inline__ constexpr vec2<T> operator /(T scale) const
	{
		return vec2<T>(x / scale, y / scale);
	}

	__host__ __device__ __inline__ friend constexpr vec2<T> operator*(T scale, const vec2<T> &vec)
	{
		return vec2<T>(vec.x * scale, vec.y * scale);
	}
};

template<typename T = float>
struct vec3
{
	T x, y, z;

	__host__ __device__ __inline__ constexpr vec3() : x(T()), y(T()), z(T()) {}
	__host__ __device__ __inline__ constexpr vec3(T x, T y, T z) : x(x), y(y), z(z) {}
	__host__ __device__ __inline__ constexpr vec3(vec2<T> v, T z) : x(v.x), y(v.y), z(z) {}
	
	template<typename V>
	__host__ __device__ __inline__ constexpr vec3<T>(V v) : x(v.x), y(v.y), z(v.z) {}

	__host__ __device__ __inline__ constexpr vec3<T> operator +(const vec3<T> &other) const
	{
		return vec3<T>(x + other.x, y + other.y, z + other.z);
	}

	__host__ __device__ __inline__ constexpr vec3<T> operator -(const vec3<T> &other) const
	{
		return vec3<T>(x - other.x, y - other.y, z - other.z);
	}

	__host__ __device__ __inline__ constexpr vec3<T> operator *(const vec3<T> &other) const
	{
		return vec3<T>(x * other.x, y * other.y, z * other.z);
	}

	__host__ __device__ __inline__ constexpr vec3<T> operator *(T scale) const
	{
		return vec3<T>(x * scale, y * scale, z * scale);
	}

	__host__ __device__ __inline__ constexpr vec3<T> operator /(T scale) const
	{
		return vec3<T>(x / scale, y / scale, z / scale);
	}

	__host__ __device__ __inline__ friend constexpr vec3<T> operator*(T scale, const vec3<T> &vec)
	{
		return vec3<T>(scale * vec.x, scale * vec.y, scale * vec.z);
	}

	__host__ __device__ __inline__ friend constexpr vec3<T> operator/(T div, const vec3<T> &vec)
	{
		return vec3<T>(div / vec.x, div / vec.y, div / vec.z);
	}

	__host__ __device__ __inline__ constexpr vec3<T> operator-()
	{
		return vec3<T>(-x, -y, -z);
	}

	__host__ __device__ __inline__ constexpr vec3<T>& operator+=(const vec3<T> &other)
	{
		x += other.x; y += other.y; z += other.z;
		return *this;
	}

	__host__ __device__ __inline__ constexpr vec3<T>& operator-=(const vec3<T> &other)
	{
		x -= other.x; y -= other.y; z -= other.z;
		return *this;
	}

	__host__ __device__ __inline__ constexpr bool operator==(const vec3<T> &other)
	{
		return x == other.x && y == other.y && z == other.z;
	}

	__host__ __device__ __inline__ constexpr bool operator!=(const vec3<T> &other)
	{
		return x != other.x || y != other.y || z != other.z;
	}
};

template<typename T = float>
struct vec4
{
	T x, y, z, w;

	__host__ __device__ __inline__ constexpr vec4() : x(T()), y(T()), z(T()), w(T()) {}
	__host__ __device__ __inline__ constexpr vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
	__host__ __device__ __inline__ constexpr vec4(vec2<T> v, T z, T w) : x(v.x), y(v.y), z(z), w(w) {}
	__host__ __device__ __inline__ constexpr vec4(vec3<T> v, T w) : x(v.x), y(v.y), z(v.z), w(w) {}

	__host__ __device__ __inline__ constexpr vec4<T> operator +(const vec4<T> &other) const
	{
		return vec4<T>(x + other.x, y + other.y, z + other.z, w + other.w);
	}

	__host__ __device__ __inline__ constexpr vec4<T> operator *(T scale) const
	{
		return vec4<T>(x * scale, y * scale, z * scale, w * scale);
	}

	__host__ __device__ __inline__ constexpr vec4<T> operator /(T scale) const
	{
		return vec4<T>(x / scale, y / scale, z / scale, w / scale);
	}

	__host__ __device__ __inline__ friend constexpr vec4<T> operator*(T scale, const vec4<T> &vec)
	{
		return vec4<T>(vec.x * scale, vec.y * scale, vec.z * scale, vec.w * scale);
	}

	__host__ __device__ __inline__ operator vec3<T>()
	{
		return vec3<T>(x, y, z);
	}

	template<typename U>
	__host__ __device__ __inline__ operator vec4<U>()
	{
		return vec4<U>(x, y, z, w);
	}
};

template<typename T>
__host__ __device__ __inline__ constexpr T dot(const vec3<T> &a, const vec3<T> &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<typename T>
__host__ __device__ __inline__ constexpr T dot(const vec4<T> &a, const vec4<T> &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<typename T>
__host__ __device__ __inline__ constexpr vec3<T> normalize(const vec3<T> &vec)
{
	float length = sqrtf(dot(vec, vec));
	return vec / length;
}

template<typename T>
__host__ __device__ __inline__ constexpr vec4<T> normalize(const vec4<T> &vec)
{
	float length = sqrtf(dot(vec, vec));
	return vec / length;
}

template<typename T>
__host__ __device__ __inline__ constexpr T clamp(const T val, const T low, const T high)
{
	return val < low ? low : val > high ? high : val;
}

template<typename T>
__host__ __device__ __inline__ constexpr vec4<T> clamp(const vec4<T> &vec, T low, T high)
{
	return vec4<T>(clamp(vec.x, low, high), clamp(vec.y, low, high), clamp(vec.z, low, high), clamp(vec.w, low, high));
}

template<typename T = float>
struct mat4
{
	T arr[4][4] = { 0 };

	__host__ __device__ __inline__ constexpr mat4<T>() {}

	template<typename M>
	__host__ __device__ __inline__ constexpr mat4<T>(const M &mat)
	{
		arr[0][0] = mat[0][0];
		arr[0][1] = mat[0][1];
		arr[0][2] = mat[0][2];
		arr[0][3] = mat[0][3];
		arr[1][0] = mat[1][0];
		arr[1][1] = mat[1][1];
		arr[1][2] = mat[1][2];
		arr[1][3] = mat[1][3];
		arr[2][0] = mat[2][0];
		arr[2][1] = mat[2][1];
		arr[2][2] = mat[2][2];
		arr[2][3] = mat[2][3];
		arr[3][0] = mat[3][0];
		arr[3][1] = mat[3][1];
		arr[3][2] = mat[3][2];
		arr[3][3] = mat[3][3];
	}

	__host__ __device__ __inline__ constexpr vec4<T> operator*(const vec4<T> &vec) const
	{
		return {
			arr[0][0] * vec.x + arr[1][0] * vec.y + arr[2][0] * vec.z + arr[3][0] * vec.w,
			arr[0][1] * vec.x + arr[1][1] * vec.y + arr[2][1] * vec.z + arr[3][1] * vec.w,
			arr[0][2] * vec.x + arr[1][2] * vec.y + arr[2][2] * vec.z + arr[3][2] * vec.w,
			arr[0][3] * vec.x + arr[1][3] * vec.y + arr[2][3] * vec.z + arr[3][3] * vec.w,
		};
	}
};

}