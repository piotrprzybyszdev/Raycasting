#include "Core.h"

#include <iostream>

namespace Raycasting
{

std::map<std::string, std::string> Stats::s_Stats = {};

void Stats::Clear()
{
	s_Stats.clear();
}

const std::map<std::string, std::string> &Stats::GetStats()
{
	return s_Stats;
}

Timer::Timer(std::string&& name)
	: m_Name(name), m_Start(std::chrono::high_resolution_clock::now())
{
}

Timer::~Timer()
{
	Stats::AddStat(m_Name, "{}: {:.3f} ms", m_Name,
		std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::high_resolution_clock::now() - m_Start
		).count() / 1000.0f
	);
}

void ThrowError(std::source_location location, const char *message)
{
	throw std::exception(
		std::format(
			"Assertion failed at {}({}:{}): {}: {}", location.file_name(),
			location.line(), location.column(), location.function_name(), message
		).c_str()
	);
}

}