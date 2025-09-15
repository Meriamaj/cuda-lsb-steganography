#include "chronoCPU.hpp"

#include <iostream>

ChronoCPU::ChronoCPU()
	: _started(false)
{
#ifdef _WIN32
	QueryPerformanceFrequency(&_frequency);
#endif
}

ChronoCPU::~ChronoCPU()
{
	if (_started)
	{
		std::cerr << "ChronoCPU::~ChronoCPU(): chrono wasn't turned off!" << std::endl;
	}
}

void ChronoCPU::start()
{
	if (!_started)
	{
		_started = true;
#ifdef _WIN32
		QueryPerformanceCounter(&_start);
#else
		gettimeofday(&_start, NULL);
#endif
	}
	else
	{
		std::cerr << "ChronoGPU::start(): chrono has already started!" << std::endl;
	}
}

void ChronoCPU::stop()
{
	if (_started)
	{
		_started = false;
#ifdef _WIN32
		QueryPerformanceCounter(&_stop);
#else
		gettimeofday(&_stop, NULL);
#endif
	}
	else
	{
		std::cerr << "ChronoCPU::stop(): chrono hadn't started!" << std::endl;
	}
}

float ChronoCPU::elapsedTime()
{
	float time = 0.f;
	if (_started)
	{
		std::cerr << "ChronoCPU::elapsedTime(): chrono is still running!" << std::endl;
	}
	else
	{
#ifdef _WIN32
		time = ((float)(_stop.QuadPart - _start.QuadPart) / (float)(_frequency.QuadPart)) * 1e3f;
#else
		time = static_cast<float>(_stop.tv_sec - _start.tv_sec) * 1e3f + static_cast<float>(_stop.tv_usec - _start.tv_usec) / 1e3f;
#endif
	}
	return time;
}
