#include "chronoGPU.hpp"
#include "commonCUDA.hpp"
#include <iostream>

ChronoGPU::ChronoGPU()
	: _started(false)
{
	HANDLE_ERROR(cudaEventCreate(&_start));
	HANDLE_ERROR(cudaEventCreate(&_end));
}

ChronoGPU::~ChronoGPU()
{
	if (_started)
	{
		std::cerr << "ChronoGPU::~ChronoGPU(): chrono wasn't turned off!" << std::endl;
	}
	HANDLE_ERROR(cudaEventDestroy(_start));
	HANDLE_ERROR(cudaEventDestroy(_end));
}

void ChronoGPU::start()
{
	if (!_started)
	{
		HANDLE_ERROR(cudaEventRecord(_start, 0));
		_started = true;
	}
	else
	{
		std::cerr << "ChronoGPU::start(): chrono has already started!" << std::endl;
	}
}

void ChronoGPU::stop()
{
	if (_started)
	{
		HANDLE_ERROR(cudaEventRecord(_end, 0));
		HANDLE_ERROR(cudaEventSynchronize(_end));
		_started = false;
	}
	else
	{
		std::cerr << "ChronoGPU::stop(): chrono hadn't started!" << std::endl;
	}
}

float ChronoGPU::elapsedTime()
{
	float time = 0.f;
	if (_started)
	{
		std::cerr << "ChronoGPU::elapsedTime(): chrono is still running!" << std::endl;
	}
	else
	{
		HANDLE_ERROR(cudaEventElapsedTime(&time, _start, _end));
	}
	return time;
}
