#ifndef __CHRONO_CPU_HPP__
#define __CHRONO_CPU_HPP__

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class ChronoCPU
{
public:
	ChronoCPU();
	~ChronoCPU();

	void start();
	void stop();
	float elapsedTime();

private:
	bool _started;
#ifdef _WIN32
	LARGE_INTEGER _frequency;
	LARGE_INTEGER _start;
	LARGE_INTEGER _stop;
#else
	timeval _start;
	timeval _stop;
#endif
};

#endif // __CHRONO_CPU_HPP__
