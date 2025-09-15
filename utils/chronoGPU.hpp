#ifndef __CHRONO_GPU_HPP__
#define __CHRONO_GPU_HPP__

class ChronoGPU
{
public:
	ChronoGPU();
	~ChronoGPU();

	void start();
	void stop();
	void reset();
	float elapsedTime();

private:
	cudaEvent_t _start;
	cudaEvent_t _end;

	bool _started;
};

#endif // __CHRONO_GPU_HPP__
