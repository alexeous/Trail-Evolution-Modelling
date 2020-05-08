#pragma once
#include "cuda_runtime.h"
#include "CudaException.h"

#define CHECK_CUDA(func){\
	cudaError_t __m = func;\
	if(__m != cudaSuccess) {\
		throw gcnew TrailEvolutionModelling::GPUProxy::CudaException(\
			cudaGetErrorString(__m), __FILE__, __LINE__); } }

#define CHECK_CUDA_KERNEL(func){ func; CUDA_CHECK(cudaGetLastError()); }