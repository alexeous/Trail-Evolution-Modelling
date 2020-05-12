#pragma once
#include "cuda_runtime.h"
#include "CudaExceptionNative.h"

#define CHECK_CUDA(func){\
	cudaError_t __m = func;\
	if(__m != cudaSuccess) {\
		throw TrailEvolutionModelling::GPUProxy::CudaExceptionNative(\
			cudaGetErrorString(__m), __FILE__, __LINE__); } }

#define CHECK_CUDA_KERNEL(func){ func; CUDA_CHECK(cudaGetLastError()); }

