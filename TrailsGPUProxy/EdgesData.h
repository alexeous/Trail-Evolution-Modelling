#pragma once
#include "CudaUtils.h"
#include "cuda_runtime.h"


template<typename T>
struct EdgesData {
	T* vertical;
	T* horizontal;
	T* leftDiagonal;
	T* rightDiagonal;

	inline T& NW(int i, int j, int w);
	inline T& N(int i, int j, int w);
	inline T& NE(int i, int j, int w);
	inline T& W(int i, int j, int w);
	inline T& E(int i, int j, int w);
	inline T& SW(int i, int j, int w);
	inline T& S(int i, int j, int w);
	inline T& SE(int i, int j, int w);
};

template<typename T>
struct EdgesDataHost : public EdgesData<T> {
	EdgesDataHost(int w, int h);
	~EdgesDataHost();
};

template<typename T>
struct EdgesDataDevice : public EdgesData<T> {
	EdgesDataDevice(int w, int h);
	~EdgesDataDevice();
	static EdgesDataDevice<T> MakeFromHost(const EdgesDataHost<T>& host, int w, int h);
};



template<typename T> inline T& EdgesData<T>::NW(int i, int j, int w) { return leftDiagonal[i + j * w]; }
template<typename T> inline T& EdgesData<T>::N(int i, int j, int w) { return vertical[i + 1 + j * w]; }
template<typename T> inline T& EdgesData<T>::NE(int i, int j, int w) { return rightDiagonal[i + 1 + j * w]; }
template<typename T> inline T& EdgesData<T>::W(int i, int j, int w) { return horizontal[i + (j + 1) * w]; }
template<typename T> inline T& EdgesData<T>::E(int i, int j, int w) { return horizontal[i + 1 + (j + 1) * w]; }
template<typename T> inline T& EdgesData<T>::SW(int i, int j, int w) { return rightDiagonal[i + (j + 1) * w]; }
template<typename T> inline T& EdgesData<T>::S(int i, int j, int w) { return vertical[i + 1 + (j + 1) * w]; }
template<typename T> inline T& EdgesData<T>::SE(int i, int j, int w) { return leftDiagonal[i + 1 + (j + 1) * w]; }

template<typename T>
EdgesDataHost<T>::EdgesDataHost(int w, int h) {
	size_t size = w * h;
	CHECK_CUDA(cudaMallocHost((void**)&vertical, size));
	CHECK_CUDA(cudaMallocHost((void**)&horizontal, size));
	CHECK_CUDA(cudaMallocHost((void**)&leftDiagonal, size));
	CHECK_CUDA(cudaMallocHost((void**)&rightDiagonal, size));
}

template<typename T>
EdgesDataHost<T>::~EdgesDataHost() {
	cudaFreeHost(vertical);
	cudaFreeHost(horizontal);
	cudaFreeHost(leftDiagonal);
	cudaFreeHost(rightDiagonal);
}

template<typename T>
EdgesDataDevice<T>::EdgesDataDevice(int w, int h) {
	size_t size = w * h;
	CHECK_CUDA(cudaMalloc((void**)&vertical, size));
	CHECK_CUDA(cudaMalloc((void**)&horizontal, size));
	CHECK_CUDA(cudaMalloc((void**)&leftDiagonal, size));
	CHECK_CUDA(cudaMalloc((void**)&rightDiagonal, size));
}

template<typename T>
EdgesDataDevice<T>::~EdgesDataDevice() {
	cudaFree(vertical);
	cudaFree(horizontal);
	cudaFree(leftDiagonal);
	cudaFree(rightDiagonal);
}

template<typename T>
EdgesDataDevice<T> EdgesDataDevice<T>::MakeFromHost(const EdgesDataHost<T>& host, int w, int h) {
	EdgesDataDevice<T> device(w, h);
	size_t size = w * h;
	CHECK_CUDA(cudaMemcpy(vertical, host.vertical, size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(horizontal, host.horizontal, size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(leftDiagonal, host.leftDiagonal, size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(rightDiagonal, host.rightDiagonal, size, cudaMemcpyHostToDevice));
}

