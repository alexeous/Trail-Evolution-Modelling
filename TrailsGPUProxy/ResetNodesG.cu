#include "ResetNodesG.h"
#include <cmath>
#include "MathUtils.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		__global__ void ResetNodesGKernel(ComputeNode* nodes, int extendedW, int extendedH, int goalIndex) {
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			if(i < extendedW && j < extendedH) {
				int index = i + j * extendedW;

				nodes[index].g = (index == goalIndex ? 0 : INFINITY);
			}
		}

		cudaError_t ResetNodesG(ComputeNode* nodes, int extendedW, int extendedH, int goalIndex) {
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(divceil(extendedW, BLOCK_SIZE_X), divceil(extendedH , BLOCK_SIZE_Y));

			ResetNodesGKernel<<<blocksDim, threadsDim>>>(nodes, extendedW, extendedH, goalIndex);

			return cudaGetLastError();
		}

	}
}