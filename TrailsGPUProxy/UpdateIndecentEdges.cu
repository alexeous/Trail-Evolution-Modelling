#include "UpdateIndecentEdges.h"

#ifndef __CUDACC__ // visual studio doesn't show functions declared in device_atomic_function.h without this
#define UNDEF__CUDACC__
#define __CUDACC__
#endif
#include <math_functions.h>
#include <device_functions.h>
#ifdef UNDEF__CUDACC__
#undef UNDEF__CUDACC__
#undef __CUDACC__
#endif

#define BLOCK_SIZE_X UPDATE_INDECENT_EDGES_BLOCK_SIZE_X
#define BLOCK_SIZE_Y UPDATE_INDECENT_EDGES_BLOCK_SIZE_Y


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		inline __device__ void Update(float original, float current, float& target) {
			target = min(original, current);
		}

		__global__ void UpdateIndecentEdgesKernel(EdgesWeightsDevice original,
			EdgesWeightsDevice current, EdgesWeightsDevice target, int graphW, int graphH) 
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			if(i > graphW || j > graphH)
				return;

			int idx = i + j * (graphW + 1);
			Update(original.horizontal[idx], current.horizontal[idx], target.horizontal[idx]);
			Update(original.vertical[idx], current.vertical[idx], target.vertical[idx]);
			Update(original.leftDiagonal[idx], current.leftDiagonal[idx], target.leftDiagonal[idx]);
			Update(original.rightDiagonal[idx], current.rightDiagonal[idx], target.rightDiagonal[idx]);
		}

		cudaError UpdateIndecentEdges(EdgesWeightsDevice* edgesIndecentOriginal, 
			EdgesWeightsDevice* currentEdgesWeights, EdgesWeightsDevice* target, int graphW, int graphH) {
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetUpdateIndecentEdgesBlocksX(graphW),
						   GetUpdateIndecentEdgesBlocksY(graphH));

			UpdateIndecentEdgesKernel<<<blocksDim, threadsDim>>>(*edgesIndecentOriginal, 
				*currentEdgesWeights, *target, graphW, graphH);

			return cudaGetLastError();
		}

	}
}