#include "NodesApplyTramplingKernel.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ // visual studio doesn't show functions declared in device_atomic_function.h without this
#define UNDEF__CUDACC__
#define __CUDACC__
#endif
#include <device_atomic_functions.h>
#include <math_functions.h>
#include <device_functions.h>
#ifdef UNDEF__CUDACC__
#undef UNDEF__CUDACC__
#undef __CUDACC__
#endif

#define BLOCK_SIZE_X NODES_APPLY_TRAMPLING_BLOCK_SIZE_X
#define BLOCK_SIZE_Y NODES_APPLY_TRAMPLING_BLOCK_SIZE_Y

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		__global__ void NodesApplyTramplingEffectKernel(float* target, float* distanceToPath,
			int graphW, int graphH, float pathThickness, float tramplingCoefficient) 
		{
			int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
			int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
			if(i <= graphW && j <= graphH) {
				int index = i + j * (graphW + 2);
				
				float t = distanceToPath[index];
				t = max(0.0f, min(1.0f, fabsf(t / pathThickness)));
				t = t * (t * (-4 * t + 6) - 3) + 1;		// cubic parabola
				
				atomicAdd(&target[index], t * tramplingCoefficient);
			}
		}

		cudaError_t NodesApplyTramplingEffect(NodesFloatDevice* target, NodesFloatDevice* distanceToPath, 
			int graphW, int graphH, float pathThickness, float tramplingCoefficient, cudaStream_t stream) 
		{
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetNodesApplyTramplingEffectBlocksX(graphW), 
				           GetNodesApplyTramplingEffectBlocksY(graphH));
			
			NodesApplyTramplingEffectKernel<<<blocksDim, threadsDim, 0, stream>>>(target->data, distanceToPath->data, 
				graphW, graphH, pathThickness, tramplingCoefficient);

			return cudaGetLastError();
		}

	}
}
