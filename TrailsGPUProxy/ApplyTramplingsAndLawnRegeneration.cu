#include "ApplyTramplingsAndLawnRegeneration.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ // visual studio doesn't show functions declared in device_atomic_function.h without this
#define UNDEF__CUDACC__
#define __CUDACC__
#endif
#include <device_atomic_functions.h>
#include <device_functions.h>
#include <math_functions.h>
#ifdef UNDEF__CUDACC__
#undef UNDEF__CUDACC__
#undef __CUDACC__
#endif

#include "Constants.h"


#define BLOCK_SIZE_X APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_X
#define BLOCK_SIZE_Y APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_Y

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		inline __device__ float clamp(float value, float minVal, float maxVal) {
			return max(minVal, min(value, maxVal));
		}

		inline __device__ void ApplyToEdge(float& edge, bool tramplability, float indecentTrampling, 
			float node1, float node2, float minWeight, float maxWeight) 
		{
			float decentTrampling = (node1 + node2) * 0.5f;
			float e = edge + tramplability * (LAWN_REGENERATION_PER_SIMULATION_STEP - indecentTrampling - decentTrampling);
			edge = clamp(e, minWeight, maxWeight);
		}

		__global__ void ApplyTramplingsAndLawnRegenerationKernel(EdgesWeightsDevice target, int graphW, int graphH,
			EdgesTramplingEffect indecentTramplingEdges, NodesFloatDevice decentTramplingNodes,
			TramplabilityMask tramplability, EdgesWeightsDevice minWeights, EdgesWeightsDevice maxWeights)
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			if(i > graphW || j > graphH)
				return;
			
			float centerNodeTrampling = decentTramplingNodes.At(i + 1, j + 1, graphW);

			ApplyToEdge(target.E(i, j, graphW), tramplability.E(i, j, graphW), indecentTramplingEdges.E(i, j, graphW), centerNodeTrampling, decentTramplingNodes.At(i + 2, j + 1, graphW), minWeights.E(i, j, graphW), maxWeights.E(i, j, graphW));
			if(j < graphH - 1) {
				ApplyToEdge(target.S(i, j, graphW), tramplability.S(i, j, graphW), indecentTramplingEdges.S(i, j, graphW), centerNodeTrampling, decentTramplingNodes.At(i + 1, j + 2, graphW), minWeights.S(i, j, graphW), maxWeights.S(i, j, graphW));
				if(i < graphW - 1)
					ApplyToEdge(target.SE(i, j, graphW), tramplability.SE(i, j, graphW), indecentTramplingEdges.SE(i, j, graphW), centerNodeTrampling, decentTramplingNodes.At(i + 2, j + 2, graphW), minWeights.SE(i, j, graphW), maxWeights.SE(i, j, graphW));
				if(i != 0)
					ApplyToEdge(target.SW(i, j, graphW), tramplability.SW(i, j, graphW), indecentTramplingEdges.SW(i, j, graphW), centerNodeTrampling, decentTramplingNodes.At(i, j + 2, graphW), minWeights.SW(i, j, graphW), maxWeights.SW(i, j, graphW));
			}
		}

		cudaError ApplyTramplingsAndLawnRegeneration(EdgesWeightsDevice* target, int graphW, int graphH, 
			EdgesTramplingEffect* indecentTramplingEdges, NodesFloatDevice* decentTramplingNodes,
			TramplabilityMask* tramplabilityMask, EdgesWeightsDevice* minWeights, EdgesWeightsDevice* maxWeights)
		{
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetApplyTramplingsAndLawnRegenerationBlocksX(graphW),
				           GetApplyTramplingsAndLawnRegenerationBlocksY(graphH));

			ApplyTramplingsAndLawnRegenerationKernel<<<blocksDim, threadsDim>>>(*target, graphW, graphH,
				*indecentTramplingEdges, *decentTramplingNodes, 
				*tramplabilityMask, *minWeights, *maxWeights);

			return cudaGetLastError();
		}

	}
}