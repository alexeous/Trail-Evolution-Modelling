#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ComputeNodesPair.h"
#include "EdgesWeights.h"
#include "ExitFlag.h"
#include "MathUtils.h"

#define WAVEFRONT_PATH_FINDING_BLOCK_SIZE_X 16
#define WAVEFRONT_PATH_FINDING_BLOCK_SIZE_Y 16

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		template<bool SwapNodesPair, bool SetExitFlag>
		cudaError_t WavefrontPathFinding(ComputeNodesPair* nodes, int graphW, int graphH, EdgesWeightsDevice* edgesWeights,
			int goalIndex, float* maxAgentsGPerGroup, ExitFlag* exitFlag, cudaStream_t stream);

		inline int GetWavefrontPathFindingBlocksX(int graphW) {
			return divceil(graphW, WAVEFRONT_PATH_FINDING_BLOCK_SIZE_X);
		}
		inline int GetWavefrontPathFindingBlocksY(int graphH) {
			return divceil(graphH, WAVEFRONT_PATH_FINDING_BLOCK_SIZE_Y);
		}

		template cudaError_t WavefrontPathFinding<false, false>(ComputeNodesPair*, int, int, EdgesWeightsDevice*, int, float*, ExitFlag*, cudaStream_t);
		template cudaError_t WavefrontPathFinding<false, true>(ComputeNodesPair*, int, int, EdgesWeightsDevice*, int, float*, ExitFlag*, cudaStream_t);
		template cudaError_t WavefrontPathFinding<true, false>(ComputeNodesPair*, int, int, EdgesWeightsDevice*, int, float*, ExitFlag*, cudaStream_t);
		template cudaError_t WavefrontPathFinding<true, true>(ComputeNodesPair*, int, int, EdgesWeightsDevice*, int, float*, ExitFlag*, cudaStream_t);
		
	}
}