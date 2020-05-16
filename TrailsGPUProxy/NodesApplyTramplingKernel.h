#pragma once
#include "cuda_runtime.h"
#include "NodesDataHaloed.h"
#include "MathUtils.h"

#define NODES_APPLY_TRAMPLING_BLOCK_SIZE_X 16
#define NODES_APPLY_TRAMPLING_BLOCK_SIZE_Y 16


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatDevice = NodesDataHaloedDevice<float>;

		cudaError_t NodesApplyTramplingEffect(NodesFloatDevice* target, NodesFloatDevice* distanceToPath,
			int graphW, int graphH, float pathThickness, float tramplingCoefficient, cudaStream_t stream);

		inline int GetNodesApplyTramplingEffectBlocksX(int graphW) {
			return divceil(graphW, NODES_APPLY_TRAMPLING_BLOCK_SIZE_X);
		}
		inline int GetNodesApplyTramplingEffectBlocksY(int graphH) {
			return divceil(graphH, NODES_APPLY_TRAMPLING_BLOCK_SIZE_Y);
		}
	}
}
