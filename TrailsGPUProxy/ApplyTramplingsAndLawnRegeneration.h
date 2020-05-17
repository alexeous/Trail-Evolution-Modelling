#pragma once
#include "cuda_runtime.h"
#include "EdgesWeights.h"
#include "NodesDataHaloed.h"
#include "EdgesTramplingEffect.h"
#include "TramplabilityMask.h"
#include "MathUtils.h"

#define APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_X 16
#define APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_Y 16

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatDevice = NodesDataHaloedDevice<float>;

		cudaError ApplyTramplingsAndLawnRegeneration(EdgesWeightsDevice* target, int graphW, int graphH,
			EdgesTramplingEffect* indecentTramplingEdges, NodesFloatDevice* decentTramplingNodes,
			TramplabilityMask* tramplabilityMask, EdgesWeightsDevice* minWeights, EdgesWeightsDevice* maxWeights);

		inline int GetApplyTramplingsAndLawnRegenerationBlocksX(int graphW) {
			return divceil(graphW, APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_X);
		}
		inline int GetApplyTramplingsAndLawnRegenerationBlocksY(int graphH) {
			return divceil(graphH, APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_Y);
		}
	}
}