#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "EdgesWeights.h"
#include "MathUtils.h"

#define UPDATE_INDECENT_EDGES_BLOCK_SIZE_X 16
#define UPDATE_INDECENT_EDGES_BLOCK_SIZE_Y 16

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		cudaError UpdateIndecentEdges(EdgesWeightsDevice* edgesIndecentOriginal,
			EdgesWeightsDevice* currentEdgesWeights, EdgesWeightsDevice* target, int graphW, int graphH);

		inline int GetUpdateIndecentEdgesBlocksX(int graphW) {
			return divceil(graphW + 1, UPDATE_INDECENT_EDGES_BLOCK_SIZE_X);
		}
		inline int GetUpdateIndecentEdgesBlocksY(int graphH) {
			return divceil(graphH + 1, UPDATE_INDECENT_EDGES_BLOCK_SIZE_Y);
		}
	}
}
