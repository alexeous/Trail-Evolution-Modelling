#pragma once
#include "cuda_runtime.h"
#include "NodesDataHaloed.h"
#include "MathUtils.h"
#include "EdgesData.h"
#include "TramplabilityMask.h"

#define SAVE_NODES_TRAMPLING_AS_EDGES_BLOCK_SIZE_X 16
#define SAVE_NODES_TRAMPLING_AS_EDGES_BLOCK_SIZE_Y 16

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using EdgesTramplingEffect = EdgesDataDevice<float>;
		using NodesFloatDevice = NodesDataHaloedDevice<float>;

		cudaError SaveNodesTramplingAsEdges(NodesFloatDevice* nodesTrampling, int graphW, int graphH,
			EdgesTramplingEffect* targetEdges, TramplabilityMask* tramplabilityMask);

		inline int GetSaveNodesTramplingAsEdgesBlocksX(int graphW) {
			return divceil(graphW + 1, SAVE_NODES_TRAMPLING_AS_EDGES_BLOCK_SIZE_X);
		}
		inline int GetSaveNodesTramplingAsEdgesBlocksY(int graphH) {
			return divceil(graphH + 1, SAVE_NODES_TRAMPLING_AS_EDGES_BLOCK_SIZE_Y);
		}
	}
}
