#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "NodesDataDevicePair.h"
#include "TramplabilityMask.h"
#include "MathUtils.h"

#define PATH_THICKENING_BLOCK_SIZE_X 16
#define PATH_THICKENING_BLOCK_SIZE_Y 16

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		template<bool SwapNodesPair>
		cudaError_t PathThickening(NodesDataDevicePair<float>* distanceToPath, int graphW, int graphH, 
			TramplabilityMask* tramplabilityMask, cudaStream_t stream);

		template cudaError_t PathThickening<false>(NodesDataDevicePair<float>* distanceToPath, int graphW, int graphH,
			TramplabilityMask* tramplabilityMask, cudaStream_t stream);
		template cudaError_t PathThickening<true>(NodesDataDevicePair<float>* distanceToPath, int graphW, int graphH,
			TramplabilityMask* tramplabilityMask, cudaStream_t stream);

		inline int GetPathThickeningBlocksX(int graphW) {
			return divceil(graphW, PATH_THICKENING_BLOCK_SIZE_X);
		}
		inline int GetPathThickeningBlocksY(int graphH) {
			return divceil(graphH, PATH_THICKENING_BLOCK_SIZE_Y);
		}

	}
}