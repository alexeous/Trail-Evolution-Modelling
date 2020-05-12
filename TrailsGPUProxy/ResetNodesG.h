#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ComputeNode.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		cudaError_t ResetNodesG(ComputeNode* nodes, int extendedW, int extendedH, int goalIndex, cudaStream_t stream);

	}
}