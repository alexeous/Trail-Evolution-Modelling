#include <utility>
#include "cuda_runtime.h"
#include "ComputeNodesPair.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ComputeNodesPair::ComputeNodesPair(int graphW, int graphH) {
			size_t size = (graphW + 2) * (graphH + 2) * sizeof(ComputeNode);
			CHECK_CUDA(cudaMalloc(&readOnly, size));
			CHECK_CUDA(cudaMalloc(&writeOnly, size));
		}

		void ComputeNodesPair::Swap() {
			std::swap(readOnly, writeOnly);
		}

		void ComputeNodesPair::Free() {
			cudaFree(readOnly);
			cudaFree(writeOnly);
		}

	}
}