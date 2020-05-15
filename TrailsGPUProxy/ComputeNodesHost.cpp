#include "ComputeNodesHost.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ComputeNodesHost::ComputeNodesHost(int graphW, int graphH)
			: NodesDataHaloedHost<ComputeNode>(graphW, graphH)
		{
			size_t arraySize = NodesDataHaloed<ComputeNode>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemset(data, 0, arraySize));
		}

		void ComputeNodesHost::InitForStartAttractors(const std::vector<Attractor>& attractors) {
			for(Attractor attractor : attractors) {
				At(attractor.nodeI + 1, attractor.nodeJ + 1).SetStart(true);
			}
		}

		void ComputeNodesHost::Free(ResourceManager&) {
			cudaFreeHost(data);
		}

	}
}