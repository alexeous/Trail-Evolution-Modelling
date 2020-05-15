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

		void ComputeNodesHost::CopyToDevicePairSync(ComputeNodesPair* pair) {
			CopyToSync(pair->readOnly);
			pair->CopyReadToWriteSync(graphW, graphH);
		}

		void ComputeNodesHost::CopyToDevicePair(ComputeNodesPair* pair, cudaStream_t stream) {
			CopyTo(pair->readOnly, stream);
			pair->CopyReadToWrite(graphW, graphH, stream);
		}

		void ComputeNodesHost::Free(ResourceManager&) {
			cudaFreeHost(data);
		}

	}
}