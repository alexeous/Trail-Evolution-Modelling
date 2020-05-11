#include "ComputeNodesHost.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ComputeNodesHost::ComputeNodesHost(int graphW, int graphH)
			: graphW(graphW),
			  graphH(graphH),
			  extendedW(graphW + 2), 
			  extendedH(graphH + 2)
		{
			size_t size = extendedW * extendedH * sizeof(ComputeNode);
			CHECK_CUDA(cudaMallocHost(&nodes, size));
			CHECK_CUDA(cudaMemset(nodes, 0, size))
		}

		void ComputeNodesHost::InitForStartAttractors(const std::vector<Attractor>& attractors) {
			for(Attractor attractor : attractors) {
				At(attractor.nodeI, attractor.nodeJ).SetStart(true);
			}
		}

		void ComputeNodesHost::DeinitForStartAttractors(const std::vector<Attractor>& attractors) {
			for(Attractor attractor : attractors) {
				At(attractor.nodeI, attractor.nodeJ).SetStart(false);
			}
		}

		void ComputeNodesHost::CopyToDevicePair(ComputeNodesPair* pair) {
			int size = extendedW * extendedH * sizeof(ComputeNode);
			CHECK_CUDA(cudaMemcpy(pair->readOnly, nodes, size, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(pair->writeOnly, pair->readOnly, size, cudaMemcpyDeviceToDevice));
		}

		void ComputeNodesHost::Free() {
			cudaFreeHost(nodes);
		}

		ComputeNode& ComputeNodesHost::At(int graphI, int graphJ) {
			int computeI = graphI + 1;
			int computeJ = graphJ + 1;
			return nodes[computeI + computeJ * extendedW];
		}

	}
}