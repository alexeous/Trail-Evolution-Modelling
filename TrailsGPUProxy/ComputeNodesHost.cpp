#include "ComputeNodesHost.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ComputeNodesHost::ComputeNodesHost(int graphW, int graphH)
			: graphW(graphW),
			  graphH(graphH),
			  extendedW(graphW + 2), 
			  extendedH(graphH + 2),
			  arraySize(extendedW* extendedH * sizeof(ComputeNode))
		{
			CHECK_CUDA(cudaMallocHost(&nodes, arraySize));
			CHECK_CUDA(cudaMemset(nodes, 0, arraySize))
		}

		void ComputeNodesHost::InitForStartAttractors(const std::vector<Attractor>& attractors) {
			for(Attractor attractor : attractors) {
				At(attractor.nodeI, attractor.nodeJ).SetStart(true);
			}
		}

		void ComputeNodesHost::CopyToDevicePair(ComputeNodesPair* pair) {
			CHECK_CUDA(cudaMemcpy(pair->readOnly, nodes, arraySize, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(pair->writeOnly, pair->readOnly, arraySize, cudaMemcpyDeviceToDevice));
		}

		void ComputeNodesHost::CopyFromDeviceAsync(ComputeNode* device, cudaStream_t stream) {
			CHECK_CUDA(cudaMemcpyAsync(nodes, device, arraySize, cudaMemcpyDeviceToHost, stream));
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