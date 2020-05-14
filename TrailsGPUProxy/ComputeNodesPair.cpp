#include <utility>
#include "cuda_runtime.h"
#include "ComputeNodesPair.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ComputeNodesPair::ComputeNodesPair(int graphW, int graphH, ResourceManager* resources)
			: readOnly(resources->New<NodesDataHaloedDevice<ComputeNode>>(graphW, graphH)),
			  writeOnly(resources->New<NodesDataHaloedDevice<ComputeNode>>(graphW, graphH))
		{	
		}

		void ComputeNodesPair::CopyReadToWrite(int graphW, int graphH, cudaStream_t stream) {
			size_t size = NodesDataHaloed<ComputeNode>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemcpyAsync(writeOnly->data, readOnly->data, size, cudaMemcpyDeviceToDevice, stream));
		}

		void ComputeNodesPair::CopyWriteToRead(int graphW, int graphH, cudaStream_t stream) {
			size_t size = NodesDataHaloed<ComputeNode>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemcpyAsync(readOnly->data, writeOnly->data, size, cudaMemcpyDeviceToDevice, stream));
		}

		void ComputeNodesPair::Free(ResourceManager& resources) {
			resources.Free(readOnly);
			resources.Free(readOnly);
		}

	}
}