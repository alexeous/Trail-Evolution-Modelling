#pragma once
#include "IResource.h"
#include "ResourceManager.h"
#include "ComputeNode.h"
#include "NodesDataHaloed.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct ComputeNodesPair : public IResource {
			friend class ResourceManager;

			NodesDataHaloedDevice<ComputeNode>* readOnly = nullptr;
			NodesDataHaloedDevice<ComputeNode>* writeOnly = nullptr;

			void CopyReadToWrite(int graphW, int graphH, cudaStream_t stream = 0);
			void CopyWriteToRead(int graphW, int graphH, cudaStream_t stream = 0);

		protected:
			ComputeNodesPair(int graphW, int graphH, ResourceManager* resources);
			void Free(ResourceManager& resources) override;
		};

	}
}