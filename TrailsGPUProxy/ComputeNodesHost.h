#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "IResource.h"
#include "ResourceManager.h"
#include "ComputeNode.h"
#include "Attractor.h"
#include "ComputeNodesPair.h"

using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct ComputeNodesHost : public IResource {
			friend class ResourceManager;

			const int graphW;
			const int graphH;
			const int extendedW;
			const int extendedH;
			ComputeNode* nodes = nullptr;
			
			void InitForStartAttractors(const std::vector<Attractor>& attractors);
			void CopyToDevicePair(ComputeNodesPair* pair);
			void CopyFromDeviceAsync(ComputeNode* device, cudaStream_t stream);

		protected:
			ComputeNodesHost(int graphW, int graphH);
			void Free() override;

		private:
			ComputeNode& At(int graphI, int graphJ);

		private:
			size_t arraySize;
		};

	}
}