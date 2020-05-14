#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "IResource.h"
#include "ResourceManager.h"
#include "ComputeNode.h"
#include "Attractor.h"
#include "NodesDataHaloed.h"
#include "ComputeNodesPair.h"

using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct ComputeNodesHost : public NodesDataHaloedHost<ComputeNode> {
			friend class ResourceManager;

			const int graphW;
			const int graphH;
			const int extendedW;
			const int extendedH;
			
			void InitForStartAttractors(const std::vector<Attractor>& attractors);
			void CopyToDevicePair(ComputeNodesPair* pair, cudaStream_t stream = 0);

		protected:
			ComputeNodesHost(int graphW, int graphH);
			void Free() override;
		};

	}
}