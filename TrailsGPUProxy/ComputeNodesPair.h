#pragma once
#include "IResource.h"
#include "ComputeNode.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct ComputeNodesPair : public IResource {
			friend class ResourceManager;

			ComputeNode* readOnly = nullptr;
			ComputeNode* writeOnly = nullptr;

			void Swap();

		protected:
			ComputeNodesPair(int graphW, int graphH);
			void Free() override;
		};

	}
}