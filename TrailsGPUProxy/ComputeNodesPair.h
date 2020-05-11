#pragma once
#include "IResource.h"
#include "ComputeNode.h"


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct ComputeNodesPair : public IResource {
			ComputeNode* readOnly;
			ComputeNode* writeOnly;

		protected:
			ComputeNodesPair(int graphW, int graphH);
			void Free() override;
		};

	}
}