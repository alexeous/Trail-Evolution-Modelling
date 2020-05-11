#include "WavefrontJob.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		WavefrontJob::WavefrontJob(Attractor goal, ComputeNodesHost* nodesTemplate, ResourceManager* resources)
		: goal(goal), 
		  nodes(CreateNodes(nodesTemplate, *resources)),
		  resources(resources)
		{
		}

		void WavefrontJob::Free() {
			resources->Free(nodes);
		}

		ComputeNodesPair* WavefrontJob::CreateNodes(ComputeNodesHost* nodesTemplate, ResourceManager& resources) {
			auto deviceNodesPair = resources.New<ComputeNodesPair>(nodesTemplate->graphW, nodesTemplate->graphH);
			nodesTemplate->CopyToDevicePair(deviceNodesPair);
			return deviceNodesPair;
		}

	}
}