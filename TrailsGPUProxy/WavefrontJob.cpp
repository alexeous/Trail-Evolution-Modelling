#include "WavefrontJob.h"
#include "CudaUtils.h"
#include "ResetNodesG.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		WavefrontJob::WavefrontJob(Attractor goal, ComputeNodesHost* nodesTemplate, ResourceManager* resources)
		: goal(goal), 
		  resources(resources),
		  hostNodes(CreateHostNodes(nodesTemplate, *resources)),
		  deviceNodes(CreateDeviceNodes(nodesTemplate, *resources))
		{
		}

		void WavefrontJob::ResetReadOnlyNodesGParallel() {
			int extW = hostNodes->extendedW;
			int extH = hostNodes->extendedH;
			int goalIdx = (goal.nodeI + 1) + (goal.nodeJ + 1) * extW;
			CHECK_CUDA(ResetNodesG(deviceNodes->readOnly, extW, extH, goalIdx));
		}

		void WavefrontJob::Free() {
			resources->Free(deviceNodes);
		}

		ComputeNodesPair* WavefrontJob::CreateDeviceNodes(
			ComputeNodesHost* nodesTemplate, ResourceManager& resources) 
		{
			auto deviceNodesPair = resources.New<ComputeNodesPair>(nodesTemplate->graphW, nodesTemplate->graphH);
			nodesTemplate->CopyToDevicePair(deviceNodesPair);
			return deviceNodesPair;
		}

		ComputeNodesHost* WavefrontJob::CreateHostNodes(
			ComputeNodesHost* nodesTemplate, ResourceManager& resources) 
		{
			return resources.New<ComputeNodesHost>(nodesTemplate->graphW, nodesTemplate->graphH);
		}

	}
}