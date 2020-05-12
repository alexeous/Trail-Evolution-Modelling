#include "WavefrontJob.h"
#include <algorithm>
#include "CudaUtils.h"
#include "ResetNodesG.h"
#include "CudaScheduler.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		int WavefrontJob::asdf = 10;

		WavefrontJob::WavefrontJob(Attractor goal, const std::vector<Attractor>& starts,
			ComputeNodesHost* nodesTemplate, ResourceManager* resources, CudaScheduler* cudaScheduler
		)
		  : goal(goal),
			minIterations(GetMinIterations(goal, starts)),
			hostNodes(CreateHostNodes(nodesTemplate, *resources)),
			deviceNodes(CreateDeviceNodes(nodesTemplate, *resources)),
			resources(resources),
			cudaScheduler(cudaScheduler)
		{
			CHECK_CUDA(cudaStreamCreate(&stream));
		}

		void WavefrontJob::ResetReadOnlyNodesGParallel() {
			int extW = hostNodes->extendedW;
			int extH = hostNodes->extendedH;
			int goalIdx = (goal.nodeI + 1) + (goal.nodeJ + 1) * extW;
			CHECK_CUDA(ResetNodesG(deviceNodes->readOnly, extW, extH, goalIdx, stream));

			cudaScheduler->Schedule(stream, []() {
				
			});
		}

		int WavefrontJob::GetMinIterations(Attractor goal, const std::vector<Attractor>& starts) {
			int minDistance = INT_MAX;
			for(Attractor start : starts) {
				int di = goal.nodeI - start.nodeI;
				int dj = goal.nodeJ - start.nodeJ;
				int dist = std::max(std::abs(di), std::abs(dj));
				minDistance = std::min(minDistance, dist);
			}
			return minDistance;
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

		void WavefrontJob::Free() {
			cudaStreamDestroy(stream);
			resources->Free(deviceNodes);
		}

	}
}