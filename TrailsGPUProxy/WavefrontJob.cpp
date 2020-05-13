#include "WavefrontJob.h"
#include <algorithm>
#include "CudaUtils.h"
#include "ResetNodesG.h"
#include "CudaScheduler.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		WavefrontJob::WavefrontJob(int graphW, int graphH, Attractor goal, 
			const std::vector<Attractor>& starts, ResourceManager* resources)
		  : goal(goal),
			minIterations(GetMinIterations(goal, starts)),
			hostNodes(CreateHostNodes(graphW, graphH, starts, *resources)),
			deviceNodes(CreateDeviceNodes(hostNodes, *resources)),
			resources(resources)
		{
			CHECK_CUDA(cudaStreamCreate(&stream));
		}

		void WavefrontJob::ResetReadOnlyNodesGParallel() {
			int extW = hostNodes->extendedW;
			int extH = hostNodes->extendedH;
			int goalIdx = (goal.nodeI + 1) + (goal.nodeJ + 1) * extW;
			CHECK_CUDA(ResetNodesG(deviceNodes->readOnly, extW, extH, goalIdx, stream));
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

		ComputeNodesPair* WavefrontJob::CreateDeviceNodes(ComputeNodesHost* hostNodes, 
			ResourceManager& resources) 
		{
			auto deviceNodesPair = resources.New<ComputeNodesPair>(hostNodes->graphW, hostNodes->graphH);
			hostNodes->CopyToDevicePair(deviceNodesPair);
			return deviceNodesPair;
		}

		ComputeNodesHost* WavefrontJob::CreateHostNodes(int w, int h,
			const std::vector<Attractor>& starts, ResourceManager& resources)
		{
			auto hostNodes = resources.New<ComputeNodesHost>(w, h);
			hostNodes->InitForStartAttractors(starts);
			return hostNodes;
		}

		void WavefrontJob::Free() {
			cudaStreamDestroy(stream);
			resources->Free(hostNodes);
			resources->Free(deviceNodes);
		}

	}
}