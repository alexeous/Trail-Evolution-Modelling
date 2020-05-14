#include "WavefrontJob.h"
#include <algorithm>
#include "CudaUtils.h"
#include "ResetNodesG.h"
#include "CudaScheduler.h"
#include "WavefrontPathFinding.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		WavefrontJob::WavefrontJob(int graphW, int graphH, Attractor goal,
			const std::vector<Attractor>& starts, ResourceManager* resources)
		  : goal(goal),
			minIterations(GetMinIterations(goal, starts)),
			hostNodes(CreateHostNodes(graphW, graphH, starts, resources)),
			deviceNodes(CreateDeviceNodes(hostNodes, resources)),
			exitFlag(resources->New<ExitFlag>())
		{
			CHECK_CUDA(cudaStreamCreate(&stream));
			size_t maxAgentsGArraySize = sizeof(float) * GetWavefrontPathFindingBlocksX(graphW) * GetWavefrontPathFindingBlocksY(graphH);
			CHECK_CUDA(cudaMalloc(&maxAgentsGPerGroup, maxAgentsGArraySize));
		}

		void WavefrontJob::Start(WavefrontCompletenessTable* wavefrontTable, 
			EdgesWeights* edges, CudaScheduler* scheduler) 
		{
			constexpr int ExitFlagCheckPeriod = 10;

			ResetReadOnlyNodesGParallelAsync();

			int graphW = hostNodes->graphW;
			int graphH = hostNodes->graphH;
			
			delete withoutExitFlagCheck;
			delete withExitFlagCheck;
			withoutExitFlagCheck = new std::function<void(int)>([=](int doubleIterations) {
				for(int i = 0; i < doubleIterations; i++) {
					CHECK_CUDA((WavefrontPathFinding<false, false>(deviceNodes, graphW, graphH, edges, GetGoalIndex(), maxAgentsGPerGroup, nullptr, stream)));
					if(i < doubleIterations - 1) {
						CHECK_CUDA((WavefrontPathFinding<true, false>(deviceNodes, graphW, graphH, edges, GetGoalIndex(), maxAgentsGPerGroup, nullptr, stream)));
					}
					else {
						exitFlag->ResetAsync(stream);
						CHECK_CUDA((WavefrontPathFinding<true, true>(deviceNodes, graphW, graphH, edges, GetGoalIndex(), maxAgentsGPerGroup, exitFlag, stream)));
					}
				}
			});
			(*withoutExitFlagCheck)(minIterations / 2);

			exitFlag->ReadFromDeviceAsync(stream);
			withExitFlagCheck = new std::function<void()>;
			*withExitFlagCheck = [=]() {
				if(exitFlag->GetLastHostValue()) {
					deviceNodes->readOnly->CopyTo(hostNodes, graphW, graphH, stream);
					scheduler->Schedule(stream, [=]() {
						WavefrontCompletenessTable* t = wavefrontTable;
						wavefrontTable->SetCompleted(goal, hostNodes);
					});
				}
				else {
					(*withoutExitFlagCheck)(ExitFlagCheckPeriod / 2);
					exitFlag->ReadFromDeviceAsync(stream);
					scheduler->Schedule(stream, *withExitFlagCheck);
				}
			};

			scheduler->Schedule(stream, *withExitFlagCheck);
		}

		void WavefrontJob::ResetReadOnlyNodesGParallelAsync() {
			int extW = hostNodes->extendedW;
			int extH = hostNodes->extendedH;
			CHECK_CUDA(ResetNodesG(*deviceNodes->readOnly, extW, extH, GetGoalIndex(), stream));
		}

		int WavefrontJob::GetGoalIndex() {
			return (goal.nodeI + 1) + (goal.nodeJ + 1) * hostNodes->extendedW;
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
			ResourceManager* resources) 
		{
			auto deviceNodesPair = resources->New<ComputeNodesPair>(hostNodes->graphW, hostNodes->graphH, resources);
			hostNodes->CopyToDevicePair(deviceNodesPair);
			return deviceNodesPair;
		}

		ComputeNodesHost* WavefrontJob::CreateHostNodes(int w, int h,
			const std::vector<Attractor>& starts, ResourceManager* resources)
		{
			auto hostNodes = resources->New<ComputeNodesHost>(w, h);
			hostNodes->InitForStartAttractors(starts);
			return hostNodes;
		}

		void WavefrontJob::Free(ResourceManager& resources) {
			cudaStreamDestroy(stream);
			cudaFree(maxAgentsGPerGroup);
			resources.Free(hostNodes);
			resources.Free(deviceNodes);
			delete withoutExitFlagCheck;
			delete withExitFlagCheck;
		}

	}
}