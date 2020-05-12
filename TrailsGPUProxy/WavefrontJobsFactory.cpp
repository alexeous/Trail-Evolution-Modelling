#include "WavefrontJobsFactory.h"
#include "ComputeNode.h"
#include "ComputeNodesHost.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		std::vector<WavefrontJob*> WavefrontJobsFactory::CreateJobs(int graphW, int graphH,
			ResourceManager& resources, const AttractorsMap& attractors, CudaScheduler* cudaScheduler)
		{
			std::vector<WavefrontJob*> jobs;
			ComputeNodesHost* hostNodes = resources.New<ComputeNodesHost>(graphW, graphH);
			for(auto pair : attractors) {
				auto goal = pair.first;
				auto& starts = pair.second;

				hostNodes->InitForStartAttractors(starts);
				WavefrontJob* job = resources.New<WavefrontJob>(goal, starts, hostNodes, &resources, cudaScheduler);
				hostNodes->DeinitForStartAttractors(starts);
				jobs.push_back(job);
			}
			resources.Free(hostNodes);

			return jobs;
		}

	}
}