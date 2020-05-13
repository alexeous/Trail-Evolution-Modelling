#include "WavefrontJobsFactory.h"
#include "ComputeNode.h"
#include "ComputeNodesHost.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		std::vector<WavefrontJob*> WavefrontJobsFactory::CreateJobs(int graphW, int graphH,
			ResourceManager* resources, const AttractorsMap& attractors)
		{
			std::vector<WavefrontJob*> jobs;
			for(auto goal : attractors.uniqueAttractors) {
				auto starts = attractors.at(goal);

				WavefrontJob* job = resources->New<WavefrontJob>(graphW, graphH, goal, starts, resources);
				jobs.push_back(job);
			}

			return jobs;
		}

	}
}