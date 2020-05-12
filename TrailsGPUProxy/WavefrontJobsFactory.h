#pragma once
#include <vector>
#include "WavefrontJob.h"
#include "AttractorsMap.h"
#include "ResourceManager.h"
#include "CudaScheduler.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class WavefrontJobsFactory {
		public:
			static std::vector<WavefrontJob*> CreateJobs(int graphW, int graphH,
				ResourceManager& resources, const AttractorsMap& attractors, CudaScheduler* cudaScheduler);
		};

	}
}