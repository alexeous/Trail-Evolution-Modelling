#include "PathThickener.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		PathThickener::PathThickener(int graphW, int graphH, float graphStep, 
			float thickness, TramplabilityMask* tramplabilityMask, ResourceManager* resources)
			: graphW(graphW),
			  graphH(graphH),
			  graphStep(graphStep),
			  thickness(thickness),
			  tramplabilityMask(tramplabilityMask),
			  jobsPool(CreateJobsPool(resources))
		{
		}

		void PathThickener::StartThickening(PoolEntry<NodesFloatHost> distanceToPath,
			CudaScheduler* scheduler) 
		{
			PoolEntry<PathThickenerJob> job = jobsPool->Take();
			job.object->StartThickening(distanceToPath, thickness, graphStep, tramplabilityMask, job, scheduler);
		}

		ObjectPool<PathThickenerJob>* PathThickener::CreateJobsPool(ResourceManager* resources) {
			return resources->New<ObjectPool<PathThickenerJob>>(
				PATH_THICKENER_JOBS_POOL_SIZE,
				[=] { return resources->New<PathThickenerJob>(graphW, graphH, resources); }
			);
		}

		void PathThickener::Free(ResourceManager& resources) {
			resources.Free(jobsPool);
		}

	}
}