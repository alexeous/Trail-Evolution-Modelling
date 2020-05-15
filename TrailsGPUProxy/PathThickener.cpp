#include "PathThickener.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		PathThickener::PathThickener(int graphW, int graphH, float thickness, ResourceManager* resources)
			: graphW(graphW),
			  graphH(graphH),
			  thickness(thickness),
			  jobsPool(CreateJobsPool(resources))
		{
		}

		void PathThickener::StartThickening(PoolEntry<NodesFloatHost> distanceToPath,
			CudaScheduler* scheduler) 
		{
			PoolEntry<PathThickenerJob> job = jobsPool->Take();
			job.object->StartThickening(distanceToPath, thickness, job, scheduler);
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