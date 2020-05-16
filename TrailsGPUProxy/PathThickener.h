#pragma once
#include "IResource.h"
#include "ResourceManager.h"
#include "PathThickenerJob.h"
#include "ObjectPool.h"
#include "TramplabilityMask.h"

#define PATH_THICKENER_JOBS_POOL_SIZE 20

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatHost = NodesDataHaloedHost<float>;

		class PathThickener : public IResource {
			friend class ResourceManager;

		public:
			float thickness;
			void StartThickening(PoolEntry<NodesFloatHost*> distanceToPath,
				CudaScheduler* scheduler);

		protected:
			PathThickener(int graphW, int graphH, float graphStep, float thickness, 
				TramplabilityMask* tramplabilityMask, ResourceManager* resources);
			void Free(ResourceManager& resources) override;

		private:
			ObjectPool<PathThickenerJob*>* CreateJobsPool(ResourceManager* resources);

		private:
			int graphW;
			int graphH;
			float graphStep;
			TramplabilityMask* tramplabilityMask;
			ObjectPool<PathThickenerJob*>* jobsPool;
		};

	}
}