#pragma once
// TODO: remove
#include <atomic>
#include "cuda_runtime.h"
#include "IResource.h"
#include "ResourceManager.h"
#include "ObjectPool.h"
#include "CudaScheduler.h"
#include "TramplabilityMask.h"

#define PATH_THICKENER_STREAMS_POOL_SIZE 10
#define PATH_THICKENER_DISTANCE_DEVICE_POOL_SIZE 20

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatHost = NodesDataHaloedHost<float>;
		using DistancePairDevice = NodesDataDevicePair<float>;

		class PathThickener : public IResource {
			friend class ResourceManager;

		public:
			// TODO: remove
			static std::atomic<int> numRemaining;

			float thickness;
			void StartThickening(PoolEntry<NodesFloatHost*> distanceToPath,
				CudaScheduler* scheduler);

		protected:
			PathThickener(int graphW, int graphH, float graphStep, float thickness, 
				TramplabilityMask* tramplabilityMask, ResourceManager* resources);
			void Free(ResourceManager& resources) override;

		private:
			//ObjectPool<PathThickenerJob*>* CreateJobsPool(ResourceManager* resources);
			ObjectPool<cudaStream_t>* CreateStreamsPool(ResourceManager* resources);
			ObjectPool<DistancePairDevice*>* CreateDistancePairPool(ResourceManager* resources);
			void ThickenPathAsync(float thickness, float graphStep,
				DistancePairDevice* distancePair, TramplabilityMask* tramplabilityMask, cudaStream_t stream);

		private:
			int graphW;
			int graphH;
			float graphStep;
			TramplabilityMask* tramplabilityMask;
			ObjectPool<cudaStream_t>* streamsPool;
			ObjectPool<DistancePairDevice*>* distancePairPool;

			//ObjectPool<PathThickenerJob*>* jobsPool;
		};

	}
}