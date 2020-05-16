#pragma once
#include "cuda_runtime.h"
// TODO: remove
#include <atomic>
#include "IResource.h"
#include "ResourceManager.h"
#include "NodesDataHaloed.h"
#include "NodesDataDevicePair.h"
#include "CudaScheduler.h"
#include "ObjectPool.h"
#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatHost = NodesDataHaloedHost<float>;

		class PathThickenerJob : public IResource {
			friend class ResourceManager;

		public:
			// TODO: remove
			static std::atomic<int> numRemaining;

			void StartThickening(PoolEntry<NodesFloatHost*> distanceToPath, 
				float thickness, float graphStep, TramplabilityMask* tramplabilityMask, 
				PoolEntry<PathThickenerJob*> selfInPool, CudaScheduler* scheduler);

		protected:
			PathThickenerJob(int graphW, int graphH, ResourceManager* resources);
			void Free(ResourceManager& resources) override;

		private:
			void ThickenPathAsync(float thickness, float graphStep, TramplabilityMask* tramplabilityMask);

		private:
			int graphW;
			int graphH;
			cudaStream_t stream;
			NodesDataDevicePair<float>* distanceDevicePair;
		};

	}
}