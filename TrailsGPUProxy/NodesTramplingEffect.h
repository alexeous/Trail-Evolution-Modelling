#pragma once
#include <atomic>
#include <gcroot.h>
#include "cuda_runtime.h"
#include "NodesDataHaloed.h"
#include "ResourceManager.h"
#include "ObjectPool.h"
#include "CudaScheduler.h"
#include "Constants.h"
#include "EdgesData.h"
#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using DistancePairDevice = NodesDataDevicePair<float>;
		using NodesFloatDevice = NodesDataHaloedDevice<float>;

		using namespace System;
		using namespace System::Threading;

		struct NodesTramplingEffect : public IResource {
			friend class ResourceManager;

			void SetAwaitedPathsNumber(int numAwaitedPaths);
			void DecrementAwaitedPathNumber();
			void ClearSync();
			void ApplyTramplingAsync(PoolEntry<DistancePairDevice*> distancePairEntry,
				float pathThickness, float peoplePerSecond, PoolEntry<cudaStream_t> streamEntry,
				CudaScheduler* scheduler);
			void SaveAsEdgesSync(EdgesDataDevice<float>* target, TramplabilityMask* tramplabilityMask);

			void AwaitAllPaths();
			void CancelWaiting();

		protected:
			NodesTramplingEffect(int graphW, int graphH, float graphStep,
				float performanceFactor, ResourceManager* resources);
			void Free(ResourceManager& resources) override;

		private:
			void InitWaitObject();
			float CalcTramplingFactor(float peoplePerSecond);

		private:
			int graphW;
			int graphH;
			float performanceFactor;
			float graphStep;
			NodesFloatDevice* effectDataDevice = nullptr;

			std::atomic<int> numAwaitedPaths = 0;
			gcroot<Object^> waitObj;
		};

	}
}