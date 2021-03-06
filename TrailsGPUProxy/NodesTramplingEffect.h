#pragma once
#include "cuda_runtime.h"
#include "ResourceManager.h"
#include <atomic>
#include <gcroot.h>
#include "NodesDataHaloed.h"
#include "ObjectPool.h"
#include "CudaScheduler.h"
#include "Constants.h"
#include "EdgesData.h"
#include "TramplabilityMask.h"
#include "EdgesTramplingEffect.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using DistancePairDevice = NodesDataDevicePair<float>;
		using NodesFloatDevice = NodesDataHaloedDevice<float>;

		using namespace System;
		using namespace System::Threading;

		struct NodesTramplingEffect : public IResource {
			friend class ResourceManager;

			float performanceFactor;
			float simulationStepSeconds;
			NodesFloatDevice* GetDataDevice();

			void SetAwaitedPathsNumber(int numAwaitedPaths);
			void DecrementAwaitedPathNumber();
			void ClearSync();
			void ApplyTramplingAsync(PoolEntry<DistancePairDevice*> distancePairEntry,
				float pathThickness, float peoplePerSecond, PoolEntry<cudaStream_t> streamEntry,
				CudaScheduler* scheduler);
			void SaveAsEdgesSync(EdgesTramplingEffect* target, TramplabilityMask* tramplabilityMask);

			void AwaitAllPaths();
			void CancelWaiting();

		protected:
			NodesTramplingEffect(int graphW, int graphH, float graphStep,
				float performanceFactor, float simulationStepSeconds, ResourceManager* resources);
			void Free(ResourceManager& resources) override;

		private:
			void InitWaitObject();
			float CalcTramplingFactor(float peoplePerSecond);

		private:
			int graphW;
			int graphH;
			float graphStep;
			NodesFloatDevice* effectDataDevice = nullptr;

			std::atomic<int> numAwaitedPaths = 0;
			gcroot<Object^> waitObj;
		};

	}
}