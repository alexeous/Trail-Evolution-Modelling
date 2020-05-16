#include "PathThickener.h"
#include "CudaUtils.h"
#include "PathThickeningKernel.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		PathThickener::PathThickener(int graphW, int graphH, float graphStep, 
			float thickness, TramplabilityMask* tramplabilityMask, 
			NodesTramplingEffect* targetTramplingEffect, ResourceManager* resources)
			: graphW(graphW),
			  graphH(graphH),
			  graphStep(graphStep),
			  thickness(thickness),
			  tramplabilityMask(tramplabilityMask),
			  targetTramplingEffect(targetTramplingEffect),
			  streamsPool(CreateStreamsPool(resources)),
			  distancePairPool(CreateDistancePairPool(resources))
		{
		}

		void PathThickener::StartThickening(PoolEntry<NodesFloatHost*> distanceToPath, 
			float peoplePerSecond, CudaScheduler* scheduler) 
		{
			PoolEntry<DistancePairDevice*> distancePairEntry = distancePairPool->Take();
			DistancePairDevice* distancePair = distancePairEntry.object;
			PoolEntry<cudaStream_t> streamEntry = streamsPool->Take();
			cudaStream_t stream = streamEntry.object;

			distanceToPath.object->CopyToDevicePair(distancePair, stream);
			scheduler->Schedule(stream, [=] {
				distanceToPath.ReturnToPool();
				ThickenPathAsync(thickness, graphStep, distancePair, tramplabilityMask, stream);
				scheduler->Schedule(stream, [=] {
					targetTramplingEffect->ApplyTramplingAsync(distancePairEntry, thickness, peoplePerSecond, streamEntry, scheduler);
				});
			});
		}

		ObjectPool<cudaStream_t>* PathThickener::CreateStreamsPool(ResourceManager* resources) {
			return resources->New<ObjectPool<cudaStream_t>>(
				PATH_THICKENER_STREAMS_POOL_SIZE,
				[] { cudaStream_t stream; cudaStreamCreate(&stream); return stream; },
				[] (cudaStream_t stream, ResourceManager&) { cudaStreamDestroy(stream); }
			);
		}

		ObjectPool<DistancePairDevice*>* PathThickener::CreateDistancePairPool(ResourceManager* resources) {
			return resources->New<ObjectPool<DistancePairDevice*>>(
				PATH_THICKENER_DISTANCE_DEVICE_POOL_SIZE,
				[=] { return resources->New<DistancePairDevice>(graphW, graphH, resources); }
			);
		}

		void PathThickener::ThickenPathAsync(float thickness, float graphStep, 
			DistancePairDevice* distancePair, TramplabilityMask* tramplabilityMask, cudaStream_t stream) 
		{
			int doubleIterations = (int)ceilf(thickness / graphStep / 2);
			for(int i = 0; i < doubleIterations; i++) {
				CHECK_CUDA((PathThickening<false>(distancePair, graphW, graphH, tramplabilityMask, stream)));
				CHECK_CUDA((PathThickening<true>(distancePair, graphW, graphH, tramplabilityMask, stream)));
			}
		}

		void PathThickener::Free(ResourceManager& resources) {
			resources.Free(streamsPool);
			resources.Free(distancePairPool);
			//resources.Free(jobsPool);
		}

	}
}