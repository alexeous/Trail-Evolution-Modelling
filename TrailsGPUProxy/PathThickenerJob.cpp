#include "PathThickenerJob.h"
#include "CudaUtils.h"
#include "PathThickeningKernel.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		std::atomic<int> PathThickenerJob::numRemaining = 0;

		PathThickenerJob::PathThickenerJob(int graphW, int graphH, ResourceManager* resources)
			: graphW(graphW),
			  graphH(graphH),
			  distanceDevicePair(resources->New<NodesDataDevicePair<float>>(graphW, graphH, resources))
		{
			CHECK_CUDA(cudaStreamCreate(&stream));
		}

		void PathThickenerJob::StartThickening(PoolEntry<NodesFloatHost*> distanceToPath,
			float thickness, float graphStep, TramplabilityMask* tramplabilityMask, 
			PoolEntry<PathThickenerJob*> selfInPool, CudaScheduler* scheduler)
		{
			distanceToPath.object->CopyToDevicePair(distanceDevicePair, stream);
			scheduler->Schedule(stream, [=] {
				distanceToPath.ReturnToPool();
				ThickenPathAsync(thickness, graphStep, tramplabilityMask);
				scheduler->Schedule(stream, [=] {
					// TODO: pass result for further work
					selfInPool.ReturnToPool();
					numRemaining--;
				});
			});
		}

		void PathThickenerJob::ThickenPathAsync(float thickness, float graphStep, TramplabilityMask* tramplabilityMask) {
			int doubleIterations = (int)ceilf(thickness / graphStep / 2);
			for(int i = 0; i < doubleIterations; i++) {
				CHECK_CUDA((PathThickening<false>(distanceDevicePair, graphW, graphH, tramplabilityMask, stream)));
				CHECK_CUDA((PathThickening<true>(distanceDevicePair, graphW, graphH, tramplabilityMask, stream)));
			}
		}

		void PathThickenerJob::Free(ResourceManager& resources) {
			cudaStreamDestroy(stream);
			resources.Free(distanceDevicePair);
		}

	}
}