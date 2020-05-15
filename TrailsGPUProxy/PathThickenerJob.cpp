#include "PathThickenerJob.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		std::atomic<int> PathThickenerJob::numRemaining = 0;

		PathThickenerJob::PathThickenerJob(int graphW, int graphH, ResourceManager* resources)
			: graphW(graphW),
			  graphH(graphH),
			  distanceDevice(resources->New<NodesFloatDevice>(graphW, graphH))
		{
			CHECK_CUDA(cudaStreamCreate(&stream));
		}

		void PathThickenerJob::StartThickening(PoolEntry<NodesFloatHost> distanceToPath,
			float thickness, PoolEntry<PathThickenerJob> selfInPool, CudaScheduler* scheduler)
		{
			distanceToPath.object->CopyTo(distanceDevice, stream);
			scheduler->Schedule(stream, [=] {
				distanceToPath.ReturnToPool();
				ThickenPathAsync(thickness);
				scheduler->Schedule(stream, [=] {
					// TODO: pass result for further work
					selfInPool.ReturnToPool();
					numRemaining--;
				});
			});
		}

		void PathThickenerJob::ThickenPathAsync(float thickness) {
		}

		void PathThickenerJob::Free(ResourceManager& resources) {
			cudaStreamDestroy(stream);
			resources.Free(distanceDevice);
		}

	}
}