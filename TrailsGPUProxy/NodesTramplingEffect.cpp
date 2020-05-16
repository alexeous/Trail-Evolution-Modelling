#include "NodesTramplingEffect.h"
#include "NodesApplyTramplingKernel.h"
#include "SaveNodesTramplingAsEdgesKernel.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		NodesTramplingEffect::NodesTramplingEffect(int graphW, int graphH, float graphStep, 
			float performanceFactor, ResourceManager* resources)
			: graphW(graphW),
			  graphH(graphH),
			  graphStep(graphStep),
			  performanceFactor(performanceFactor),
			  effectDataDevice(resources->New<NodesFloatDevice>(graphW, graphH))
		{
			InitWaitObject();
		}

		void NodesTramplingEffect::InitWaitObject() {
			waitObj = gcnew Object();
		}

		void NodesTramplingEffect::Free(ResourceManager& resources) {
			resources.Free(effectDataDevice);
		}


		void NodesTramplingEffect::ClearSync() {
			size_t size = NodesDataHaloed<float>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemset(effectDataDevice->data, 0, size));
		}

		void NodesTramplingEffect::ApplyTramplingAsync(PoolEntry<DistancePairDevice*> distancePairEntry, float pathThickness, 
			float peoplePerSecond, PoolEntry<cudaStream_t> streamEntry, CudaScheduler* scheduler)
		{
			NodesFloatDevice* distanceToPath = distancePairEntry.object->readOnly;
			cudaStream_t stream = streamEntry.object;

			float tramplingCoefficient = CalcTramplingFactor(peoplePerSecond);
			CHECK_CUDA(NodesApplyTramplingEffect(effectDataDevice, distanceToPath,
				graphW, graphH, pathThickness, tramplingCoefficient, stream));

			scheduler->Schedule(stream, [=] {
				distancePairEntry.ReturnToPool();
				streamEntry.ReturnToPool();
				DecrementAwaitedPathNumber();
			});
		}

		void NodesTramplingEffect::SaveAsEdgesSync(EdgesDataDevice<float>* target, 
			TramplabilityMask* tramplabilityMask) 
		{
			SaveNodesTramplingAsEdges(effectDataDevice, graphW, graphH, target, tramplabilityMask);
		}

		float NodesTramplingEffect::CalcTramplingFactor(float peoplePerSecond) {
			return TRAMPLING_EFFECT_PER_HUMAN_STEP * HUMAN_STEPS_PER_METER * graphStep *
				peoplePerSecond * performanceFactor * SIMULATION_STEP_SECONDS;
		}


		void NodesTramplingEffect::SetAwaitedPathsNumber(int numAwaitedPaths) {
			this->numAwaitedPaths = numAwaitedPaths;
			if(numAwaitedPaths <= 0) {
				CancelWaiting();
			}
		}

		void NodesTramplingEffect::DecrementAwaitedPathNumber() {
			numAwaitedPaths--;
			if(numAwaitedPaths <= 0) {
				CancelWaiting();
			}
		}

		void NodesTramplingEffect::AwaitAllPaths() {
			Monitor::Enter(waitObj);
			Monitor::Wait(waitObj);
			Monitor::Exit(waitObj);
		}

		void NodesTramplingEffect::CancelWaiting() {
			Monitor::Enter(waitObj);
			Monitor::PulseAll(waitObj);
			Monitor::Exit(waitObj);
		}

	}
}