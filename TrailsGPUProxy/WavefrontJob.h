#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "IResource.h"
#include "Attractor.h"
#include "ComputeNodesHost.h"
#include "ComputeNodesPair.h"
#include "ResourceManager.h"
#include "CudaScheduler.h"
#include "ExitFlag.h"
#include "WavefrontCompletenessTable.h"
#include "EdgesWeights.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct WavefrontJob : public IResource {
			friend class ResourceManager;

		public:
			void Start(WavefrontCompletenessTable* wavefrontTable, EdgesWeights* edges,
				CudaScheduler* scheduler);

		protected:
			WavefrontJob(int graphW, int graphH, Attractor goal, 
				const std::vector<Attractor>& starts, ResourceManager* resources);
			void Free(ResourceManager& resources) override;

		private:

			void ResetReadOnlyNodesGParallelAsync();
			int GetGoalIndex();

			static int GetMinIterations(Attractor goal, const std::vector<Attractor>& starts);
			static ComputeNodesHost* CreateHostNodes(int w, int h, const std::vector<Attractor>& starts, 
				ResourceManager* resources);
			static ComputeNodesPair* CreateDeviceNodes(ComputeNodesHost* hostNodes,
				ResourceManager* resources);

		private:
			Attractor goal;
			int minIterations;
			ComputeNodesHost* hostNodes = nullptr;
			ComputeNodesPair* deviceNodes = nullptr;

			float* maxAgentsGPerGroup;
			ExitFlag* exitFlag;
			std::function<void(int)>* withoutExitFlagCheck = nullptr;
			std::function<void()>* withExitFlagCheck = nullptr;

			cudaStream_t stream;
		};

	}
}