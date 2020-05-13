#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "IResource.h"
#include "Attractor.h"
#include "ComputeNodesHost.h"
#include "ComputeNodesPair.h"
#include "ResourceManager.h"
#include "CudaScheduler.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct WavefrontJob : public IResource {
			friend class ResourceManager;


		protected:
			WavefrontJob(int graphW, int graphH, Attractor goal, 
				const std::vector<Attractor>& starts, ResourceManager* resources);
			void Free() override;

		private:

			void ResetReadOnlyNodesGParallel();

			static int GetMinIterations(Attractor goal, const std::vector<Attractor>& starts);
			static ComputeNodesHost* CreateHostNodes(int w, int h, const std::vector<Attractor>& starts, 
				ResourceManager& resources);
			static ComputeNodesPair* CreateDeviceNodes(ComputeNodesHost* hostNodes,
				ResourceManager& resources);

		private:
			Attractor goal;
			int minIterations;
			ComputeNodesHost* hostNodes = nullptr;
			ComputeNodesPair* deviceNodes = nullptr;
			ResourceManager* resources;

			cudaStream_t stream;
		};

	}
}