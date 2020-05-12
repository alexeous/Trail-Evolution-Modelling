#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "IResource.h"
#include "Attractor.h"
#include "ComputeNodesHost.h"
#include "ComputeNodesPair.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct WavefrontJob : public IResource {
			friend class ResourceManager;


		protected:
			WavefrontJob(Attractor goal, const std::vector<Attractor>& starts,
				ComputeNodesHost* nodesTemplate, ResourceManager* resources);
			void Free() override;

		private:
			void ResetReadOnlyNodesGParallel();

			static int GetMinIterations(Attractor goal, const std::vector<Attractor>& starts);
			static ComputeNodesPair* CreateDeviceNodes(ComputeNodesHost* nodesTemplate, 
				ResourceManager& resources);
			static ComputeNodesHost* CreateHostNodes(ComputeNodesHost* nodesTemplate,
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