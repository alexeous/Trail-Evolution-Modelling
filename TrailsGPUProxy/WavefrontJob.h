#pragma once
#include "IResource.h"
#include "Attractor.h"
#include "ComputeNodesHost.h"
#include "ComputeNodesPair.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct WavefrontJob : public IResource {
			friend class ResourceManager;

			void ResetReadOnlyNodesGParallel();

		protected:
			WavefrontJob(Attractor goal, ComputeNodesHost* nodesTemplate, ResourceManager* resources);
			void Free() override;

		private:
			static ComputeNodesPair* CreateDeviceNodes(ComputeNodesHost* nodesTemplate, 
				ResourceManager& resources);
			static ComputeNodesHost* CreateHostNodes(ComputeNodesHost* nodesTemplate,
				ResourceManager& resources);

		private:
			Attractor goal;
			ComputeNodesHost* hostNodes = nullptr;
			ComputeNodesPair* deviceNodes = nullptr;
			ResourceManager* resources;
		};

	}
}