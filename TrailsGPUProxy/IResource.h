#pragma once

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class IResource {
			friend class ResourceManager;

		protected:
			virtual void Free(ResourceManager& resources) = 0;
		};

	}
}
