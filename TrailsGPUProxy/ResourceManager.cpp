#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		void ResourceManager::FreeAll() {
			decltype(resources) resourcesCopy(resources);
			for(IResource* resource : resourcesCopy) {
				resource->Free();
			}
			resources.clear();
		}

	}
}