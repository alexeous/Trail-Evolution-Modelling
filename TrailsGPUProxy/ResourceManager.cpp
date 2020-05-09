#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		void ResourceManager::FreeAll() {
			for(IResource* resource : resources) {
				resource->Free();
			}
		}

	}
}