#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		void ResourceManager::FreeAll() {
			while(!resources.empty()) {
				IResource* resource = *resources.begin();
				Free(resource);
			}
		}

	}
}