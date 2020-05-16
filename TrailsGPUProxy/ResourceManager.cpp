#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		void ResourceManager::FreeAll() {
			while(true) {
				IResource* next = nullptr;
				
				Monitor::Enter(sync);
				try {
					if(!resources.empty()) {
						next = *resources.begin();
					}
					Monitor::Exit(sync);
				}
				catch(...) {
					Monitor::Exit(sync);
					throw;
				}
				
				if(next == nullptr) {
					break;
				}
				
				Free(next);
			}
		}

	}
}