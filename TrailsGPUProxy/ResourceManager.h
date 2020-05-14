#pragma once
#include <unordered_set>
#include "IResource.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class ResourceManager {
		private:
		public:
			template<typename TResource, typename... TConstructorArgs>
			TResource* New(TConstructorArgs... constructorArgs);
			template<typename T> void Free(T*& resource);
			void FreeAll();

		private:
			std::unordered_set<IResource*> resources;
		};

		template<typename TResource, typename... TConstructorArgs>
		inline TResource* ResourceManager::New(TConstructorArgs... constructorArgs) {
			auto resource = new TResource(constructorArgs...);
			resources.insert(resource);
			return resource;
		}

		template<typename T>
		inline void ResourceManager::Free(T*& resource) {
			if(resource != nullptr && resources.erase(resource) != 0) {
				resource->Free(*this);
				delete resource;
				resource = nullptr;
			}
		}

	}
}
