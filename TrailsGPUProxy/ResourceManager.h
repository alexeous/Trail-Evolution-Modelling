#pragma once
#include <unordered_set>

#ifndef __CUDACC__
#include <gcroot.h>
#endif

#include "IResource.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		using namespace System;
		using namespace System::Threading;
#endif

		class ResourceManager {
#ifndef __CUDACC__
		public:
			template<typename TResource, typename... TConstructorArgs>
			TResource* New(TConstructorArgs... constructorArgs);
			template<typename T> void Free(T*& resource);
			void FreeAll();

		private:
			std::unordered_set<IResource*> resources;
			gcroot<Object^> sync = gcnew Object();
#endif
		};

#ifndef __CUDACC__
		template<typename TResource, typename... TConstructorArgs>
		inline TResource* ResourceManager::New(TConstructorArgs... constructorArgs) {
			auto resource = new TResource(constructorArgs...);

			Monitor::Enter(sync);
			try {
				resources.insert(resource);
				Monitor::Exit(sync);
			}
			catch(...) {
				Monitor::Exit(sync);
				throw;
			}
			
			return resource;
		}

		template<typename T>
		inline void ResourceManager::Free(T*& resource) {
			if(resource == nullptr)
				return;

			bool hadResource;

			Monitor::Enter(sync);
			try {
				hadResource = resources.erase(resource) != 0;
				Monitor::Exit(sync);
			}
			catch(...) {
				Monitor::Exit(sync);
				throw;
			}

			if(hadResource) {
				resource->Free(*this);
				delete resource;
				resource = nullptr;
			}
		}
#endif

	}
}
