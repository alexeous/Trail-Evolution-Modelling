#pragma once
#include "EdgesData.h"
#include <cstdint>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		using namespace TrailEvolutionModelling::GraphTypes;
#endif

		struct TramplabilityMask : public EdgesDataDevice<bool> {
			friend class ResourceManager;

#ifndef __CUDACC__
		protected:
			TramplabilityMask(Graph^ graph, ResourceManager& resources);
#endif
		};

	}
}