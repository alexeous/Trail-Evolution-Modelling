#pragma once
#include "EdgesData.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		using namespace TrailEvolutionModelling::GraphTypes;
#endif

		struct TramplabilityMaskHost : public EdgesDataHost<bool> {
			friend class ResourceManager;

#ifndef __CUDACC__
		protected:
			TramplabilityMaskHost(Graph^ graph);
#endif
		};

	}
}