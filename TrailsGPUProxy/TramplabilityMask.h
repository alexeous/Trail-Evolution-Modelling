#pragma once
#include "EdgesData.h"
#include <cstdint>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using namespace TrailEvolutionModelling::GraphTypes;

		struct TramplabilityMask : public EdgesDataHost<uint8_t> {
			friend class ResourceManager;

		protected:
			TramplabilityMask(Graph^ graph);
		};

	}
}