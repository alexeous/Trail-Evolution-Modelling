#pragma once
#include "IsolatedAttractorsException.h"
#include <unordered_map>
#include <vector>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using namespace System::Collections::Generic;
		using RefAttractor = TrailEvolutionModelling::GraphTypes::Attractor;
		using Graph = TrailEvolutionModelling::GraphTypes::Graph;

		struct Attractor {
			int nodeI, nodeJ;
			float performance;

			Attractor() = default;
			Attractor(RefAttractor^ refAttractor);

			bool operator==(const Attractor& other) const;
		};

	}
}