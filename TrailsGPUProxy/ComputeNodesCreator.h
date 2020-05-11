#pragma once
#include <unordered_map>
#include "Attractor.h"
#include "AttractorsMap.h"
#include "ComputeNode.h"
#include "ComputeNodesPair.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		static class ComputeNodesCreator {
		public:
			static std::unordered_map<Attractor&, ComputeNodesPair> Create(
				Graph^ graph, const AttractorsMap& attractors, ResourceManager& resources);
		};

	}
}