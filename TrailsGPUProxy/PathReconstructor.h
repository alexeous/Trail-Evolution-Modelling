#pragma once
#include "Attractor.h"
#include "ComputeNodesHost.h"


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class PathReconstructor {
		public:
			void StartPathReconstruction(const Attractor& start, const Attractor& goal,
				ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes);
		};

	}
}