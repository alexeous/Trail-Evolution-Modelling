#pragma once
#include "ResourceManager.h"
#include "EdgesData.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct EdgesWeights : public EdgesDataDevice<float> {
			friend class ResourceManager;

		protected:
			EdgesWeights(Graph^ graph, ResourceManager& resources, bool initiallyTrampleAll);

		private:
			static EdgesDataHost<float> CreateHostWeights(Graph^ graph,
				ResourceManager& resources, bool initiallyTrampleAll);
		};

	}
}