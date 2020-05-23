#pragma once
#include "EdgesWeights.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		float CalcEdgesDelta(EdgesWeightsDevice* lastWeights, EdgesWeightsDevice* currentWeights, int graphW, int graphH);

	}
}
