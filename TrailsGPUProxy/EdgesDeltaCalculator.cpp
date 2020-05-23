#include "EdgesDeltaCalculator.h"
#include "cuda_runtime.h"
#include "CudaUtils.h"
#include "EdgesDeltaCalcKernel.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		EdgesDeltaCalculator::EdgesDeltaCalculator(int graphW, int graphH,
			EdgesWeightsDevice* initialWeights, ResourceManager& resources)
			: graphW(graphW),
			  graphH(graphH),
			  lastWeights(resources.New<EdgesWeightsDevice>(initialWeights, graphW, graphH))
		{
		}

		float EdgesDeltaCalculator::CalculateDelta(EdgesWeightsDevice* currentWeights) {
			float delta = CalcEdgesDelta(lastWeights, currentWeights, graphW, graphH);
			currentWeights->CopyToSync(lastWeights, graphW, graphH);
			return delta;
		}

		void EdgesDeltaCalculator::Free(ResourceManager& resources) {
			resources.Free(lastWeights);
		}

	}
}
