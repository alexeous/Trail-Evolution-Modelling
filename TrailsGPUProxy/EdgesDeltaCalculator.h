#pragma once
#include "IResource.h"
#include "ResourceManager.h"
#include "EdgesWeights.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class EdgesDeltaCalculator : public IResource {
			friend class ResourceManager;
		
		public:
			float CalculateDelta(EdgesWeightsDevice* currentWeights);

		protected:
			EdgesDeltaCalculator(int graphW, int graphH, EdgesWeightsDevice* initialWeights, ResourceManager& resources);
			void Free(ResourceManager& resources) override;

		private:
			int graphW;
			int graphH;
			EdgesWeightsDevice* lastWeights = nullptr;
		};

	}
}