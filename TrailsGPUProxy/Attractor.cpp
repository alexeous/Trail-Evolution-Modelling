#include "Attractor.h"
#include <unordered_map>


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		Attractor::Attractor(RefAttractor^ refAttr) :
			nodeI(refAttr->Node->I),
			nodeJ(refAttr->Node->J),
			isSource(refAttr->IsSource),
			isDrain(refAttr->IsDrain),
			performance(refAttr->Performance) {
		}

		bool Attractor::operator==(const Attractor& other) const {
			return nodeI == other.nodeI
				&& nodeJ == other.nodeJ
				&& isSource == other.isSource
				&& isDrain == other.isDrain
				&& performance == other.performance;
		}

	}
}