#include "AttractorsMap.h"
#include "IsolatedAttractorsException.h"

using namespace TrailEvolutionModelling::GPUProxy;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		AttractorsMap::AttractorsMap(Graph^ graph, array<RefAttractor^>^ refAttractors)
			: numSources(0), numDrains(0)
		{
			std::vector<Attractor> attractors;
			for each(auto refAttr in refAttractors) {
				attractors.push_back(Attractor(refAttr));
				
				if(refAttr->IsSource)
					numSources++;
				if(refAttr->IsDrain)
					numDrains++;
			}

#define REF_TO_NATIVE(ref) attractors[Array::IndexOf(refAttractors, (ref))]

			auto isolated = gcnew List<RefAttractor^>();
			for each(auto attrI in refAttractors) {
				std::vector<Attractor> reachable;
				for each(auto attrJ in refAttractors) {
					if(attrI == attrJ || !(attrI->IsSource && attrJ->IsDrain || 
										   attrI->IsDrain && attrJ->IsSource))
						continue;

					if(CanReach(graph, attrI, attrJ)) {
						reachable.push_back(REF_TO_NATIVE(attrJ));
					}
				}
				if(!reachable.empty()) {
					(*this)[REF_TO_NATIVE(attrI)] = reachable;
				}
				else {
					isolated->Add(attrI);
				}
			}

			if(isolated->Count != 0) {
				throw gcnew IsolatedAttractorsException(isolated);
			}

#undef REF_TO_NATIVE
		}

		bool AttractorsMap::CanReach(Graph^ graph, RefAttractor^ a, RefAttractor^ b) {
			float distance = graph->Distance(a->Node, b->Node);
			return distance <= a->WorkingRadius
				&& distance <= b->WorkingRadius
				&& a->Node->ComponentParent == b->Node->ComponentParent;
		}

	}
}