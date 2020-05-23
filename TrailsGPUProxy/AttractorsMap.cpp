#include "AttractorsMap.h"
#include "IsolatedAttractorsException.h"

using namespace TrailEvolutionModelling::GPUProxy;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		AttractorsMap::AttractorsMap(Graph^ graph, array<RefAttractor^>^ refAttractors)
		{
			std::vector<Attractor> nativeAttractors = ConvertRefAttractors(refAttractors);
#define REF_TO_NATIVE(ref) nativeAttractors[Array::IndexOf(refAttractors, (ref))]

			auto isolated = gcnew List<RefAttractor^>();
			for each(auto attrI in refAttractors) {
				std::vector<Attractor> allReachable;
				float sumReachablePerformance = 0;
				for each(auto attrJ in refAttractors) {
					if(attrI == attrJ || !(attrI->IsSource && attrJ->IsDrain || 
										   attrI->IsDrain && attrJ->IsSource))
						continue;

					if(CanReach(graph, attrI, attrJ)) {
						Attractor reachable = REF_TO_NATIVE(attrJ);
						allReachable.push_back(reachable);
						sumReachablePerformance += reachable.performance;
					}
				}
				if(!allReachable.empty()) {
					Attractor attractor = REF_TO_NATIVE(attrI);
					(*this)[attractor] = allReachable;
					sumReachablePerformances[attractor] = sumReachablePerformance;
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

		float AttractorsMap::GetSumReachablePerformance(const Attractor& attractor) const {
			return sumReachablePerformances.at(attractor);
		}

		bool AttractorsMap::CanReach(Graph^ graph, RefAttractor^ a, RefAttractor^ b) {
			float distance = graph->Distance(a->Node, b->Node);
			return distance <= a->WorkingRadius
				&& distance <= b->WorkingRadius
				&& a->Node->ComponentParent == b->Node->ComponentParent;
		}

		std::vector<Attractor> AttractorsMap::ConvertRefAttractors(array<RefAttractor^>^ refAttractors) {
			std::vector<Attractor> attractors;
			for each(auto refAttr in refAttractors) {
				attractors.push_back(Attractor(refAttr));
			}
			return attractors;
		}

	}
}