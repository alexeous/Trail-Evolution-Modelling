#include "TrailsGPUProxy.h"
#include "IsolatedAttractorsException.h"
#include "kernel.h"
#include "EdgesData.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		TrailsComputationsOutput^ TrailsGPUProxy::ComputeTrails(TrailsComputationsInput^ input) {
			auto attractors = CreateAttractorsMap(input);

			return nullptr;
		}

		Dictionary<Attractor^, List<Attractor^>^>^ TrailsGPUProxy::CreateAttractorsMap(TrailsComputationsInput^ input) {
			auto map = gcnew Dictionary<Attractor^, List<Attractor^>^>();
			auto isolated = gcnew List<Attractor^>();
			for each(auto attrI in input->Attractors) {
				auto reachable = gcnew List<Attractor^>();
				map[attrI] = reachable;
				for each(auto attrJ in input->Attractors) {
					if(attrI == attrJ)
						continue;

					if(CanReach(input->Graph, attrI, attrJ)) {
						reachable->Add(attrJ);
					}
				}
				if(reachable->Count == 0) {
					isolated->Add(attrI);
				}
			}
			if(isolated->Count != 0) {
				throw gcnew IsolatedAttractorsException(isolated);
			}
			return map;
		}

		bool TrailsGPUProxy::CanReach(Graph^ graph, Attractor^ a, Attractor^ b) {
			float distance = graph->Distance(a->Node, b->Node);
			return distance <= a->WorkingRadius
				&& distance <= b->WorkingRadius
				&& a->Node->ComponentParent == b->Node->ComponentParent;
		}
	}
}