#include "EdgesWeights.h"
#include "TrailsGPUProxy.h"
#include "TramplabilityMask.h"
#include "Constants.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		EdgesWeightsHost::EdgesWeightsHost(Graph^ graph, bool initiallyTrampleAll)
			: EdgesDataHost(graph->Width, graph->Height) 
		{
			InitFromGraph(graph, initiallyTrampleAll);
		}

		void EdgesWeightsHost::InitFromGraph(Graph^ graph, bool initiallyTrampleAll) {
			if(initiallyTrampleAll) {
				ZipWithGraphEdges(graph, [](float& weight, Edge^ edge) {
					if(edge == nullptr) {
						weight = INFINITY;
						return;
					}
					weight = edge->IsTramplable ? TRAMPLABLE_WEIGHT_FOR_INDECENT : edge->Weight;
				});
			}
			else {
				ZipWithGraphEdges(graph, [](float& weight, Edge^ edge) {
					if(edge == nullptr) {
						weight = INFINITY;
						return;
					}
					weight = edge->Weight;
				});
			}
		}

	}
}