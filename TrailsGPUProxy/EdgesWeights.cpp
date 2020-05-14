#include "EdgesWeights.h"
#include "TrailsGPUProxy.h"
#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		EdgesWeightsHost::EdgesWeightsHost(Graph^ graph, bool initiallyTrampleAll)
			: EdgesDataHost(graph->Width, graph->Height) 
		{
			if(initiallyTrampleAll) {
				ZipWithGraphEdges(graph, [](float& weight, Edge^ edge) {
					if(edge == nullptr) {
						weight = INFINITY;
						return;
					}
					weight = edge->IsTramplable ? TrailsGPUProxy::MinimumTramplableWeight : edge->Weight;
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