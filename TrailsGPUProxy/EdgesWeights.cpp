#include "EdgesWeights.h"
#include "TrailsGPUProxy.h"
#include "TramplabilityMask.h"
#include "Constants.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		EdgesWeightsHost::EdgesWeightsHost(Graph^ graph)
			: EdgesDataHost(graph->Width, graph->Height) 
		{
			InitFromGraph(graph);
		}

		EdgesWeightsHost::EdgesWeightsHost(Graph^ graph, float setAllTramplable)
			: EdgesDataHost(graph->Width, graph->Height) {
			InitFromGraph(graph, setAllTramplable);
		}

		void EdgesWeightsHost::InitFromGraph(Graph^ graph) {
			void (*func)(float&, Edge^) = [](float& weight, Edge^ edge) {
				if(edge == nullptr) {
					weight = INFINITY;
					return;
				}
				weight = edge->Weight;
			};
			ZipWithGraphEdges(graph, func);
		}

		void EdgesWeightsHost::InitFromGraph(Graph^ graph, float setAllTramplable) {
			void(*func)(float&, Edge^, float) = [](float& weight, Edge^ edge, float setAllTramplable) {
				if(edge == nullptr) {
					weight = INFINITY;
					return;
				}
				weight = edge->IsTramplable ? setAllTramplable : edge->Weight;
			};
			ZipWithGraphEdges(graph, func, setAllTramplable);
		}

	}
}