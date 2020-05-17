#include "EdgesWeights.h"
#include "TrailsGPUProxy.h"
#include "TramplabilityMask.h"
#include "Constants.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		//EdgesWeightsHost::EdgesWeightsHost(Graph^ graph)
		//	: EdgesDataHost(graph->Width, graph->Height) 
		//{
		//	InitFromGraph(graph);
		//}

		EdgesWeightsHost::EdgesWeightsHost(Graph^ graph, bool trampleTramplable)
			: EdgesDataHost(graph->Width, graph->Height) {
			InitFromGraph(graph, trampleTramplable);
		}

		//void EdgesWeightsHost::InitFromGraph(Graph^ graph) {
		//	
		//	ZipWithGraphEdges(graph, func);
		//}

		void EdgesWeightsHost::InitFromGraph(Graph^ graph, float replaceAllTramplable) {
			void(*func)(float&, Edge^, float) = [](float& weight, Edge^ edge, float replaceTramplable) {
				if(edge == nullptr) {
					weight = INFINITY;
					return;
				}
				weight = edge->IsTramplable ? replaceTramplable : edge->Weight;
			};
			ZipWithGraphEdges(graph, func, replaceAllTramplable);
		}

		void EdgesWeightsHost::InitFromGraph(Graph^ graph, bool trampleTramplable) {
			void(*func)(float&, Edge^);
			if(trampleTramplable) {
				func = [](float& weight, Edge^ edge) {
					if(edge == nullptr) {
						weight = INFINITY;
						return;
					}
					weight = edge->IsTramplable 
						? edge->Weight / TRAMPLABLE_WEIGHT_REDUCTION_FACTOR_FOR_INDECENT 
						: edge->Weight;
					weight = std::fmaxf(MIN_TRAMPLABLE_WEIGHT, weight);
				};
			}
			else {
				func = [](float& weight, Edge^ edge) {
					if(edge == nullptr) {
						weight = INFINITY;
						return;
					}
					weight = edge->Weight;
				};
			}
			ZipWithGraphEdges(graph, func);
		}

	}
}