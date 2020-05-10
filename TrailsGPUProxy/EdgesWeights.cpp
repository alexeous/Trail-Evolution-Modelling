#include "EdgesWeights.h"
#include "TrailsGPUProxy.h"
#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		EdgesWeights::EdgesWeights(Graph^ graph, ResourceManager& resources, bool initiallyTrampleAll)
			: EdgesDataDevice(graph->Width, graph->Height) 
		{
			EdgesDataHost<float> host = CreateHostWeights(graph, resources, initiallyTrampleAll);
			host.CopyTo(*this, graph->Width, graph->Height);
			resources.Free(host);
		}

		EdgesDataHost<float> EdgesWeights::CreateHostWeights(Graph^ graph, ResourceManager& resources, bool initiallyTrampleAll) {
			auto host = resources.New<EdgesDataHost<float>>(graph->Width, graph->Height);

			if(initiallyTrampleAll) {
				host.ZipWithGraphEdges(graph, [](float& weight, Edge^ edge) {
					weight = edge->IsTramplable ? TrailsGPUProxy::MinimumTramplableWeight : edge->Weight;
				});
			}
			else {
				host.ZipWithGraphEdges(graph, [](float& weight, Edge^ edge) {
					weight = edge->Weight;
				});
			}

			return host;
		}

	}
}