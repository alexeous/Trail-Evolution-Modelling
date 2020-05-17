#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TramplabilityMask::TramplabilityMask(Graph^ graph, ResourceManager& resources)
			: EdgesDataDevice<bool>(graph->Width, graph->Height) 
		{
			EdgesDataHost<bool>* host = resources.New<EdgesDataHost<bool>>(graph->Width, graph->Height);

			void (*func)(bool& tramplable, Edge^ edge) = [](bool& tramplable, Edge^ edge) {
				tramplable = (edge != nullptr && edge->IsTramplable);
			};
			host->ZipWithGraphEdges(graph, func);
			host->CopyToSync(this, graph->Width, graph->Height);

			resources.Free(host);
		}
	}
}