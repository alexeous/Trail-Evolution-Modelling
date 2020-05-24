#include "TramplabilityMaskHost.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TramplabilityMaskHost::TramplabilityMaskHost(Graph^ graph)
			: EdgesDataHost<bool>(graph->Width, graph->Height)
		{
			void (*func)(bool& tramplable, Edge^ edge) = [](bool& tramplable, Edge^ edge) {
				tramplable = (edge != nullptr && edge->IsTramplable);
			};
			ZipWithGraphEdges(graph, func);
		}

	}
}