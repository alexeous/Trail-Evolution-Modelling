#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TramplabilityMask::TramplabilityMask(Graph^ graph)
			: EdgesDataHost<uint8_t>(graph->Width, graph->Height) {
			int w = graph->Width;
			int h = graph->Height;
			for(int i = 0; i < w; i++) {
				for(int j = 0; j < h; j++) {
					bool notLastColumn = i < w - 1;
					bool notLastRow = j < h - 1;
					bool notFirstColumn = i != 0;

					Node^ node = graph->GetNodeAtOrNull(i, j);

					E(i, j, w) = IsTramplable(node, Direction::E);
					if(notLastRow) {
						S(i, j, w) = IsTramplable(node, Direction::S);
						if(notLastColumn)
							SE(i, j, w) = IsTramplable(node, Direction::SE);
						if(notFirstColumn)
							SW(i, j, w) = IsTramplable(node, Direction::SW);
					}
				}
			}
		}

		inline bool TramplabilityMask::IsTramplable(Node^ node, Direction direction) {
			if(node == nullptr)
				return false;

			Edge^ edge = node->GetIncidentEdge(direction);
			return edge != nullptr && edge->IsTramplable;
		}

	}
}