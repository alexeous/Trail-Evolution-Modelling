#include "TramplabilityMask.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TramplabilityMask::TramplabilityMask(TramplabilityMaskHost* host, int graphW, int graphH) 
		  : EdgesDataDevice<bool>(graphW, graphH) 
		{
			host->CopyToSync(this, graphW, graphH);
		}

	}
}