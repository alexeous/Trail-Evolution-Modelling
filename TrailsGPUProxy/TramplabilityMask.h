#pragma once
#include "EdgesData.h"
#ifndef __CUDACC__
#include "TramplabilityMaskHost.h"
#endif

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct TramplabilityMask : public EdgesDataDevice<bool> {
			friend class ResourceManager;

#ifndef __CUDACC__
		protected:
			TramplabilityMask(TramplabilityMaskHost* host, int graphW, int graphH);
#endif
		};

	}
}