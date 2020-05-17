#pragma once
#include "ResourceManager.h"
#include "EdgesData.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		struct EdgesWeightsHost : public EdgesDataHost<float> {
			friend class ResourceManager;
			
			void InitFromGraph(Graph^ graph, bool initiallyTrampleAll);

		protected:
			EdgesWeightsHost(Graph^ graph, bool initiallyTrampleAll);
		};
#endif

		using EdgesWeightsDevice = EdgesDataDevice<float>;

	}
}