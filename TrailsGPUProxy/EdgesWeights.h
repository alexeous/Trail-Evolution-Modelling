#pragma once
#include "ResourceManager.h"
#include "EdgesData.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		struct EdgesWeightsHost : public EdgesDataHost<float> {
			friend class ResourceManager;
			
			void InitFromGraph(Graph^ graph);
			void InitFromGraph(Graph^ graph, float setAllTramplable);

		protected:
			EdgesWeightsHost(Graph^ graph);
			EdgesWeightsHost(Graph^ graph, float setAllTramplable);

		};
#endif

		using EdgesWeightsDevice = EdgesDataDevice<float>;

	}
}