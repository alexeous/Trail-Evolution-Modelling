#pragma once
#include "ResourceManager.h"
#include "EdgesData.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		struct EdgesWeightsHost : public EdgesDataHost<float> {
			friend class ResourceManager;
			
			//void InitFromGraph(Graph^ graph);
			void InitFromGraph(Graph^ graph, bool trampleTramplable);
			void InitFromGraph(Graph^ graph, float replaceAllTramplable);

		protected:
			//EdgesWeightsHost(Graph^ graph);
			EdgesWeightsHost(Graph^ graph, bool trampleTramplable);

		};
#endif

		using EdgesWeightsDevice = EdgesDataDevice<float>;

	}
}