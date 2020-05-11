#pragma once
#include "EdgesData.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class TrailsGPUProxy {
		public:
			static initonly int StepSeconds = 5 * 60;
			static initonly float MinimumTramplableWeight = 1.1f;

			event Action<String^>^ ProgressChanged;

			TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);

		private:
			void NotifyProgress(const wchar_t* stage);
			void ApplyTrampledness(Graph^ graph, EdgesDataHost<float>* edgesData);
		//private:
		//	Dictionary<Attractor^, List<Attractor^>^>^ CreateAttractorsMap(TrailsComputationsInput^ input);
		//	bool CanReach(Graph^ graph, Attractor^ a, Attractor^ b);
		};

		inline void ApplyTramplednessFunc(float& trampledness, Edge^ edge) {
			if(edge != nullptr) {
				edge->Trampledness = trampledness;
			}
		}

		inline void TrailsGPUProxy::ApplyTrampledness(Graph^ graph, EdgesDataHost<float>* edgesData) {
			edgesData->ZipWithGraphEdges(graph, ApplyTramplednessFunc);
		}
	}
}

