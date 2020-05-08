#pragma once
#include <unordered_map>
#include <vector>

using namespace System::Collections::Generic;
using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class TrailsGPUProxy {
		public:
			static initonly int StepSeconds = 5 * 60;

			static TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);

		private:
			static Dictionary<Attractor^, List<Attractor^>^>^ CreateAttractorsMap(TrailsComputationsInput^ input);
			static bool CanReach(Graph^ graph, Attractor^ a, Attractor^ b);
		};
	}
}

