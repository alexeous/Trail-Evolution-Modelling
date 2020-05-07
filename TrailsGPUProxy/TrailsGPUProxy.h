#pragma once

using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class TrailsGPUProxy {
		public:
			static initonly int StepSeconds = 5 * 60;

			static TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);
		};
	}
}

