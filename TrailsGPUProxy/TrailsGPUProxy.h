#pragma once

using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class TrailsGPUProxy {
		public:
			static TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);
		};
	}
}

