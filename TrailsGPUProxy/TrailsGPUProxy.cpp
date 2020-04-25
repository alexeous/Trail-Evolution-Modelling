#include "TrailsGPUProxy.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		TrailsComputationsOutput^ TrailsGPUProxy::ComputeTrails(TrailsComputationsInput^ input) {
			auto output = gcnew TrailsComputationsOutput();
			output->Graph = gcnew Graph(1, 3, 3, 7, 0);
			return output;
		}
	}
}