#include "TrailsGPUProxy.h"
#include "kernel.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		TrailsComputationsOutput^ TrailsGPUProxy::ComputeTrails(TrailsComputationsInput^ input) {
			auto output = gcnew TrailsComputationsOutput();

			const int arraySize = 5;
			const int a[arraySize] = { 1, 2, 3, 4, 5 };
			const int b[arraySize] = { 10, 20, 30, 40, 50 };
			int c[arraySize] = { 0 };
			addWithCuda(c, a, b, arraySize);

			output->Graph = gcnew Graph(c[0], c[1], c[2], c[3], c[4]);
			return output;
		}
	}
}