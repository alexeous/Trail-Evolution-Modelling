#include "TrailsGPUProxy.h"
#include "ComputationThread.h"

using namespace System::Threading;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TrailsComputationsOutput^ TrailsGPUProxy::ComputeTrails(TrailsComputationsInput^ input) {
			ComputationThread^ computationThread = gcnew ComputationThread(this, input);
			return computationThread->GetResult();
		}

		void TrailsGPUProxy::NotifyProgress(const wchar_t* stage) {
			ProgressChanged(gcnew String(stage));
		}
	}
}