#include "TrailsGPUProxy.h"
#include "ComputationThread.h"

using namespace System::Threading;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TrailsComputationsOutput^ TrailsGPUProxy::ComputeTrails(TrailsComputationsInput^ input) {
			ComputationThread^ computationThread = gcnew ComputationThread(this, input);
			try {
				return computationThread->GetResult();
			}
			catch(ThreadAbortException^ ex) {
				if(GiveUnripeResultFlag) {
					Thread::ResetAbort();
					computationThread->GiveUnripeResultImmediate();
					return computationThread->GetResult();
				}
				computationThread->CancelAll();
				throw ex;
			}
		}

		void TrailsGPUProxy::NotifyProgress(String^ stage) {
			ProgressChanged(stage);
		}

		void TrailsGPUProxy::NotifyCanGiveUnripeResult() {
			CanGiveUnripeResult();
		}
	}
}