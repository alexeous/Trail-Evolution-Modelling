#include "ThreadPool.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TrailEvolutionModelling::GPUProxy::ThreadPool::ThreadPool(std::function<void(std::exception_ptr, void*)> onExceptionCallback,
			void* exceptionCallbackArg
		)
			: onExceptionCallback(onExceptionCallback),
			exceptionCallbackArg(exceptionCallbackArg),
			cancellation(gcnew CancellationTokenSource()) 
		{
		}

		ThreadPool::~ThreadPool() {
			CancelAll();
		}

		void ThreadPool::CancelAll() {
			cancellation->Cancel();
		}

		void ThreadPool::InvokeExceptionCallback(std::exception_ptr ex) {
			cancellation->Cancel();
			if(isThrown) {
				return;
			}
			isThrown = true;
			onExceptionCallback(ex, exceptionCallbackArg);
		}

	}
}