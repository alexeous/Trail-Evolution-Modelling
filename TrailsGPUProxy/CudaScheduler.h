#pragma once
#include <functional>
#include <stdexcept>
#include "cuda_runtime.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class CudaScheduler {
		private:

		private:
			template <typename TFunction, typename... TArgs>
			static inline void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* data);

		public:
			inline CudaScheduler(std::function<void(std::exception_ptr, void*)> onExceptionCallback,
				void* exceptionCallbackArg)
				: onExceptionCallback(onExceptionCallback),
				  exceptionCallbackArg(exceptionCallbackArg)
			{ }

			template <typename TFunction, typename... TArgs>
			inline void Schedule(cudaStream_t stream, TFunction function, TArgs&&... args) {
				auto callData = new CallData<TFunction, TArgs...>(this, function, std::make_tuple(args...));
				cudaStreamCallback_t callback = &CudaScheduler::Callback<TFunction, TArgs...>;
				CHECK_CUDA(cudaStreamAddCallback(stream, callback, callData, 0));
			}

			inline void InvokeExceptionCallback(std::exception_ptr ex) {
				if(isThrown) {
					return;
				}
				isThrown = true;
				onExceptionCallback(ex, exceptionCallbackArg);
			}

		private:
			std::function<void(std::exception_ptr, void*)> onExceptionCallback;
			void *exceptionCallbackArg;
			bool isThrown = false;
		};

		template<typename TFunction, typename ...TArgs>
		struct CallData {
			CudaScheduler* scheduler;
			TFunction function;
			std::tuple<TArgs...> args;

			CallData(CudaScheduler* scheduler, TFunction function, std::tuple<TArgs...> args)
				: scheduler(scheduler), function(function), args(args) {
			}
		};

		template<typename TFunction, typename ...TArgs>
		ref class Caller {
		public:
			Caller(CallData<TFunction, TArgs...>* callData)
				: callData(callData)
			{
				System::Threading::ThreadPool::QueueUserWorkItem(
					gcnew System::Threading::WaitCallback(this, &Caller::Call));
			}

		private:
			inline void Call(Object^) {
				try {
					std::apply(callData->function, callData->args);
				}
				catch(...) {
					callData->scheduler->InvokeExceptionCallback(std::current_exception());
				}
				delete callData;
			}

		private:
			CallData<TFunction, TArgs...>* callData;
		};

		template<typename TFunction, typename ...TArgs>
		void CUDART_CB CudaScheduler::Callback(cudaStream_t stream, cudaError_t status, void* data) {
			CHECK_CUDA(status);
			auto callData = reinterpret_cast<CallData<TFunction, TArgs...>*>(data);
			gcnew Caller<TFunction, TArgs...>(callData);
		}

	}
}
