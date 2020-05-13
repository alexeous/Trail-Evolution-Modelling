#pragma once
#include <functional>
#include <stdexcept>
#include <gcroot.h>
#include "cuda_runtime.h"
#include "CudaUtils.h"
#include "ThreadPool.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class CudaScheduler {
		private:

			template<typename TFunction, typename ...TArgs>
			struct CallbackData {
				ThreadPool* threadPool;
				TFunction function;
				std::tuple<TArgs...> args;

				CallbackData(ThreadPool* threadPool, TFunction function,
					std::tuple<TArgs...> args
				)
					: threadPool(threadPool), function(function),
					args(args) {
				}
			};

		private:
			template <typename TFunction, typename... TArgs>
			static inline void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* data);

		public:
			inline CudaScheduler(ThreadPool* threadPool) 
				: threadPool(threadPool) { 
			}

			template <typename TFunction, typename... TArgs>
			inline void Schedule(cudaStream_t stream, TFunction function, TArgs&&... args) {
				auto callData = new CallbackData<TFunction, TArgs...>(threadPool,
					function, std::make_tuple(args...));

				cudaStreamCallback_t callback = &CudaScheduler::Callback<TFunction, TArgs...>;
				CHECK_CUDA(cudaStreamAddCallback(stream, callback, callData, 0));
			}

		private:
			ThreadPool* threadPool;
		};

		template<typename TFunction, typename ...TArgs>
		void CUDART_CB CudaScheduler::Callback(cudaStream_t stream, cudaError_t status, void* data) {
			CHECK_CUDA(status);
			auto callData = reinterpret_cast<CallbackData<TFunction, TArgs...>*>(data);
			callData->threadPool->Schedule(callData->function, callData->args);
		}

	}
}
