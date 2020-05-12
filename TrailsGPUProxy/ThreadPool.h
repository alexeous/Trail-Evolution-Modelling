#pragma once
#include <functional>
#include <stdexcept>


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class ThreadPool {
			friend class CudaScheduler;

		private:

			template<typename TFunction, typename ...TArgs>
			struct CallData {
				ThreadPool* pool;
				TFunction function;
				std::tuple<TArgs...> args;

				CallData(ThreadPool* pool, TFunction function, std::tuple<TArgs...> args)
					: pool(pool), function(function), args(args) {
				}
			};

			template<typename TFunction, typename ...TArgs>
			ref class Caller {
			public:
				Caller(CallData<TFunction, TArgs...>* callData)
					: callData(callData) {
					System::Threading::ThreadPool::QueueUserWorkItem(
						gcnew System::Threading::WaitCallback(this, &Caller::Call));
				}

			private:
				inline void Call(Object^) {
					try {
						std::apply(callData->function, callData->args);
					}
					catch(...) {
						callData->pool->InvokeExceptionCallback(std::current_exception());
					}
					delete callData;
				}

			private:
				CallData<TFunction, TArgs...>* callData;
			};

		public:
			inline ThreadPool(std::function<void(std::exception_ptr, void*)> onExceptionCallback,
				void* exceptionCallbackArg)
				: onExceptionCallback(onExceptionCallback),
				exceptionCallbackArg(exceptionCallbackArg) {
			}

			template <typename TFunction, typename... TArgs>
			inline void Schedule(TFunction function, TArgs&&... args) {
				auto callData = CreateCallData(function, args);
			}
			
			inline void InvokeExceptionCallback(std::exception_ptr ex) {
				if(isThrown) {
					return;
				}
				isThrown = true;
				onExceptionCallback(ex, exceptionCallbackArg);
			}

		private:
			template <typename TFunction, typename... TArgs>
			CallData<TFunction, TArgs...> CreateCallData(TFunction function, TArgs&&... args) {
				return new CallData<TFunction, TArgs...>(this, function, std::make_tuple(args...));
			}

		private:
			std::function<void(std::exception_ptr, void*)> onExceptionCallback;
			void* exceptionCallbackArg;
			bool isThrown = false;
		};

	}
}