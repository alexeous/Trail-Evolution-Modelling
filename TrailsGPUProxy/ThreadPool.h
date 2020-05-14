#pragma once
#include <functional>
#include <stdexcept>
#include <gcroot.h>


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using namespace System::Threading;

		class ThreadPool {
			friend class CudaScheduler;

		public:

			template<typename TFunction, typename ...TArgs>
			struct CallData {
				ThreadPool* pool;
				TFunction function;
				std::tuple<TArgs...> args;
				gcroot<CancellationToken> cancellationToken;

				CallData(ThreadPool* pool, TFunction function, 
					std::tuple<TArgs...> args, CancellationToken cancellationToken
				)
					: pool(pool), function(function), 
					  args(args), cancellationToken(cancellationToken) 
				{
				}
			};

			template<typename TFunction, typename ...TArgs>
			ref class Caller {
			public:
				inline Caller(CallData<TFunction, TArgs...>* callData)
					: callData(callData)
				{
					this->cancellationToken = callData->cancellationToken;
					System::Threading::ThreadPool::QueueUserWorkItem(gcnew WaitCallback(this, &Caller::Call));
				}

			private:
				inline void Call(Object^) {
					if(cancellationToken.IsCancellationRequested) {
						delete callData;
						return;
					}
					try {
						std::apply(callData->function, callData->args);
					}
					catch(...) {
						if(!cancellationToken.IsCancellationRequested) {
							callData->pool->InvokeExceptionCallback(std::current_exception());
						}
					}
					delete callData;
				}

			private:
				CallData<TFunction, TArgs...>* callData;
				CancellationToken cancellationToken;
			};

		public:
			ThreadPool(std::function<void(std::exception_ptr, void*)> onExceptionCallback, 
				void* exceptionCallbackArg);
			~ThreadPool();

			template <typename TFunction, typename... TArgs>
			inline void Schedule(TFunction function, TArgs... args) {
				if(cancellation->IsCancellationRequested)
					return;

				auto callData = CreateCallData(function, args...);
				gcnew Caller<TFunction, TArgs...>(callData);
			}

			template <typename TFunction, typename... TArgs>
			inline void Schedule(TFunction function, std::tuple<TArgs...> argsTuple) {
				if(cancellation->IsCancellationRequested)
					return;

				auto callData = CreateCallData(function, argsTuple);
				gcnew Caller<TFunction, TArgs...>(callData);
			}

			void CancelAll();
			void InvokeExceptionCallback(std::exception_ptr ex);

		public:
			template <typename TFunction, typename... TArgs>
			inline CallData<TFunction, TArgs...>* CreateCallData(TFunction function, TArgs&&... args) {
				return new CallData<TFunction, TArgs...>(this, function, 
					std::make_tuple(args...), cancellation->Token);
			}

			template <typename TFunction, typename... TArgs>
			inline CallData<TFunction, TArgs...>* CreateCallData(TFunction function, std::tuple<TArgs...> argsTuple) {
				return new CallData<TFunction, TArgs...>(this, function, 
					argsTuple, cancellation->Token);
			}

		private:
			std::function<void(std::exception_ptr, void*)> onExceptionCallback;
			void* exceptionCallbackArg;
			bool isThrown = false;
			gcroot<CancellationTokenSource^> cancellation;
		};

	}
}