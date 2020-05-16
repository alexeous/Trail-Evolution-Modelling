#pragma once
#include <stdexcept>
#include "TrailsGPUProxy.h"
#include "ThreadPool.h"
#include "NodesTramplingEffect.h"

using namespace System;
using namespace System::Collections::Concurrent;
using namespace System::Threading;
using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ref class ComputationThread {
		public:
			ComputationThread(TrailsGPUProxy^ proxy, TrailsComputationsInput^ input);
			TrailsComputationsOutput^ GetResult();
			void CancelAll();

		private:
			void ComputationProc();
			void NotifyProgress(const wchar_t* stage);

			void ApplyTrampledness(Graph^ graph, EdgesDataHost<float>* edgesData);

		internal:
			void AbortWithException(std::exception_ptr ex);

			static ConcurrentDictionary<int, ComputationThread^>^ runThreads = 
				gcnew ConcurrentDictionary<int, ComputationThread^>();

		private:
			TrailsGPUProxy^ proxy;
			TrailsComputationsInput^ input;
			TrailsComputationsOutput^ output;
			Thread^ thread;
			volatile Exception^ exception;

			ThreadPool* threadPool = nullptr;
			NodesTramplingEffect* pendingTramplingEffect = nullptr;
		};
	}
}