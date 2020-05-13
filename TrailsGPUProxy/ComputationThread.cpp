#include "ComputationThread.h"
#include "CudaExceptionNative.h"
#include "CudaException.h"
#include "IsolatedAttractorsException.h"
#include "CudaScheduler.h"
#include "Attractor.h"
#include "AttractorsMap.h"
#include "EdgesData.h"
#include "TramplabilityMask.h"
#include "EdgesWeights.h"
#include "WavefrontJob.h"
#include "WavefrontJobsFactory.h"
#include "PathReconstructor.h"
#include "WavefrontCompletenessTable.h"

using namespace System;
using namespace System::Collections::Concurrent;
using namespace System::Threading;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ComputationThread::ComputationThread(TrailsGPUProxy^ proxy, TrailsComputationsInput^ input)
			: proxy(proxy), input(input)
		{
			thread = gcnew Thread(gcnew ThreadStart(this, &ComputationThread::ComputationProc));
			thread->IsBackground = true;
			thread->Start();
		}

		void ComputationThread::AbortWithException(std::exception_ptr ex) {
			try {
				std::rethrow_exception(ex);
			}
			catch(CudaExceptionNative ex) {
				exception = gcnew CudaException(ex);
			}
			catch(std::exception& ex) {
				std::string msg = ex.what();
				exception = gcnew Exception("An unmanaged exception occured: " + gcnew String(msg.c_str()));
			}
			catch(...) {
				exception = gcnew Exception("An unknown unmanaged exception occured");
			}
			thread->Abort();
		}

		void ComputationThread::CancelAll() {
			if(threadPool) {
				threadPool->CancelAll();
				threadPool = nullptr;
			}
			if(wavefrontTable) {
				wavefrontTable->CancelWait();
				wavefrontTable = nullptr;
			}
		}

		TrailsComputationsOutput^ ComputationThread::GetResult() {
			thread->Join();
			if (exception != nullptr)
			{
				throw exception;
			}
			return output;
		}

		void OnSchedulerException(std::exception_ptr ex, void *arg) {
			int threadId = (int)arg;
			ComputationThread^ thread;
			if(ComputationThread::runThreads->TryRemove(threadId, thread)) {
				thread->AbortWithException(ex);
			}
		}

		void ComputationThread::ComputationProc() {
			ResourceManager resources;

			int threadId = Thread::CurrentThread->ManagedThreadId;
			runThreads[threadId] = this;

			threadPool = new ThreadPool(&OnSchedulerException, (void*)threadId);
			CudaScheduler cudaScheduler(threadPool);
			try {
				Graph^ graph = input->Graph;

				NotifyProgress(L"Установление связей между точками притяжения");
				AttractorsMap attractors(graph, input->Attractors);

				NotifyProgress(L"Построение маски вытаптываемости");
				TramplabilityMask* tramplabilityMask = resources.New<TramplabilityMask>(graph);

				NotifyProgress(L"Инициализация весов рёбер для \"непорядочных пешеходов\"");
				EdgesWeights* edgesWeights = resources.New<EdgesWeights>(graph, resources, true);

				NotifyProgress(L"Создание исполнителей волнового алгоритма на GPU");
				std::vector<WavefrontJob*> wavefrontJobs =
					WavefrontJobsFactory::CreateJobs(graph->Width, graph->Height, &resources, attractors);
				
				PathReconstructor pathReconsturctor;
				WavefrontCompletenessTable wavefrontTable(attractors, &pathReconsturctor);
				this->wavefrontTable = &wavefrontTable;
				
				wavefrontTable.ResetCompleteness();
				for(auto job : wavefrontJobs) {
					job->Start(&wavefrontTable, edgesWeights, &cudaScheduler);
				}
				wavefrontTable.WaitForAll();

				NotifyProgress(L"Симуляция движения пешеходов");
				//wavefrontTable.ResetCompleteness();
				

				NotifyProgress(L"Выгрузка результата");
				EdgesDataHost<float>* trampledness = resources.New<EdgesDataHost<float>>(edgesWeights,
					input->Graph->Width, input->Graph->Height);

				resources.Free(edgesWeights);

				auto result = gcnew TrailsComputationsOutput();
				result->Graph = input->Graph;
				ApplyTrampledness(result->Graph, trampledness);

				output = result;
			}
			catch(ThreadAbortException^) {
				System::Threading::Thread::ResetAbort();
			}
			catch(CudaExceptionNative ex) {
				exception = gcnew CudaException(ex);
			}
			catch(std::exception& ex) {
				std::string msg = ex.what();
				exception = gcnew Exception("An unmanaged exception occured: " + gcnew String(msg.c_str()));
			}
			catch(...) {
				exception = gcnew Exception("An unknown unmanaged exception occured");
			}
			finally {
				resources.FreeAll();
				CancelAll();

				ComputationThread^ unused;
				runThreads->TryRemove(threadId, unused);
			}
		}

		void ComputationThread::NotifyProgress(const wchar_t* stage) {
			proxy->NotifyProgress(stage);
		}

		inline void ApplyTramplednessFunc(float& trampledness, Edge^ edge) {
			if(edge != nullptr) {
				edge->Trampledness = trampledness;
			}
		}

		inline void ComputationThread::ApplyTrampledness(Graph^ graph, EdgesDataHost<float>* edgesData) {
			edgesData->ZipWithGraphEdges(graph, ApplyTramplednessFunc);
		}

	}
}