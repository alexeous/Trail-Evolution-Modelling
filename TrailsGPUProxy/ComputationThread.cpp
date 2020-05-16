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
#include "PathThickener.h"
#include "WavefrontCompletenessTable.h"
#include "NodesTramplingEffect.h"

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
			catch(Exception^ ex) {
				exception = ex;
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
			if(pendingTramplingEffect) {
				pendingTramplingEffect->CancelWaiting();
				pendingTramplingEffect = nullptr;
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
				int w = graph->Width;
				int h = graph->Height;
				float stepMeters = graph->StepPhysicalMeters;

				NotifyProgress(L"Установление связей между точками притяжения");
				AttractorsMap attractors(graph, input->Attractors);

				NotifyProgress(L"Построение маски вытаптываемости");
				TramplabilityMask* tramplabilityMask = resources.New<TramplabilityMask>(graph, resources);

				NotifyProgress(L"Инициализация весов рёбер для \"непорядочных пешеходов\"");
				EdgesWeightsHost* edgesHost = resources.New<EdgesWeightsHost>(graph, true);
				EdgesWeightsDevice* edgesDevice = resources.New<EdgesWeightsDevice>(edgesHost, w, h);

				NotifyProgress(L"Создание исполнителей волнового алгоритма на GPU");
				std::vector<WavefrontJob*> wavefrontJobs = WavefrontJobsFactory::CreateJobs(w, h, &resources, attractors);
				
				NodesTramplingEffect* nodesTramplingEffect = resources.New<NodesTramplingEffect>(w, h, 
					stepMeters, INDECENT_PEDESTRIANS_SHARE, &resources);

				PathThickener *pathThickener = resources.New<PathThickener>(w, h, stepMeters, 
					FIRST_PHASE_PATH_THICKNESS, tramplabilityMask, nodesTramplingEffect, &resources);

				PathReconstructor *pathReconsturctor = resources.New<PathReconstructor>(w, h,
					edgesHost, &cudaScheduler, &resources, pathThickener);

				WavefrontCompletenessTable wavefrontTable(attractors, pathReconsturctor);

				NotifyProgress(L"Вычисление эффекта вытаптывания от \"непорядочных пешеходов\"");
				nodesTramplingEffect->SetAwaitedPathsNumber(wavefrontTable.numPaths);
				pendingTramplingEffect = nodesTramplingEffect;
				nodesTramplingEffect->ClearSync();
				
				wavefrontTable.ResetCompleteness();
				for(auto job : wavefrontJobs) {
					job->Start(&wavefrontTable, edgesDevice, &cudaScheduler);
				}
				
				pendingTramplingEffect->AwaitAllPaths();

				EdgesDataDevice<float>* indecentTrampling = resources.New<EdgesDataDevice<float>>(w, h);
				nodesTramplingEffect->SaveAsEdgesSync(indecentTrampling, tramplabilityMask);

				EdgesWeightsDevice* maximumWeights = resources.New<EdgesWeightsDevice>(w, h);
				edgesDevice->CopyToSync(maximumWeights, w, h);

				NotifyProgress(L"Выгрузка результата");
				EdgesDataHost<float>* trampledness = resources.New<EdgesDataHost<float>>(indecentTrampling, w, h);

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
			catch(Exception^ ex) {
				exception = ex;
			}
			catch(std::exception& ex) {
				std::string msg = ex.what();
				exception = gcnew Exception("An unmanaged exception occured: " + gcnew String(msg.c_str()));
			}
			catch(...) {
#ifdef _DEBUG
				throw;
#else
				exception = gcnew Exception("An unknown unmanaged exception occured");
#endif
			}
			finally {
				CancelAll();
				resources.FreeAll();

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