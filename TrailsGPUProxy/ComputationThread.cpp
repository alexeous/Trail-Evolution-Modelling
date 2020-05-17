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
#include "EdgesTramplingEffect.h"
#include "ApplyTramplingsAndLawnRegeneration.h"

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

				NotifyProgress(L"[Фаза 1] Инициализация весов рёбер для \"непорядочных пешеходов\"");
				EdgesWeightsHost* edgesHost = resources.New<EdgesWeightsHost>(graph, false);
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

				NotifyProgress(L"[Фаза 1] Вычисление эффекта вытаптывания от \"непорядочных пешеходов\"");
				
				pendingTramplingEffect = nodesTramplingEffect;
				nodesTramplingEffect->SetAwaitedPathsNumber(wavefrontTable.numPaths);
				nodesTramplingEffect->ClearSync();				
				wavefrontTable.ResetCompleteness();
				for(auto job : wavefrontJobs) {
					job->Start(&wavefrontTable, edgesDevice, &cudaScheduler);
				}
				nodesTramplingEffect->AwaitAllPaths();

				EdgesTramplingEffect* indecentTrampling = resources.New<EdgesTramplingEffect>(w, h);
				nodesTramplingEffect->SaveAsEdgesSync(indecentTrampling, tramplabilityMask);

				


				NotifyProgress(L"[Фаза 1] Инициализация весов рёбер для \"порядочных пешеходов\"");
				edgesHost->InitFromGraph(graph, false);
				edgesHost->CopyToSync(edgesDevice, w, h);
				
				EdgesWeightsDevice* maximumWeights = resources.New<EdgesWeightsDevice>(w, h);
				edgesDevice->CopyToSync(maximumWeights, w, h);

				nodesTramplingEffect->performanceFactor = DECENT_PEDESTRIANS_SHARE;
				
				constexpr int numThickIterations = 1;
				for(int i = 0; i < numThickIterations; i++) {
					NotifyProgress(String::Format(
						L"[Фаза 1] Симуляция процесса вытаптывания ({0}/{1})",
						i, numThickIterations));

					nodesTramplingEffect->SetAwaitedPathsNumber(wavefrontTable.numPaths);
					nodesTramplingEffect->ClearSync();
					wavefrontTable.ResetCompleteness();
					for(auto job : wavefrontJobs) {
						job->Start(&wavefrontTable, edgesDevice, &cudaScheduler);
					}
					nodesTramplingEffect->AwaitAllPaths();
					ApplyTramplingsAndLawnRegeneration(edgesDevice, w, h, indecentTrampling,
						nodesTramplingEffect->GetDataDevice(), tramplabilityMask, maximumWeights);

					edgesDevice->CopyToSync(edgesHost, w, h);
				}




				pathThickener->thickness = SECOND_PHASE_PATH_THICKNESS;

				NotifyProgress(L"[Фаза 2] Вычисление эффекта вытаптывания от \"непорядочных пешеходов\"");
				nodesTramplingEffect->performanceFactor = INDECENT_PEDESTRIANS_SHARE;
				nodesTramplingEffect->SetAwaitedPathsNumber(wavefrontTable.numPaths);
				nodesTramplingEffect->ClearSync();
				wavefrontTable.ResetCompleteness();
				for(auto job : wavefrontJobs) {
					job->Start(&wavefrontTable, edgesDevice, &cudaScheduler);
				}
				nodesTramplingEffect->AwaitAllPaths();
				nodesTramplingEffect->SaveAsEdgesSync(indecentTrampling, tramplabilityMask);

				nodesTramplingEffect->performanceFactor = DECENT_PEDESTRIANS_SHARE;
				constexpr int numThinIterations = 50;
				for(int i = 0; i < numThinIterations; i++) {
					NotifyProgress(String::Format(
						L"[Фаза 2] Симуляция процесса вытаптывания ({0}/{1})",
						i, numThinIterations));

					nodesTramplingEffect->SetAwaitedPathsNumber(wavefrontTable.numPaths);
					nodesTramplingEffect->ClearSync();
					wavefrontTable.ResetCompleteness();
					for(auto job : wavefrontJobs) {
						job->Start(&wavefrontTable, edgesDevice, &cudaScheduler);
					}
					nodesTramplingEffect->AwaitAllPaths();
					ApplyTramplingsAndLawnRegeneration(edgesDevice, w, h, indecentTrampling,
						nodesTramplingEffect->GetDataDevice(), tramplabilityMask, maximumWeights);

					edgesDevice->CopyToSync(edgesHost, w, h);
				}

				NotifyProgress(L"Выгрузка результата");
				edgesDevice->CopyToSync(edgesHost, w, h);

				auto result = gcnew TrailsComputationsOutput();
				result->Graph = input->Graph;
				ApplyTrampledness(result->Graph, edgesHost);

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
			proxy->NotifyProgress(gcnew String(stage));
		}

		void ComputationThread::NotifyProgress(String^ stage) {
			proxy->NotifyProgress(stage);
		}

		inline void ApplyTramplednessFunc(float& newWeight, Edge^ edge) {
			if(edge != nullptr) {
				edge->Trampledness = edge->Weight - newWeight;
			}
		}

		inline void ComputationThread::ApplyTrampledness(Graph^ graph, EdgesDataHost<float>* edgesData) {
			edgesData->ZipWithGraphEdges(graph, ApplyTramplednessFunc);
		}

	}
}