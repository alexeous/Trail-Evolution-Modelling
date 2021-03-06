#include "ComputationThread.h"
#include "CudaExceptionNative.h"
#include "CudaException.h"
#include "IsolatedAttractorsException.h"
#include "CudaScheduler.h"
#include "Attractor.h"
#include "AttractorsMap.h"
#include "EdgesData.h"
#include "TramplabilityMaskHost.h"
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
#include "UpdateIndecentEdges.h"
#include "LambdaUtility.h"
#include "EdgesDeltaCalculator.h"

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
			catch(Exception^ ex) {
				exception = ex;
			}
			catch(...) {
				exception = gcnew Exception("An unknown unmanaged exception occured");
			}
			thread->Abort();
		}

		void ComputationThread::GiveUnripeResultImmediate() {
			giveUnripeResult = true;
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

				NotifyProgress(L"������������ ������ ����� ������� ����������");
				AttractorsMap attractors(graph, input->Attractors);

				NotifyProgress(L"���������� ����� ���������������");
				TramplabilityMaskHost* tramplabilityMaskHost = resources.New<TramplabilityMaskHost>(graph);
				TramplabilityMask* tramplabilityMask = resources.New<TramplabilityMask>(tramplabilityMaskHost, w, h);

				NotifyProgress(L"[���� 1] ������������� ����� ���� ��� \"������������ ���������\"");
				EdgesWeightsHost* edgesHost = resources.New<EdgesWeightsHost>(graph, true);
				EdgesWeightsDevice* edgesDevice = resources.New<EdgesWeightsDevice>(edgesHost, w, h);
				EdgesWeightsDevice* edgesIndecentOriginal = resources.New<EdgesWeightsDevice>(edgesDevice, w, h);
				EdgesWeightsDevice* edgesIndecentPeriodicallyUpdated = resources.New<EdgesWeightsDevice>(w, h);

				NotifyProgress(L"�������� ������������ ��������� ��������� �� GPU");
				std::vector<WavefrontJob*> wavefrontJobs = WavefrontJobsFactory::CreateJobs(w, h, &resources, attractors);
				NodesTramplingEffect* nodesTramplingEffect = resources.New<NodesTramplingEffect>(w, h, stepMeters, INDECENT_PEDESTRIANS_SHARE, SIMULATION_STEP_SECONDS, &resources);
				pendingTramplingEffect = nodesTramplingEffect;
				PathThickener *pathThickener = resources.New<PathThickener>(w, h, stepMeters, FIRST_PHASE_PATH_THICKNESS, tramplabilityMask, nodesTramplingEffect, &resources);
				PathReconstructor *pathReconsturctor = resources.New<PathReconstructor>(w, h, edgesHost, &cudaScheduler, threadPool, &resources, pathThickener, attractors, tramplabilityMaskHost);
				WavefrontCompletenessTable wavefrontTable(attractors, pathReconsturctor);



				NotifyProgress(L"[���� 1] ���������� ������� ������������ �� \"������������ ���������\"");
				DoSimulationStep(INDECENT_PEDESTRIANS_SHARE, nodesTramplingEffect, wavefrontTable, wavefrontJobs, edgesDevice, cudaScheduler);
				EdgesTramplingEffect* indecentTrampling = resources.New<EdgesTramplingEffect>(w, h);
				nodesTramplingEffect->SaveAsEdgesSync(indecentTrampling, tramplabilityMask);

				NotifyProgress(L"[���� 1] ������������ ��������� �����");
				edgesHost->InitFromGraph(graph, MIN_TRAMPLABLE_WEIGHT);
				EdgesWeightsDevice* minimumWeights = resources.New<EdgesWeightsDevice>(edgesHost, w, h);

				NotifyProgress(L"[���� 1] ������������� ����� ���� ��� \"���������� ���������\"");
				edgesHost->InitFromGraph(graph, false);
				edgesHost->CopyToSync(edgesDevice, w, h);
				EdgesWeightsDevice* maximumWeights = resources.New<EdgesWeightsDevice>(edgesDevice, w, h);
				float lastDelta = 0;
				EdgesDeltaCalculator* edgesDeltaCalculator = resources.New<EdgesDeltaCalculator>(w, h, edgesDevice, resources);
				constexpr float epsilonPerEdge = 0.001f*3;
				float epsilon = epsilonPerEdge * graph->TramplableEdgesNumber;
				nodesTramplingEffect->simulationStepSeconds = SIMULATION_STEP_SECONDS;

				try {
					int i = 0;
					do {
						NotifyProgress(String::Format(L"[���� 1] ��������� �������� ������������ {0:0.000}/{1:0.000}", lastDelta, epsilon));

						DoSimulationStep(DECENT_PEDESTRIANS_SHARE, nodesTramplingEffect, wavefrontTable, wavefrontJobs, edgesDevice, cudaScheduler);
						ApplyTramplingsAndLawnRegeneration(edgesDevice, w, h, nodesTramplingEffect->simulationStepSeconds, indecentTrampling, nodesTramplingEffect->GetDataDevice(), tramplabilityMask, minimumWeights, maximumWeights);

						if(i == 0) {
							NotifyCanGiveUnripeResult();
						}

						if(i % 3 == 0) {
							lastDelta = edgesDeltaCalculator->CalculateDelta(edgesDevice);
						}

						if(i % 5 == 0) {
							//NotifyProgress(L"[���� 1] ������������� ���������� ������� ������������ �� \"������������ ���������\"");
							UpdateIndecentEdges(edgesIndecentOriginal, edgesDevice, edgesIndecentPeriodicallyUpdated, w, h);
							DoSimulationStep(INDECENT_PEDESTRIANS_SHARE, nodesTramplingEffect, wavefrontTable, wavefrontJobs, edgesIndecentPeriodicallyUpdated, cudaScheduler);
							nodesTramplingEffect->SaveAsEdgesSync(indecentTrampling, tramplabilityMask);
						}
						i++;
					} while(lastDelta > epsilon);





					pathThickener->thickness = SECOND_PHASE_PATH_THICKNESS;
					nodesTramplingEffect->simulationStepSeconds = SIMULATION_STEP_SECONDS * 10;
					i = 0;
					do {
						NotifyProgress(String::Format(L"[���� 2] ��������� �������� ������������ {0:0.000}/{1:0.000}", lastDelta, epsilon));
						if(i % 5 == 0) {
							//NotifyProgress(L"[���� 2] ������������� ���������� ������� ������������ �� \"������������ ���������\"");
							UpdateIndecentEdges(edgesIndecentOriginal, edgesDevice, edgesIndecentPeriodicallyUpdated, w, h);
							DoSimulationStep(INDECENT_PEDESTRIANS_SHARE, nodesTramplingEffect, wavefrontTable, wavefrontJobs, edgesIndecentPeriodicallyUpdated, cudaScheduler);
							nodesTramplingEffect->SaveAsEdgesSync(indecentTrampling, tramplabilityMask);
						}
						DoSimulationStep(DECENT_PEDESTRIANS_SHARE, nodesTramplingEffect, wavefrontTable, wavefrontJobs, edgesDevice, cudaScheduler);
						ApplyTramplingsAndLawnRegeneration(edgesDevice, w, h, nodesTramplingEffect->simulationStepSeconds, indecentTrampling, nodesTramplingEffect->GetDataDevice(), tramplabilityMask, minimumWeights, maximumWeights);

						if(i % 3 == 0) {
							lastDelta = edgesDeltaCalculator->CalculateDelta(edgesDevice);
						}
						i++;
					} while(lastDelta > epsilon);

					nodesTramplingEffect->simulationStepSeconds = 10.0f / LAWN_REGENERATION_PER_SECOND;
					DoSimulationStep(DECENT_PEDESTRIANS_SHARE, nodesTramplingEffect, wavefrontTable, wavefrontJobs, edgesDevice, cudaScheduler);
					ApplyTramplingsAndLawnRegeneration(edgesDevice, w, h, nodesTramplingEffect->simulationStepSeconds, indecentTrampling, nodesTramplingEffect->GetDataDevice(), tramplabilityMask, minimumWeights, maximumWeights);
				}
				catch(ThreadAbortException^ ex) {
					if(giveUnripeResult) {
						Thread::ResetAbort();
						CancelAll();
					}
					else
						throw ex;
				}

				NotifyProgress(L"�������� ����������");
				edgesDevice->CopyToSync(edgesHost, w, h);

				auto result = gcnew TrailsComputationsOutput();
				result->Graph = input->Graph;
				ApplyTrampledness(result->Graph, edgesHost);

				output = result;
			}
			catch(ThreadAbortException^) {
				Thread::ResetAbort();
			}
			catch(CudaExceptionNative ex) {
				exception = gcnew CudaException(ex);
			}
			catch(std::exception& ex) {
				std::string msg = ex.what();
				exception = gcnew Exception("An unmanaged exception occured: " + gcnew String(msg.c_str()));
			}
			catch(Exception^ ex) {
				exception = ex;
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

		void RunWavefrontJobs(std::vector<WavefrontJob*>& wavefrontJobs,
			WavefrontCompletenessTable* wavefrontTable, EdgesWeightsDevice* edgesDevice,
			CudaScheduler* cudaScheduler)
		{
			using System::Threading::Tasks::Parallel;
			Parallel::For(0, (int)wavefrontJobs.size(), CreateDelegate<Action<int>>([&](int i) {
				wavefrontJobs[i]->Start(wavefrontTable, edgesDevice, cudaScheduler);
			}));
		}

		void ComputationThread::DoSimulationStep(float performanceFactor, NodesTramplingEffect* nodesTramplingEffect, 
			WavefrontCompletenessTable& wavefrontTable, std::vector<WavefrontJob*>& wavefrontJobs, 
			EdgesWeightsDevice* edgesDevice, CudaScheduler& cudaScheduler) 
		{
			nodesTramplingEffect->performanceFactor = performanceFactor;
			nodesTramplingEffect->SetAwaitedPathsNumber(wavefrontTable.numPaths);
			nodesTramplingEffect->ClearSync();
			wavefrontTable.ResetCompleteness();

			
			//for(auto job : wavefrontJobs) {
			//	job->Start(&wavefrontTable, edgesDevice, &cudaScheduler);
			//}
			RunWavefrontJobs(wavefrontJobs, &wavefrontTable, edgesDevice, &cudaScheduler);
			nodesTramplingEffect->AwaitAllPaths();
		}

		void ComputationThread::NotifyProgress(const wchar_t* stage) {
			proxy->NotifyProgress(gcnew String(stage));
		}

		void ComputationThread::NotifyProgress(String^ stage) {
			proxy->NotifyProgress(stage);
		}

		void ComputationThread::NotifyCanGiveUnripeResult() {
			proxy->NotifyCanGiveUnripeResult();
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