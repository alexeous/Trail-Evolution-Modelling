#include "TrailsGPUProxy.h"
#include "IsolatedAttractorsException.h"
#include "Attractor.h"
#include "AttractorsMap.h"
#include "EdgesData.h"
#include "TramplabilityMask.h"
#include "EdgesWeights.h"
#include "WavefrontJob.h"
#include "WavefrontJobsFactory.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		TrailsComputationsOutput^ TrailsGPUProxy::ComputeTrails(TrailsComputationsInput^ input) {
			TrailsComputationsOutput^ result = nullptr;

			ResourceManager resources;
			Graph^ graph = input->Graph;
			try {
				NotifyProgress(L"Установление связей между точками притяжения");
				AttractorsMap attractors(graph, input->Attractors);
				
				NotifyProgress(L"Построение маски вытаптываемости");
				TramplabilityMask* tramplabilityMask = resources.New<TramplabilityMask>(graph);

				NotifyProgress(L"Инициализация весов рёбер для \"непорядочных пешеходов\"");
				EdgesWeights* indecentEdgesWeights = resources.New<EdgesWeights>(graph, resources, true);

				NotifyProgress(L"Создание исполнителей волнового алгоритма на GPU");
				std::vector<WavefrontJob*> wavefrontJobs =
					WavefrontJobsFactory::CreateJobs(graph->Width, graph->Height, resources, attractors);
				
				for(auto job : wavefrontJobs) {
					job->ResetReadOnlyNodesGParallel();
				}
				WaitForGPU();

				NotifyProgress(L"Симуляция движения пешеходов");

				NotifyProgress(L"Выгрузка результата");
				EdgesDataHost<float>* trampledness = resources.New<EdgesDataHost<float>>(indecentEdgesWeights, 
					input->Graph->Width, input->Graph->Height);

				resources.Free(indecentEdgesWeights);

				result = gcnew TrailsComputationsOutput();
				result->Graph = input->Graph;
				ApplyTrampledness(result->Graph, trampledness);
			}
			finally {
				resources.FreeAll();
			}

			return result;
		}

		void TrailsGPUProxy::NotifyProgress(const wchar_t* stage) {
			ProgressChanged(gcnew String(stage));
		}
	}
}