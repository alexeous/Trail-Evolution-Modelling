#pragma once
#include "Constants.h"
#include "EdgesData.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		public ref class TrailsGPUProxy {
		public:
			static initonly int StepSeconds = SIMULATION_STEP_SECONDS;
			static initonly float MinimumTramplableWeight = MIN_TRAMPLABLE_WEIGHT;

		public:

			event Action<String^>^ ProgressChanged;

			TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);

		internal:
			void NotifyProgress(String^ stage);
		};

	}
}

