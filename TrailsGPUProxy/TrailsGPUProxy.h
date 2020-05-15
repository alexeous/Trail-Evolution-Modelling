#pragma once
#include "EdgesData.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class TrailsGPUProxy {
		public:
			static initonly int StepSeconds = 5 * 60;
			static initonly float MinimumTramplableWeight = 1.1f;

			static constexpr float FirstPhasePathThickness = 5;
			static constexpr float SecondPhasePathThickness = 1.5f;

			event Action<String^>^ ProgressChanged;

			TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);

		internal:
			void NotifyProgress(const wchar_t* stage);
		};
	}
}

