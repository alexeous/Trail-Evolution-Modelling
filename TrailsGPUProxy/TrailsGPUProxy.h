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
			static initonly float MinimumTramplableWeight = MIN_TRAMPLABLE_WEIGHT;

		public:
			event Action<String^>^ ProgressChanged;
			event Action^ CanGiveUnripeResult;

			property bool GiveUnripeResultFlag;

			TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);

		internal:
			void NotifyProgress(String^ stage);
			void NotifyCanGiveUnripeResult();
		};

	}
}

