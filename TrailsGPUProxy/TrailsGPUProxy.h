#pragma once

using namespace System;
using namespace System::Collections::Generic;
using namespace TrailEvolutionModelling::GraphTypes;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class TrailsGPUProxy {
		public:
			static initonly int StepSeconds = 5 * 60;

			event Action<String^>^ ProgressChanged;

			TrailsComputationsOutput^ ComputeTrails(TrailsComputationsInput^ input);

		private:
			void NotifyProgress(const wchar_t* stage);
		//private:
		//	Dictionary<Attractor^, List<Attractor^>^>^ CreateAttractorsMap(TrailsComputationsInput^ input);
		//	bool CanReach(Graph^ graph, Attractor^ a, Attractor^ b);
		};
	}
}

