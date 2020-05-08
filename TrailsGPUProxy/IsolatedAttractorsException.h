#pragma once
using namespace System;
using namespace System::Collections::Generic;

using RefAttractor = TrailEvolutionModelling::GraphTypes::Attractor;

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		public ref class IsolatedAttractorsException : public Exception {
		public:
			property List<RefAttractor^>^ IsolatedAttractors;

			IsolatedAttractorsException(List<RefAttractor^>^ IsolatedAttractors);
		};
	}
}