#include "IsolatedAttractorsException.h"
namespace TrailEvolutionModelling {
	namespace GPUProxy {

		IsolatedAttractorsException::IsolatedAttractorsException(List<RefAttractor^>^ IsolatedAttractors) {
			this->IsolatedAttractors = IsolatedAttractors;
		}
	}
}