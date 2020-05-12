#include "CudaExceptionNative.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		CudaExceptionNative::CudaExceptionNative(const char* message, const char* srcFilename, int line)
			: message(message), srcFilename(srcFilename), line(line)
		{
		}

	}
}