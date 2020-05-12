#include "CudaExceptionNative.h"
#include <string>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		CudaExceptionNative::CudaExceptionNative(const char* message, const char* srcFilename, int line)
			: std::exception(
				(std::string() + "Error '" + message + "' at line " + 
					std::to_string(line) + " in file " + srcFilename).c_str()
			),
			message(message), srcFilename(srcFilename), line(line)
		{
		}

	}
}