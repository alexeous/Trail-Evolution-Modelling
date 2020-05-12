#pragma once
#include <exception>

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class CudaExceptionNative : public std::exception {
		public:
			CudaExceptionNative(const char* message, const char* srcFilename, int line);

			const char* message;
			const char* srcFilename; 
			const int line;
		};

	}
}
