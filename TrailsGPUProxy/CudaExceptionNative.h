#pragma once

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		class CudaExceptionNative {
		public:
			CudaExceptionNative(const char* message, const char* srcFilename, int line);

			const char* message;
			const char* srcFilename; 
			const int line;
		};

	}
}
