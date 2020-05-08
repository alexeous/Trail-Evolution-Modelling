#pragma once

using namespace System;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ref class CudaException : Exception {
		public:
			CudaException(const char* message, const char* srcFilename, int line);

		private:
			static String^ CombineMessage(const char* message, const char* srcFilename, int line);
		};
	}
}