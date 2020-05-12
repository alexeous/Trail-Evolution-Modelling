#pragma once
#include "CudaExceptionNative.h"

using namespace System;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		ref class CudaException : Exception {
		public:
			CudaException(const CudaExceptionNative& nativeEx);

		private:
			static String^ CombineMessage(const char* message, const char* srcFilename, int line);
		};
	}
}