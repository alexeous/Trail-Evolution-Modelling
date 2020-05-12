#include "CudaException.h"

using namespace System;

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		CudaException::CudaException(const CudaExceptionNative& ex)
			: Exception(CombineMessage(ex.message, ex.srcFilename, ex.line)) {
		}

		String^ CudaException::CombineMessage(const char* message, const char* srcFilename, int line) {
			String^ managedMessage = gcnew String(message);
			String^ managedSrcFilename = gcnew String(srcFilename);
			return String::Format("Error '{0}' at line {1} in file {2}",
				managedMessage, managedSrcFilename, line);
		}
	}
}