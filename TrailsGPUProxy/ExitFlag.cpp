#include "ExitFlag.h"
#include "CudaUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		ExitFlag::ExitFlag() {
			CHECK_CUDA(cudaMallocHost(&valueHost, sizeof(int)));
			CHECK_CUDA(cudaMalloc(&valueDevice, sizeof(int)));
		}

		void ExitFlag::ResetAsync(cudaStream_t stream) {
			*valueHost = false;
			CHECK_CUDA(cudaMemcpyAsync(valueDevice, valueHost, sizeof(int), cudaMemcpyHostToDevice, stream));
		}

		void ExitFlag::ReadFromDeviceAsync(cudaStream_t stream) {
			CHECK_CUDA(cudaMemcpyAsync(valueHost, valueDevice, sizeof(int), cudaMemcpyDeviceToHost, stream));
		}

		bool ExitFlag::GetLastHostValue() const {
			return *valueHost;
		}

		void ExitFlag::Free() {
			cudaFreeHost(valueHost);
			cudaFree(valueDevice);
		}

	}
}