#pragma once
#include "cuda_runtime.h"
#include "IResource.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		struct ExitFlag : public IResource {
			friend class ResourceManager;

			void ResetAsync(cudaStream_t stream);
			void ReadFromDeviceAsync(cudaStream_t stream);
			bool GetLastHostValue() const;

			int* valueDevice;

		protected:
			ExitFlag();
			void Free() override;

		private:
			int* valueHost;
		};

	}
}
