#pragma once
#include "cuda_runtime.h"
#include "CudaUtils.h"
#include "IResource.h"
#include "ResourceManager.h"
#include "NodeIndex.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		template<typename T> struct NodesDataHaloed;
		template<typename T> struct NodesDataHaloedHost;
		template<typename T> struct NodesDataHaloedDevice;

		template<typename T>
		struct NodesDataHaloed : public IResource {
			friend class ResourceManager;

			T* data = nullptr;

			inline __host__ __device__ T& At(int i, int j, int graphW);

#ifndef __CUDACC__
		public:
			static int ArraySize(int graphW, int graphH);
			static int ArraySizeBytes(int graphW, int graphH);

		protected:
			virtual void Free(ResourceManager&) = 0;
			static void CudaCopy(const NodesDataHaloed<T>* src, const NodesDataHaloed<T>* dest, int count, cudaMemcpyKind kind, cudaStream_t stream);
#endif
		};

		template<typename T>
		inline __host__ __device__ T& NodesDataHaloed<T>::At(int i, int j, int graphW) {
			return data[i + j * (graphW + 2)];
		}


#ifndef __CUDACC__
		template<typename T>
		struct NodesDataHaloedHost : public NodesDataHaloed<T> {
			friend class ResourceManager;

			const int graphW;
			const int graphH;
			const int extendedW;
			const int extendedH;
			
		public:
			inline T& At(int i, int j);
			inline T& At(NodeIndex index);
			void CopyTo(NodesDataHaloedHost<T>* other, cudaStream_t stream = 0) const;
			void CopyTo(NodesDataHaloedDevice<T>* other, cudaStream_t stream = 0) const;
			void Fill(T value);

		protected:
			NodesDataHaloedHost(int graphW, int graphH);
			void Free(ResourceManager&) override;
		};
#endif
		
		template<typename T>
		struct NodesDataHaloedDevice : public NodesDataHaloed<T> {
#ifndef __CUDACC__
			friend class ResourceManager;

			void CopyTo(NodesDataHaloedHost<T>* other, int graphW, int graphH, cudaStream_t stream = 0) const;
			void CopyTo(NodesDataHaloedDevice<T>* other, int graphW, int graphH, cudaStream_t stream = 0) const;
		protected:
			NodesDataHaloedDevice(int graphW, int graphH);
#endif
		protected:
			void Free(ResourceManager&) override;
		};

#ifndef __CUDACC__
		template<typename T>
		inline int NodesDataHaloed<T>::ArraySize(int graphW, int graphH) {
			return (graphW + 2) * (graphH + 2);
		}

		template<typename T>
		inline int NodesDataHaloed<T>::ArraySizeBytes(int graphW, int graphH) {
			return ArraySize(graphW, graphH) * sizeof(T);
		}

		template<typename T>
		inline void NodesDataHaloed<T>::CudaCopy(const NodesDataHaloed<T>* src, const NodesDataHaloed<T>* dest, int count, cudaMemcpyKind kind, cudaStream_t stream) {
			CHECK_CUDA(cudaMemcpyAsync(dest->data, src->data, count * sizeof(T), kind, stream));
		}



		template<typename T>
		inline T& NodesDataHaloedHost<T>::At(int i, int j) {
			return NodesDataHaloed<T>::At(i, j, graphW);
		}

		template<typename T>
		inline T& NodesDataHaloedHost<T>::At(NodeIndex index) {
			return At(index.i, index.j);
		}

		template<typename T>
		inline void NodesDataHaloedHost<T>::CopyTo(NodesDataHaloedHost<T>* other, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(graphW, graphH), cudaMemcpyHostToHost, stream);
		}

		template<typename T>
		inline void NodesDataHaloedHost<T>::CopyTo(NodesDataHaloedDevice<T>* other, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(graphW, graphH), cudaMemcpyHostToDevice, stream);
		}

		template<typename T>
		inline void NodesDataHaloedHost<T>::Fill(T value) {
			std::fill_n(data, extendedW * extendedH, value);
		}

		template<typename T>
		inline NodesDataHaloedHost<T>::NodesDataHaloedHost(int graphW, int graphH) 
		  : graphW(graphW),
			graphH(graphH),
			extendedW(graphW + 2),
			extendedH(graphH + 2) 
		{
			CHECK_CUDA(cudaMallocHost(&data, ArraySizeBytes(graphW, graphH)));
		}

		template<typename T>
		inline void NodesDataHaloedHost<T>::Free(ResourceManager&) {
			cudaFreeHost(data);
		}



		template<typename T>
		inline void NodesDataHaloedDevice<T>::CopyTo(NodesDataHaloedHost<T>* other, int graphW, int graphH, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(graphW, graphH), cudaMemcpyDeviceToHost, stream);
		}

		template<typename T>
		inline void NodesDataHaloedDevice<T>::CopyTo(NodesDataHaloedDevice<T>* other, int graphW, int graphH, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(graphW, graphH), cudaMemcpyDeviceToDevice, stream);
		}

		template<typename T>
		inline NodesDataHaloedDevice<T>::NodesDataHaloedDevice(int graphW, int graphH) {
			CHECK_CUDA(cudaMalloc(&data, ArraySizeBytes(graphW, graphH)));
		}

		template<typename T>
		inline void NodesDataHaloedDevice<T>::Free(ResourceManager&) {
			cudaFree(data);
		}

#endif

	}
}