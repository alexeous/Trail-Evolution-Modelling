#pragma once
#include <functional>
#include <exception>
#include <string>
#include "cuda_runtime.h"
#include "CudaUtils.h"
#include "IResource.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

#ifndef __CUDACC__
		using namespace TrailEvolutionModelling::GraphTypes;
#endif
		template<typename T> struct EdgesData;
		template<typename T> struct EdgesDataHost;
		template<typename T> struct EdgesDataDevice;

		template<typename T>
		struct EdgesData : public IResource {
			friend class ResourceManager;

			T* vertical = nullptr;
			T* horizontal = nullptr;
			T* leftDiagonal = nullptr;
			T* rightDiagonal = nullptr;

			inline __host__ __device__ T& NW(int i, int j, int w);
			inline __host__ __device__ T& N(int i, int j, int w);
			inline __host__ __device__ T& NE(int i, int j, int w);
			inline __host__ __device__ T& W(int i, int j, int w);
			inline __host__ __device__ T& E(int i, int j, int w);
			inline __host__ __device__ T& SW(int i, int j, int w);
			inline __host__ __device__ T& S(int i, int j, int w);
			inline __host__ __device__ T& SE(int i, int j, int w);

#ifndef __CUDACC__
			inline T& AtDir(int i, int j, int w, int dir);
			inline T& AtShift(int i, int j, int w, int shiftI, int shiftJ);

			template<typename... TArgs>
			inline void ZipWithGraphEdges(Graph^ graph, void (*func)(T&, Edge^, TArgs...), TArgs... args) {
				int w = graph->Width;
				int h = graph->Height;
				for(int i = 0; i < w; i++) {
					for(int j = 0; j < h; j++) {
						bool notLastColumn = i < w - 1;
						bool notLastRow = j < h - 1;
						bool notFirstColumn = i != 0;

						Node^ node = graph->GetNodeAtOrNull(i, j);

						func(E(i, j, w), GetEdge(node, Direction::E), args...);
						if(notLastRow) {
							func(S(i, j, w), GetEdge(node, Direction::S), args...);
							if(notLastColumn)
								func(SE(i, j, w), GetEdge(node, Direction::SE), args...);
							if(notFirstColumn)
								func(SW(i, j, w), GetEdge(node, Direction::SW), args...);
						}
					}
				}
			}

		protected:
			virtual void Free(ResourceManager&) = 0;
			static void CudaCopySync(const EdgesData<T>* src, const EdgesData<T>* dest, int count, cudaMemcpyKind kind);
			static void CudaCopy(const EdgesData<T>* src, const EdgesData<T>* dest, int count, cudaMemcpyKind kind, cudaStream_t stream);
			static int ArraySize(int w, int h);
			static int ArraySizeBytes(int w, int h);
			static Edge^ GetEdge(Node^ node, Direction direction);
#endif
		};

#ifndef __CUDACC__
		template<typename T>
		struct EdgesDataHost : public EdgesData<T> {
			friend class ResourceManager;

			void CopyToSync(const EdgesDataHost<T>* other, int w, int h) const;
			void CopyToSync(const EdgesDataDevice<T>* other, int w, int h) const;
			void CopyTo(const EdgesDataHost<T>* other, int w, int h, cudaStream_t stream) const;
			void CopyTo(const EdgesDataDevice<T>* other, int w, int h, cudaStream_t stream) const;
		protected:
			EdgesDataHost(int w, int h);
			EdgesDataHost(const EdgesDataDevice<T>* device, int w, int h);
			void Free(ResourceManager&) override;
		};
#endif

		template<typename T>
		struct EdgesDataDevice : public EdgesData<T> {
			friend class ResourceManager;

			void CopyToSync(const EdgesDataHost<T>* other, int w, int h) const;
			void CopyToSync(const EdgesDataDevice<T>* other, int w, int h) const;
			void CopyTo(const EdgesDataHost<T>* other, int w, int h, cudaStream_t stream) const;
			void CopyTo(const EdgesDataDevice<T>* other, int w, int h, cudaStream_t stream) const;
		protected:
			EdgesDataDevice(int w, int h);
			EdgesDataDevice(const EdgesDataHost<T>* host, int w, int h);
			EdgesDataDevice(const EdgesDataDevice<T>* device, int w, int h);
			void Free(ResourceManager&) override;
		};




		template<typename T> inline __host__ __device__ T& EdgesData<T>::NW(int i, int j, int w) { return leftDiagonal[i + j * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::N(int i, int j, int w) { return vertical[i + 1 + j * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::NE(int i, int j, int w) { return rightDiagonal[i + 1 + j * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::W(int i, int j, int w) { return horizontal[i + (j + 1) * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::E(int i, int j, int w) { return horizontal[i + 1 + (j + 1) * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::SW(int i, int j, int w) { return rightDiagonal[i + (j + 1) * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::S(int i, int j, int w) { return vertical[i + 1 + (j + 1) * (w + 1)]; }
		template<typename T> inline __host__ __device__ T& EdgesData<T>::SE(int i, int j, int w) { return leftDiagonal[i + 1 + (j + 1) * (w + 1)]; }


#ifndef __CUDACC__
		template<typename T>
		inline __host__ T& EdgesData<T>::AtDir(int i, int j, int w, int dir) {
			switch(dir) {
				case 0: return NW(i, j, w);
				case 1: return N (i, j, w);
				case 2: return NE(i, j, w);
				case 3: return W (i, j, w);
				case 4: return E (i, j, w);
				case 5: return SW(i, j, w);
				case 6: return S (i, j, w);
				case 7: return SE(i, j, w);
				default:
					throw std::exception(("Invalid direction: " + std::to_string(dir)).c_str());
			}
		}

		template<typename T>
		inline T& EdgesData<T>::AtShift(int i, int j, int w, int shiftI, int shiftJ) {
			int dir = ComputeNode::ShiftToDirection(NodeIndex(shiftI, shiftJ));
			return AtDir(i, j, w, dir);
		}

		//template<typename T, typename... TArgs>
		//inline void EdgesData<T>::ZipWithGraphEdges(Graph^ graph, void (*func)(T&, Edge^, TArgs...))
		//{
		//	
		//}

		template<typename T>
		inline Edge^ EdgesData<T>::GetEdge(Node^ node, Direction direction) {
			return (node == nullptr ? nullptr : node->GetIncidentEdge(direction));
		}

		template<typename T>
		inline void EdgesData<T>::CudaCopy(const EdgesData<T>* src, const EdgesData<T>* dest, 
			int count, cudaMemcpyKind kind, cudaStream_t stream) {
			size_t size = count * sizeof(T);
			CHECK_CUDA(cudaMemcpyAsync(dest->vertical, src->vertical, size, kind, stream));
			CHECK_CUDA(cudaMemcpyAsync(dest->horizontal, src->horizontal, size, kind, stream));
			CHECK_CUDA(cudaMemcpyAsync(dest->leftDiagonal, src->leftDiagonal, size, kind, stream));
			CHECK_CUDA(cudaMemcpyAsync(dest->rightDiagonal, src->rightDiagonal, size, kind, stream));
		}

		template<typename T>
		inline void EdgesData<T>::CudaCopySync(const EdgesData<T>* src, const EdgesData<T>* dest,
			int count, cudaMemcpyKind kind) {
			size_t size = count * sizeof(T);
			CHECK_CUDA(cudaMemcpy(dest->vertical, src->vertical, size, kind));
			CHECK_CUDA(cudaMemcpy(dest->horizontal, src->horizontal, size, kind));
			CHECK_CUDA(cudaMemcpy(dest->leftDiagonal, src->leftDiagonal, size, kind));
			CHECK_CUDA(cudaMemcpy(dest->rightDiagonal, src->rightDiagonal, size, kind));
		}

		template<typename T>
		inline int EdgesData<T>::ArraySize(int w, int h) {
			return (w + 1) * (h + 1);
		}

		template<typename T>
		inline int EdgesData<T>::ArraySizeBytes(int w, int h) {
			return ArraySize(w, h) * sizeof(T);
		}



		template<typename T>
		inline void EdgesDataHost<T>::CopyToSync(const EdgesDataHost<T>* other, int w, int h) const {
			CudaCopySync(this, other, ArraySize(w, h), cudaMemcpyHostToHost);
		}

		template<typename T>
		inline void EdgesDataHost<T>::CopyToSync(const EdgesDataDevice<T>* other, int w, int h) const {
			CudaCopySync(this, other, ArraySize(w, h), cudaMemcpyHostToDevice);
		}

		template<typename T>
		inline void EdgesDataHost<T>::CopyTo(const EdgesDataHost<T>* other, int w, int h, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(w, h), cudaMemcpyHostToHost, stream);
		}

		template<typename T>
		inline void EdgesDataHost<T>::CopyTo(const EdgesDataDevice<T>* other, int w, int h, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(w, h), cudaMemcpyHostToDevice, stream);
		}

		template<typename T>
		EdgesDataHost<T>::EdgesDataHost(int w, int h) {
			size_t size = ArraySizeBytes(w, h);
			CHECK_CUDA(cudaMallocHost(&vertical, size));
			CHECK_CUDA(cudaMallocHost(&horizontal, size));
			CHECK_CUDA(cudaMallocHost(&leftDiagonal, size));
			CHECK_CUDA(cudaMallocHost(&rightDiagonal, size));
		}

		template<typename T>
		EdgesDataHost<T>::EdgesDataHost(const EdgesDataDevice<T>* device, int w, int h)
			: EdgesDataHost(w, h) {
			device->CopyToSync(this, w, h);
		}

		template<typename T>
		void EdgesDataHost<T>::Free(ResourceManager&) {
			cudaFreeHost(vertical);
			cudaFreeHost(horizontal);
			cudaFreeHost(leftDiagonal);
			cudaFreeHost(rightDiagonal);
		}




		template<typename T>
		EdgesDataDevice<T>::EdgesDataDevice(int w, int h) {
			size_t size = ArraySizeBytes(w, h);
			CHECK_CUDA(cudaMalloc(&vertical, size));
			CHECK_CUDA(cudaMalloc(&horizontal, size));
			CHECK_CUDA(cudaMalloc(&leftDiagonal, size));
			CHECK_CUDA(cudaMalloc(&rightDiagonal, size));
		}

		template<typename T>
		inline void EdgesDataDevice<T>::CopyToSync(const EdgesDataHost<T>* other, int w, int h) const {
			CudaCopySync(this, other, ArraySize(w, h), cudaMemcpyDeviceToHost);
		}

		template<typename T>
		inline void EdgesDataDevice<T>::CopyToSync(const EdgesDataDevice<T>* other, int w, int h) const {
			CudaCopySync(this, other, ArraySize(w, h), cudaMemcpyDeviceToDevice);
		}

		template<typename T>
		inline void EdgesDataDevice<T>::CopyTo(const EdgesDataHost<T>* other, int w, int h, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(w, h), cudaMemcpyDeviceToHost, stream);
		}

		template<typename T>
		inline void EdgesDataDevice<T>::CopyTo(const EdgesDataDevice<T>* other, int w, int h, cudaStream_t stream) const {
			CudaCopy(this, other, ArraySize(w, h), cudaMemcpyDeviceToDevice, stream);
		}

		template<typename T>
		EdgesDataDevice<T>::EdgesDataDevice(const EdgesDataHost<T>* host, int w, int h) :
			EdgesDataDevice(w, h)
		{
			host->CopyToSync(this, w, h);
		}

		template<typename T>
		inline EdgesDataDevice<T>::EdgesDataDevice(const EdgesDataDevice<T>* device, int w, int h) :
			EdgesDataDevice(w, h) 
		{
			device->CopyToSync(this, w, h);
		}

		template<typename T>
		void EdgesDataDevice<T>::Free(ResourceManager&) {
			cudaFree(vertical);
			cudaFree(horizontal);
			cudaFree(leftDiagonal);
			cudaFree(rightDiagonal);
		}

#endif

	}
}