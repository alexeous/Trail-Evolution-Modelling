#pragma once
#include <functional>
#include "cuda_runtime.h"
#include "CudaUtils.h"
#include "IResource.h"
#include "ResourceManager.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {
		using namespace TrailEvolutionModelling::GraphTypes;

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

			inline T& NW(int i, int j, int w);
			inline T& N(int i, int j, int w);
			inline T& NE(int i, int j, int w);
			inline T& W(int i, int j, int w);
			inline T& E(int i, int j, int w);
			inline T& SW(int i, int j, int w);
			inline T& S(int i, int j, int w);
			inline T& SE(int i, int j, int w);
			
			void ZipWithGraphEdges(Graph^ graph, int iOffset, int jOffset, void (*func)(T&, Edge^));

		protected:
			virtual void Free() = 0;
			static void CudaCopy(const EdgesData<T>& src, const EdgesData<T>& dest, int size, cudaMemcpyKind kind);
			static int ArraySize(int w, int h);
			static Edge^ GetEdge(Node^ node, Direction direction);
		};

		template<typename T>
		struct EdgesDataHost : public EdgesData<T> {
			friend class ResourceManager;
			
			void CopyTo(const EdgesDataHost<T>& other, int w, int h);
			void CopyTo(const EdgesDataDevice<T>& other, int w, int h);
		protected:
			EdgesDataHost(int w, int h);
			EdgesDataHost(const EdgesDataDevice<T>& device, int w, int h);
			void Free() override;
		};

		template<typename T>
		struct EdgesDataDevice : public EdgesData<T> {
			friend class ResourceManager;
			
			void CopyTo(const EdgesDataHost<T>& other, int w, int h);
			void CopyTo(const EdgesDataDevice<T>& other, int w, int h);
		protected:
			EdgesDataDevice(int w, int h);
			EdgesDataDevice(const EdgesDataHost<T>& host, int w, int h);
			void Free() override;
		};




		template<typename T> inline T& EdgesData<T>::NW(int i, int j, int w) { return leftDiagonal[i + j * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::N(int i, int j, int w) { return vertical[i + 1 + j * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::NE(int i, int j, int w) { return rightDiagonal[i + 1 + j * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::W(int i, int j, int w) { return horizontal[i + (j + 1) * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::E(int i, int j, int w) { return horizontal[i + 1 + (j + 1) * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::SW(int i, int j, int w) { return rightDiagonal[i + (j + 1) * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::S(int i, int j, int w) { return vertical[i + 1 + (j + 1) * (w + 1)]; }
		template<typename T> inline T& EdgesData<T>::SE(int i, int j, int w) { return leftDiagonal[i + 1 + (j + 1) * (w + 1)]; }

		template<typename T>
		inline void EdgesData<T>::ZipWithGraphEdges(Graph^ graph, int iOffset, int jOffset, 
			void (*func)(T&, Edge^))
		{
			int w = graph->Width;
			int h = graph->Height;
			for(int i = 0; i < w; i++) {
				for(int j = 0; j < h; j++) {
					bool notLastColumn = i < w - 1;
					bool notLastRow = j < h - 1;
					bool notFirstColumn = i != 0;

					Node^ node = graph->GetNodeAtOrNull(i, j);

					func(E(i, j, w), GetEdge(node, Direction::E));
					if(notLastRow) {
						func(S(i, j, w), GetEdge(node, Direction::S));
						if(notLastColumn)
							func(SE(i, j, w), GetEdge(node, Direction::SE));
						if(notFirstColumn)
							func(SW(i, j, w), GetEdge(node, Direction::SW));
					}
				}
			}
		}

		template<typename T>
		inline Edge^ EdgesData<T>::GetEdge(Node^ node, Direction direction) {
			return (node == nullptr ? nullptr : node->GetIncidentEdge(direction));
		}

		template<typename T>
		inline void EdgesData<T>::CudaCopy(const EdgesData<T>& src, const EdgesData<T>& dest, 
			int size, cudaMemcpyKind kind) {
			CHECK_CUDA(cudaMemcpy(src.vertical, dest.vertical, size, kind));
			CHECK_CUDA(cudaMemcpy(src.horizontal, dest.horizontal, size, kind));
			CHECK_CUDA(cudaMemcpy(src.leftDiagonal, dest.leftDiagonal, size, kind));
			CHECK_CUDA(cudaMemcpy(src.rightDiagonal, dest.rightDiagonal, size, kind));
		}

		template<typename T>
		inline int EdgesData<T>::ArraySize(int w, int h) {
			return (w + 1) * (h + 1);
		}




		template<typename T>
		inline void EdgesDataHost<T>::CopyTo(const EdgesDataHost<T>& other, int w, int h) {
			CudaCopy(*this, other, ArraySize(w, h), cudaMemcpyHostToHost);
		}

		template<typename T>
		inline void EdgesDataHost<T>::CopyTo(const EdgesDataDevice<T>& other, int w, int h) {
			CudaCopy(*this, other, ArraySize(w, h), cudaMemcpyHostToDevice);
		}

		template<typename T>
		EdgesDataHost<T>::EdgesDataHost(int w, int h) {
			size_t size = ArraySize(w, h);
			CHECK_CUDA(cudaMallocHost((void**)&vertical, size));
			CHECK_CUDA(cudaMallocHost((void**)&horizontal, size));
			CHECK_CUDA(cudaMallocHost((void**)&leftDiagonal, size));
			CHECK_CUDA(cudaMallocHost((void**)&rightDiagonal, size));
		}

		template<typename T>
		EdgesDataHost<T>::EdgesDataHost(const EdgesDataDevice<T>& device, int w, int h)
			: EdgesDataHost(w, h) 
		{
			device.CopyTo(*this, w, h);
		}

		template<typename T>
		void EdgesDataHost<T>::Free() {
			cudaFreeHost(vertical);
			cudaFreeHost(horizontal);
			cudaFreeHost(leftDiagonal);
			cudaFreeHost(rightDiagonal);
		}




		template<typename T>
		EdgesDataDevice<T>::EdgesDataDevice(int w, int h) {
			size_t size = ArraySize(w, h);
			CHECK_CUDA(cudaMalloc((void**)&vertical, size));
			CHECK_CUDA(cudaMalloc((void**)&horizontal, size));
			CHECK_CUDA(cudaMalloc((void**)&leftDiagonal, size));
			CHECK_CUDA(cudaMalloc((void**)&rightDiagonal, size));
		}

		template<typename T>
		inline void EdgesDataDevice<T>::CopyTo(const EdgesDataHost<T>& other, int w, int h) {
			CudaCopy(*this, other, ArraySize(w, h), cudaMemcpyDeviceToHost);
		}

		template<typename T>
		inline void EdgesDataDevice<T>::CopyTo(const EdgesDataDevice<T>& other, int w, int h) {
			CudaCopy(*this, other, ArraySize(w, h), cudaMemcpyDeviceToDevice);
		}

		template<typename T>
		EdgesDataDevice<T>::EdgesDataDevice(const EdgesDataHost<T>& host, int w, int h) :
			EdgesDataDevice(w, h)
		{
			host.CopyTo(*this, w, h);
		}

		template<typename T>
		void EdgesDataDevice<T>::Free() {
			cudaFree(vertical);
			cudaFree(horizontal);
			cudaFree(leftDiagonal);
			cudaFree(rightDiagonal);
		}

	}
}