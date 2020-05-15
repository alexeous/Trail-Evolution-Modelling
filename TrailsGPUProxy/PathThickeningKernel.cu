#include "PathThickeningKernel.h"

#ifndef __CUDACC__ // visual studio doesn't show functions declared in device_atomic_function.h without this
#define UNDEF__CUDACC__
#define __CUDACC__
#endif
#include <device_atomic_functions.h>
#include <device_functions.h>
#include <math_functions.h>
#ifdef UNDEF__CUDACC__
#undef UNDEF__CUDACC__
#undef __CUDACC__
#endif

#include "NodesDataHaloed.h"
#include "CudaUtils.h"
#include "MathUtils.h"




#define BLOCK_SIZE_X PATH_THICKENING_BLOCK_SIZE_X
#define BLOCK_SIZE_Y PATH_THICKENING_BLOCK_SIZE_Y
#define SHARED_ARRAYS_SIZE_X (BLOCK_SIZE_X + 2)
#define SHARED_ARRAYS_SIZE_Y (BLOCK_SIZE_Y + 2)
#define SHARED_ARRAYS_SIZE (SHARED_ARRAYS_SIZE_X * SHARED_ARRAYS_SIZE_Y)
#define NUM_SHARED_HALO_NODES (2 * (BLOCK_SIZE_X + BLOCK_SIZE_Y + 2))

#if BLOCK_SIZE_X * BLOCK_SIZE_Y < NUM_SHARED_HALO_NODES
#error Too small block size! Not enough nodes within group to load all halo nodes at once!
#endif



namespace TrailEvolutionModelling {
	namespace GPUProxy {

		inline __device__ void ClampGlobalAndThreadIDToNodesBounds(int2& threadId, int2& globalId, int graphW, int graphH) {
			int clampedGlobalX = min(globalId.x, graphW + 1);
			int clampedGlobalY = min(globalId.y, graphH + 1);
			threadId.x -= globalId.x - clampedGlobalX;
			threadId.y -= globalId.y - clampedGlobalY;
			globalId.x = clampedGlobalX;
			globalId.y = clampedGlobalY;
		}

		inline __device__ void LoadNodesToSharedMemory(int i, int j, int graphW, int graphH,
			NodesDataHaloedDevice<float> read, float distanceShared[SHARED_ARRAYS_SIZE_X][SHARED_ARRAYS_SIZE_Y]) {
			int2 threadId;
			threadId.x = threadIdx.x;
			threadId.y = threadIdx.y;
			int2 globalId;
			globalId.x = i + 1;
			globalId.y = j + 1;
			int ftid = threadIdx.x + threadIdx.y * blockDim.x; // flattened thread id

			ClampGlobalAndThreadIDToNodesBounds(threadId, globalId, graphW, graphH);
			distanceShared[threadId.x + 1][threadId.y + 1] = read.At(globalId.x, globalId.y, graphW);
			// X is index of thread - flattenedThreadId - ftid
			// 0 -- BLOCK_SIZE_X+1 => (X; 0)     // top halo row
			// BLOCK_SIZE_X+2 -- 2*BLOCK_SIZE_X+3 => (X - (BLOCK_SIZE_X + 2); BLOCK_SIZE_Y + 1) // bottom halo row
			// else ((X mod 2) * (BLOCK_SIZE_X+1); (X - (2*BLOCK_SIZE_X+4)) / 2)  // alternating left and right halo columns

			if(ftid < NUM_SHARED_HALO_NODES) {
				int haloThreadIdX =
					((ftid <= BLOCK_SIZE_X + 1) * ftid) +
					(((ftid >= BLOCK_SIZE_X + 2) & (ftid <= 2 * BLOCK_SIZE_X + 3)) * (ftid - (BLOCK_SIZE_X + 2))) +
					(((ftid >= 2 * BLOCK_SIZE_X + 4) * (ftid % 2) * (BLOCK_SIZE_X + 1)));

				int haloThreadIdY =
					(((ftid >= BLOCK_SIZE_X + 2) & (ftid <= 2 * BLOCK_SIZE_X + 3)) * (BLOCK_SIZE_Y + 1)) +
					((ftid >= 2 * BLOCK_SIZE_X + 4) * (1 + (ftid - (2 * BLOCK_SIZE_X + 4)) / 2));

				globalId.x += haloThreadIdX - 1 - threadId.x;
				globalId.y += haloThreadIdY - 1 - threadId.y;
				threadId.x = haloThreadIdX;
				threadId.y = haloThreadIdY;
				ClampGlobalAndThreadIDToNodesBounds(threadId, globalId, graphW, graphH);
				distanceShared[threadId.x][threadId.y] = read.At(globalId.x, globalId.y, graphW);
			}
		}

		inline __device__ float GetAt(float distanceShared[SHARED_ARRAYS_SIZE_X][SHARED_ARRAYS_SIZE_Y],
			int threadIdX, int threadIdY) 
		{
			return distanceShared[threadIdX + 1][threadIdY + 1];
		}

		enum EdgeType { Straight, Diagonal };

		template<EdgeType EdgeType>
		inline __device__ void ProcessNeighbour(float& distance, bool isTramplable, float neighbour)
		{
			constexpr float edge = EdgeType == EdgeType::Diagonal
				? 1.41421356f // sqrt(2)
				: 1;
			
			// If isTramplable == true then ((neighbour + edge) / isTramplable) == inf
			// so min will choose 'distance'. This way we skip the non-tramplable edges.
			// Otherwise, ((neighbour + edge) / isTramplable) == (neighbour + edge)
			// and thus min does the real choice between this value and the old distance value
			distance = min(distance, (neighbour + edge) / isTramplable);
		}


		__global__ void PathThickeningKernel(NodesDataHaloedDevice<float> read,
			NodesDataHaloedDevice<float> write,
			int graphW, int graphH, TramplabilityMask tramplabilityMask) 
		{
			__shared__ float distanceShared[SHARED_ARRAYS_SIZE_X][SHARED_ARRAYS_SIZE_Y];

			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;

			LoadNodesToSharedMemory(i, j, graphW, graphH, read, distanceShared);
			__syncthreads();

			if(i < graphW && j < graphH) {
				float distance = GetAt(distanceShared, threadIdx.x, threadIdx.y);
				ProcessNeighbour<EdgeType::Diagonal>(distance, tramplabilityMask.NW(i, j, graphW), GetAt(distanceShared, (int)threadIdx.x - 1, (int)threadIdx.y - 1));
				ProcessNeighbour<EdgeType::Straight>(distance, tramplabilityMask.N (i, j, graphW), GetAt(distanceShared, (int)threadIdx.x,     (int)threadIdx.y - 1));
				ProcessNeighbour<EdgeType::Diagonal>(distance, tramplabilityMask.NE(i, j, graphW), GetAt(distanceShared, (int)threadIdx.x + 1, (int)threadIdx.y - 1));
				ProcessNeighbour<EdgeType::Straight>(distance, tramplabilityMask.W (i, j, graphW), GetAt(distanceShared, (int)threadIdx.x - 1, (int)threadIdx.y));
				ProcessNeighbour<EdgeType::Straight>(distance, tramplabilityMask.E (i, j, graphW), GetAt(distanceShared, (int)threadIdx.x + 1, (int)threadIdx.y));
				ProcessNeighbour<EdgeType::Diagonal>(distance, tramplabilityMask.SW(i, j, graphW), GetAt(distanceShared, (int)threadIdx.x - 1, (int)threadIdx.y + 1));
				ProcessNeighbour<EdgeType::Straight>(distance, tramplabilityMask.S (i, j, graphW), GetAt(distanceShared, (int)threadIdx.x,     (int)threadIdx.y + 1));
				ProcessNeighbour<EdgeType::Diagonal>(distance, tramplabilityMask.SE(i, j, graphW), GetAt(distanceShared, (int)threadIdx.x + 1, (int)threadIdx.y + 1));
				write.At(i + 1, j + 1, graphW) = distance;
			}
		}

		template<bool SwapNodesPair>
		cudaError_t PathThickening(NodesDataDevicePair<float>* distanceToPath, int graphW, int graphH,
			TramplabilityMask* tramplabilityMask, cudaStream_t stream)
		{
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetPathThickeningBlocksX(graphW), GetPathThickeningBlocksY(graphH));

			if constexpr(!SwapNodesPair) {
				PathThickeningKernel<<<blocksDim, threadsDim, 0, stream>>>
					(*distanceToPath->readOnly, *distanceToPath->writeOnly, graphW, graphH, *tramplabilityMask);
			}
			else {
				PathThickeningKernel<<<blocksDim, threadsDim, 0, stream>>>
					(*distanceToPath->writeOnly, *distanceToPath->readOnly, graphW, graphH, *tramplabilityMask);
			}

			return cudaGetLastError();
		}

	}
}