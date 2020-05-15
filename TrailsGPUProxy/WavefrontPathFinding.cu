#include "WavefrontPathFinding.h"

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




#define BLOCK_SIZE_X WAVEFRONT_PATH_FINDING_BLOCK_SIZE_X
#define BLOCK_SIZE_Y WAVEFRONT_PATH_FINDING_BLOCK_SIZE_Y
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
			NodesDataHaloedDevice<ComputeNode> read, ComputeNode nodesShared[SHARED_ARRAYS_SIZE_X][SHARED_ARRAYS_SIZE_Y])
		{
			int2 threadId;
			threadId.x = threadIdx.x;
			threadId.y = threadIdx.y;
			int2 globalId;
			globalId.x = i + 1;
			globalId.y = j + 1;
			int ftid = threadIdx.x + threadIdx.y * blockDim.x; // flattened thread id

			ClampGlobalAndThreadIDToNodesBounds(threadId, globalId, graphW, graphH);
			nodesShared[threadId.x + 1][threadId.y + 1] = read.At(globalId.x, globalId.y, graphW);
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
				nodesShared[threadId.x][threadId.y] = read.At(globalId.x, globalId.y, graphW);
			}
		}
		
		inline __device__ void CalcMaxAgentsG(float* maxAgentsGPerGroup, unsigned int& maxAgentsGSharedAsUint) {
			maxAgentsGSharedAsUint = __float_as_uint(0);
			__syncthreads();

			int ftid = threadIdx.x + threadIdx.y * blockDim.x; // flattened thread id
			if(ftid < gridDim.x * gridDim.y) {
				atomicMax(&maxAgentsGSharedAsUint, __float_as_uint(maxAgentsGPerGroup[ftid]));
			}
			__syncthreads();
		}

		inline __device__ ComputeNode GetNode(ComputeNode nodesShared[SHARED_ARRAYS_SIZE_X][SHARED_ARRAYS_SIZE_Y], 
			int threadIdX, int threadIdY) 
		{
			return nodesShared[threadIdX + 1][threadIdY + 1]; 
		}

		inline __device__ int ReplaceIfXLessThanY(int value, int replacement, float x, float y) {
			return (!(x < y)) * value + (x < y) * replacement;
		}

		enum EdgeType { Straight, Diagonal };

		template<bool SetRepeatFlag, EdgeType EdgeType>
		inline __device__ void ProcessNeighbour(float edge, ComputeNode& node,
			const ComputeNode& neighbour, float maxAgentsGShared, int dir, bool& repeat)
		{
			float oldG = node.g;
			if constexpr(EdgeType == EdgeType::Diagonal) {
				edge *= 1.41421356f; // sqrt(2)
			}
			float tentativeNewG = neighbour.g + edge;
			node.SetDirNext(ReplaceIfXLessThanY(node.GetDirNext(), dir, tentativeNewG, node.g));
			node.g = min(node.g, tentativeNewG);
			
			if constexpr(SetRepeatFlag) {
				repeat = repeat | ((node.g != oldG) & (node.g < maxAgentsGShared));
			}
		}

		template<bool SetExitFlag>
		__global__ void WavefrontPathFindingKernel(NodesDataHaloedDevice<ComputeNode> read, 
			NodesDataHaloedDevice<ComputeNode> write,
			int graphW, int graphH, EdgesWeightsDevice edges, int goalIndex, 
			float* maxAgentsGPerGroup, int* exitFlag)
		{
			__shared__ float maxAgentsGShared;
			__shared__ unsigned int maxAgentsGSharedAsUint;
			__shared__ int repeatShared;
			__shared__ unsigned int newMaxAgentsGSharedAsUint;
			__shared__ ComputeNode nodesShared[SHARED_ARRAYS_SIZE_X][SHARED_ARRAYS_SIZE_Y];

			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			
			LoadNodesToSharedMemory(i, j, graphW, graphH, read, nodesShared);
			CalcMaxAgentsG(maxAgentsGPerGroup, maxAgentsGSharedAsUint);
			if((threadIdx.x == 0) & (threadIdx.y == 0)) {
				maxAgentsGShared = __uint_as_float(maxAgentsGSharedAsUint);
				newMaxAgentsGSharedAsUint = __float_as_uint(0.0);
				repeatShared = false;
			}
			__syncthreads();

			if(i < graphW && j < graphH) {
				ComputeNode node = GetNode(nodesShared, threadIdx.x, threadIdx.y);
				bool repeat = false;
				ProcessNeighbour<SetExitFlag, EdgeType::Diagonal>(edges.NW(i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x - 1,	(int)threadIdx.y - 1),	maxAgentsGShared, 0, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Straight>(edges.N (i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x,		(int)threadIdx.y - 1),	maxAgentsGShared, 1, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Diagonal>(edges.NE(i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x + 1,	(int)threadIdx.y - 1),	maxAgentsGShared, 2, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Straight>(edges.W (i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x - 1,	(int)threadIdx.y),		maxAgentsGShared, 3, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Straight>(edges.E (i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x + 1,	(int)threadIdx.y),		maxAgentsGShared, 4, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Diagonal>(edges.SW(i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x - 1,	(int)threadIdx.y + 1),	maxAgentsGShared, 5, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Straight>(edges.S (i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x,		(int)threadIdx.y + 1),	maxAgentsGShared, 6, repeat);
				ProcessNeighbour<SetExitFlag, EdgeType::Diagonal>(edges.SE(i, j, graphW), node, GetNode(nodesShared, (int)threadIdx.x + 1,	(int)threadIdx.y + 1),	maxAgentsGShared, 7, repeat);
				write.At(i + 1, j + 1, graphW) = node;

				if(node.IsStart()) {
					atomicMax(&newMaxAgentsGSharedAsUint, __float_as_uint(node.g));
				}
				if constexpr(SetExitFlag) {
					bool isUninitializedStart = node.IsStart() & isinf(node.g);
					atomicOr(&repeatShared, repeat | isUninitializedStart);
				}
			}
			__syncthreads();

			if((threadIdx.x == 0) & (threadIdx.y == 0)) {
				maxAgentsGPerGroup[blockIdx.x + blockIdx.y * gridDim.x] = __uint_as_float(newMaxAgentsGSharedAsUint);
				if constexpr(SetExitFlag) {
					atomicAnd(exitFlag, !repeatShared);
				}
			}
		}

		template<bool SwapNodesPair, bool SetExitFlag>
		cudaError_t WavefrontPathFinding(ComputeNodesPair* nodes, int graphW, int graphH, EdgesWeightsDevice* edgesWeights,
			int goalIndex, float* maxAgentsGPerGroup, ExitFlag* exitFlag, cudaStream_t stream)
		{
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetWavefrontPathFindingBlocksX(graphW), GetWavefrontPathFindingBlocksY(graphH));

			int* exitFlagDevice = nullptr;
			if constexpr(SetExitFlag) {
				exitFlagDevice = exitFlag->valueDevice;
			}

			if constexpr(!SwapNodesPair) {
				WavefrontPathFindingKernel<SetExitFlag><<<blocksDim, threadsDim, 0, stream>>>
					(*nodes->readOnly, *nodes->writeOnly, graphW, graphH, *edgesWeights, goalIndex, maxAgentsGPerGroup, exitFlagDevice);
			}
			else {
				WavefrontPathFindingKernel<SetExitFlag><<<blocksDim, threadsDim, 0, stream>>>
					(*nodes->writeOnly, *nodes->readOnly, graphW, graphH, *edgesWeights, goalIndex, maxAgentsGPerGroup, exitFlagDevice);
			}

			return cudaGetLastError();
		}

	}
}