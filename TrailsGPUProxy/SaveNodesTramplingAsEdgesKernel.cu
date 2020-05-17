#include "SaveNodesTramplingAsEdgesKernel.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#define BLOCK_SIZE_X SAVE_NODES_TRAMPLING_AS_EDGES_BLOCK_SIZE_X
#define BLOCK_SIZE_Y SAVE_NODES_TRAMPLING_AS_EDGES_BLOCK_SIZE_Y

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatDevice = NodesDataHaloedDevice<float>;

		inline __device__ void SaveTramplingToEdge(float& edge, bool tramplability, float node1, float node2) {
			edge = tramplability * (node1 + node2) * 0.5f;
		}

		__global__ void SaveNodesTramplingAsEdgesKernel(NodesFloatDevice nodesTrampling, int graphW, int graphH,
			EdgesTramplingEffect edges, TramplabilityMask tramplability) 
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			if(i > graphW || j > graphH)
				return;

			float centerNodeTrampling = nodesTrampling.At(i + 1, j + 1, graphW);

			SaveTramplingToEdge(edges.E(i, j, graphW), tramplability.E(i, j, graphW), centerNodeTrampling, nodesTrampling.At(i + 2, j + 1, graphW));
			if(j < graphH - 1) {
				SaveTramplingToEdge(edges.S(i, j, graphW), tramplability.S(i, j, graphW), centerNodeTrampling, nodesTrampling.At(i + 1, j + 2, graphW));
				if(i < graphW - 1)
					SaveTramplingToEdge(edges.SE(i, j, graphW), tramplability.SE(i, j, graphW), centerNodeTrampling, nodesTrampling.At(i + 2, j + 2, graphW));
				if(i != 0)
					SaveTramplingToEdge(edges.SW(i, j, graphW), tramplability.SW(i, j, graphW), centerNodeTrampling, nodesTrampling.At(i, j + 2, graphW));
			}
		}

		cudaError SaveNodesTramplingAsEdges(NodesFloatDevice* nodesTrampling, int graphW, int graphH,
			EdgesTramplingEffect* targetEdges, TramplabilityMask* tramplabilityMask) 
		{
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetSaveNodesTramplingAsEdgesBlocksX(graphW),
				           GetSaveNodesTramplingAsEdgesBlocksY(graphH));
			
			SaveNodesTramplingAsEdgesKernel<<<blocksDim, threadsDim>>>(*nodesTrampling, graphW, graphH,
				*targetEdges, *tramplabilityMask);

			return cudaGetLastError();
		}

	}
}