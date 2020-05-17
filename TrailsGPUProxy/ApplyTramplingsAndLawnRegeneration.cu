#include "ApplyTramplingsAndLawnRegeneration.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE_X APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_X
#define BLOCK_SIZE_Y APPLY_TRAMPLINGS_AND_LAWN_REGENERATION_BLOCK_SIZE_Y

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		inline __device__ void SaveTramplingToEdge(float& edge, bool tramplability, float node1, float node2) {
			edge = tramplability * (node1 + node2) * 0.5f;
		}

		__global__ void ApplyTramplingsAndLawnRegenerationKernel(EdgesWeightsDevice target, int graphW, int graphH,
			EdgesTramplingEffect indecentTrampling, NodesTramplingEffect decentTrampling,
			TramplabilityMask tramplabilityMask) 
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

		void ApplyTramplingsAndLawnRegeneration(EdgesWeightsDevice* target, int graphW, int graphH, 
			EdgesTramplingEffect* indecentTramplingEdges, NodesTramplingEffect* decentTramplingNodes, 
			TramplabilityMask* tramplabilityMask) 
		{
			dim3 threadsDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocksDim(GetApplyTramplingsAndLawnRegenerationBlocksX(graphW),
				           GetApplyTramplingsAndLawnRegenerationBlocksY(graphH));


		}

	}
}