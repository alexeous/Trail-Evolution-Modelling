#pragma once
#include <unordered_set>
#include <vector>
#include "IResource.h"
#include "ResourceManager.h"
#include "Attractor.h"
#include "ThreadPool.h"
#include "CudaScheduler.h"
#include "ComputeNodesHost.h"
#include "NodesDataHaloed.h"
#include "ObjectPool.h"
#include "EdgesWeights.h"
#include "PathThickener.h"

#define DISTANCE_HOST_POOL_SIZE 20
#define DISTANCE_DEVICE_POOL_SIZE 20


namespace TrailEvolutionModelling {
	namespace GPUProxy {

		using NodesFloatHost = NodesDataHaloedHost<float>;

		class PathReconstructor : public IResource {
			friend class ResourceManager;

		public:

			void StartPathReconstruction(Attractor start, Attractor goal,
				ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes);

		protected:
			PathReconstructor(int graphW, int graphH, EdgesWeightsHost* edges, 
				CudaScheduler* cudaScheduler, ResourceManager* resources, 
				PathThickener* pathThickener);

			void Free(ResourceManager& resources) override;

		private:
			ObjectPool<NodesFloatHost*>* CreateDistanceHostPool(ResourceManager*);
			
			void ReconstructPath(Attractor start, Attractor goal,
				ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes, 
				NodesFloatHost* distanceToPath);

			inline void SimilarCostNodesSearch(NodeIndex index, 
				ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes,
				float minForwardG, float maxForwardG, float minBackwardG, float maxBackwardG,
				std::unordered_set<NodeIndex>& visited, std::vector<NodeIndex>& result, 
				NodeIndex& sumPos);

			inline NodeIndex GetClosestTo(const std::vector<NodeIndex>& indices, float2 to);
			inline void Pave(NodesFloatHost* distanceToPath, NodeIndex index);
			inline void BridgeExclusively(NodesFloatHost* distanceToPath, NodeIndex from, NodeIndex to);


		private:
			int graphW;
			int graphH;
			EdgesWeightsHost* edges = nullptr;
			CudaScheduler* cudaScheduler = nullptr;
			ThreadPool* threadPool = nullptr;
			PathThickener* pathThickener = nullptr;
			ObjectPool<NodesFloatHost*>* distancePool = nullptr;
		};

	}
}