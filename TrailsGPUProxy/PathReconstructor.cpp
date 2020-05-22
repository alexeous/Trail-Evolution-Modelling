#include "PathReconstructor.h"
#include "NodeIndex.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		PathReconstructor::PathReconstructor(int graphW, int graphH, EdgesWeightsHost* edges,
			CudaScheduler* cudaScheduler, ResourceManager* resources, PathThickener* pathThickener)
			: graphW(graphW),
			  graphH(graphH),
			  edges(edges),
			  cudaScheduler(cudaScheduler),
			  threadPool(threadPool),
			  pathThickener(pathThickener),
			  distancePool(CreateDistanceHostPool(resources))
		{
		}

		ObjectPool<NodesFloatHost*>* PathReconstructor::CreateDistanceHostPool(
			ResourceManager* resources) 
		{
			return resources->New<ObjectPool<NodesFloatHost*>>(
				DISTANCE_HOST_POOL_SIZE, 
				[=] { return resources->New<NodesFloatHost>(graphW, graphH); }
			);
		}

		void PathReconstructor::StartPathReconstruction(Attractor start, Attractor goal, 
			ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes) 
		{
			PoolEntry<NodesFloatHost*> distanceEntry = distancePool->Take();
			ReconstructPath(start, goal, startNodes, goalNodes, distanceEntry.object);
			float averagePerformance = (start.performance + goal.performance) / 2;
			pathThickener->StartThickening(distanceEntry, averagePerformance, cudaScheduler);
		}
		
		template<typename T> int sign(T val) { return (T(0) < val) - (val < T(0)); }

		void PathReconstructor::ReconstructPath(Attractor startAttractor, Attractor goalAttractor, 
			ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes, 
			NodesFloatHost* distanceToPath) 
		{
			distanceToPath->Fill(INFINITY);

			NodeIndex start(startAttractor.nodeI + 1, startAttractor.nodeJ + 1);
			NodeIndex goal(goalAttractor.nodeI + 1, goalAttractor.nodeJ + 1);
			NodeIndex prev = start;
			NodeIndex prevGuide = start;
			NodeIndex guide = goalNodes->At(start).NextIndex(start);

			Pave(distanceToPath, start);
			while(guide != goal) {
				NodeIndex nextGuide = goalNodes->At(guide).NextIndex(guide);

				float minForwardG = goalNodes->At(nextGuide).g;
				float maxForwardG = goalNodes->At(prevGuide).g;
				float minBackwardG = startNodes->At(prevGuide).g;
				float maxBackwardG = startNodes->At(nextGuide).g;

				std::unordered_set<NodeIndex> visited { guide };
				std::vector<NodeIndex> similarCostNodes { guide };
				NodeIndex sumPos = guide;

				SimilarCostNodesSearch(guide, startNodes, goalNodes,
					minForwardG, maxForwardG, minBackwardG, maxBackwardG,
					visited, similarCostNodes, sumPos);

				float2 averagePos = sumPos / (float)similarCostNodes.size();
				NodeIndex next = GetClosestTo(similarCostNodes, averagePos);
				
				BridgeExclusively(distanceToPath, prev, next);
				Pave(distanceToPath, next);

				prev = next;
				prevGuide = guide;
				guide = nextGuide;
			}
			Pave(distanceToPath, goal);
		}

		inline void PathReconstructor::SimilarCostNodesSearch(NodeIndex index, 
			ComputeNodesHost* startNodes, ComputeNodesHost* goalNodes, 
			float minForwardG, float maxForwardG, float minBackwardG, float maxBackwardG, 
			std::unordered_set<NodeIndex>& visited, std::vector<NodeIndex>& result, NodeIndex& sumPos) 
		{
			for(int dir = 0; dir < 8; dir++) {
				if(isinf(edges->AtDir(index.i - 1, index.j - 1, graphW, dir)))
					continue;

				NodeIndex other = index + ComputeNode::DirectionToShift(dir);
				if(!visited.emplace(other).second)
					continue;

				float otherForwardG = goalNodes->At(other).g;
				float otherBackwardG = startNodes->At(other).g;

				if(otherForwardG > minForwardG && otherForwardG < maxForwardG &&
				    otherBackwardG > minBackwardG && otherBackwardG < maxBackwardG)
				{
					result.push_back(other);
				    sumPos += other;
					SimilarCostNodesSearch(other, startNodes, goalNodes,
						minForwardG, maxForwardG, minBackwardG, maxBackwardG,
						visited, result, sumPos);
				}
			}
		}

		inline NodeIndex PathReconstructor::GetClosestTo(const std::vector<NodeIndex>& indices, float2 to) {
			NodeIndex closest(-1, -1);
			float minSqrDistance = INFINITY;
			for(NodeIndex other : indices) {
				float sqrDist = other.SqrEuclideanDistance(to);
				if(sqrDist < minSqrDistance) {
					minSqrDistance = sqrDist;
					closest = other;
				}
			}
			return closest;
		}

		inline void PathReconstructor::Pave(NodesFloatHost* distanceToPath, NodeIndex index) {
			distanceToPath->At(index) = 0;
		}

		inline void PathReconstructor::BridgeExclusively(NodesFloatHost* distanceToPath, NodeIndex from, NodeIndex to) {
			NodeIndex intermediate = from;
			NodeIndex delta = to - intermediate;
			while(abs(delta.i) > 1 || abs(delta.j) > 1) {
				intermediate += NodeIndex(sign(delta.i), sign(delta.j));
				delta = to - intermediate;

				Pave(distanceToPath, intermediate);
			}
		}

		void PathReconstructor::Free(ResourceManager& resources) {
			resources.Free(distancePool);
		}


	}
}