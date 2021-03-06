#include "PathReconstructor.h"
#include "NodeIndex.h"
#include "MathUtils.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		PathReconstructor::PathReconstructor(int graphW, int graphH, EdgesWeightsHost* edges,
			CudaScheduler* cudaScheduler, ThreadPool *threadPool, ResourceManager* resources, 
			PathThickener* pathThickener, const AttractorsMap& attractorsMap,
			TramplabilityMaskHost* tramplabilityMaskHost)
			: graphW(graphW),
			  graphH(graphH),
			  edges(edges),
			  cudaScheduler(cudaScheduler),
			  threadPool(threadPool),
			  pathThickener(pathThickener),
			  attractorsMap(attractorsMap),
		      tramplabilityMask(tramplabilityMaskHost),
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
			threadPool->Schedule([=] {
				PoolEntry<NodesFloatHost*> distanceEntry = distancePool->Take();
				ReconstructPath(start, goal, startNodes, goalNodes, distanceEntry.object);
				float peoplePerSecond = CalcPathFlow(start, goal);
				pathThickener->StartThickening(distanceEntry, peoplePerSecond, cudaScheduler);
			});
		}

		float PathReconstructor::CalcPathFlow(const Attractor& start, const Attractor& goal) const {
			//return (start.performance * goal.performance / attractorsMap.GetSumReachablePerformance(start)
			//	+ goal.performance * start.performance / attractorsMap.GetSumReachablePerformance(goal)) / 2;
			return (start.performance * goal.performance / attractorsMap.GetSumReachablePerformance(start)
				+ goal.performance * start.performance / attractorsMap.GetSumReachablePerformance(goal));
			//return (start.performance + goal.performance) / 2;
		}

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
				
				NodeIndex bridgeLast = BridgeExclusively(distanceToPath, prev, next);
				Pave(distanceToPath, next, bridgeLast);

				prev = next;
				prevGuide = guide;
				guide = nextGuide;
			}
			NodeIndex preGoal = BridgeExclusively(distanceToPath, prev, goal);
			Pave(distanceToPath, goal, preGoal);
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

		inline void PathReconstructor::Pave(NodesFloatHost* distanceToPath, NodeIndex index, NodeIndex lastPaved) {
			if(index == lastPaved)
				return;

			NodeIndex shift = index - lastPaved;
			bool isEdgeTramplable = tramplabilityMask->AtShift(lastPaved.i, lastPaved.j, graphW, shift.i, shift.j);
			if(!isEdgeTramplable) {
				return;
			}
			distanceToPath->At(index) = 0;
		}

		inline NodeIndex PathReconstructor::BridgeExclusively(NodesFloatHost* distanceToPath, NodeIndex from, NodeIndex to) {
			NodeIndex intermediate = from;
			NodeIndex delta = to - intermediate;
			while(abs(delta.i) > 1 || abs(delta.j) > 1) {
				NodeIndex lastPaved = intermediate;
				intermediate += NodeIndex(sign(delta.i), sign(delta.j));
				delta = to - intermediate;

				Pave(distanceToPath, intermediate, lastPaved);
			}
			return intermediate;
		}

		void PathReconstructor::Free(ResourceManager& resources) {
			resources.Free(distancePool);
		}


	}
}