using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using TrailEvolutionModelling;
using UnityEngine;
using UnityEngine.Rendering;

public class ParallelWavefrontPathFinding : MonoBehaviour
{
    [SerializeField] ComputeShader shader = null;

    public Node[] FindPath(Graph graph, Node start, Node goal)
    {
        if (start == goal)
        {
            return new Node[] { start };
        }

        int nodesReadID = Shader.PropertyToID("nodesRead");
        int nodesWriteID = Shader.PropertyToID("nodesWrite");
        int goalIndexID = Shader.PropertyToID("goalIndex");
        int maxAgentsGID = Shader.PropertyToID("maxAgentsG");

        int initKernel = shader.FindKernel("WavefrontInitNodesKernel");
        int plannerKernel = shader.FindKernel("WavefrontPlannerKernel");


        int width = graph.Nodes.Length;
        int height = graph.Nodes[0].Length;

        shader.SetInt("nodesWidth", width);
        shader.SetInt("nodesHeight", height);


        var edgesVert = NewFloatsBuffer(graph.ComputeEdgesVert);
        var edgesHoriz = NewFloatsBuffer(graph.ComputeEdgesHoriz);
        var edgesLeftDiag = NewFloatsBuffer(graph.ComputeEdgesLeftDiag);
        var edgesRightDiag = NewFloatsBuffer(graph.ComputeEdgesRightDiag);
        var nodesRead = NewNodesBuffer(graph.ComputeNodes.Length);
        var nodesWrite = NewNodesBuffer(graph.ComputeNodes.Length);
        var exitFlagBuffer = NewIntBuffer(0);
        var maxAgentsGPerGroupBuffer = NewMaxAgentsGPerGroupBuffer(plannerKernel, width, height);

        var allBuffers = new[] { edgesVert, edgesHoriz, edgesLeftDiag, edgesRightDiag,
            nodesRead, nodesWrite, exitFlagBuffer, maxAgentsGPerGroupBuffer };

        ComputeNode[] nodesStartToGoal = ComputePaths(goal, start);
        ComputeNode[] nodesGoalToStart = ComputePaths(start, goal);

        ReleaseAndDisposeBuffers(allBuffers);

        //for (int i = 0; i < graph.Nodes.Length; i++)
        //{
        //    for (int j = 0; j < graph.Nodes[0].Length; j++)
        //    {
        //        if (graph.Nodes[i][j] != null)
        //        {
        //            float g = nodesStartToGoal[graph.Nodes[i][j].ComputeIndex].G;
        //            graph.Nodes[i][j].G1 = g;
        //        }
        //    }
        //}
        //Node[] nodesFlattened = graph.Nodes.SelectMany(t => t).Where(t => t != null).ToArray();
        //PathFinder.RedrawHeatmap(nodesFlattened);

        return ReconstructPath(nodesStartToGoal, nodesGoalToStart, start.ComputeIndexI, start.ComputeIndexJ, graph, goal);


        ComputeNode[] ComputePaths(Node _goal, params Node[] starts)
        {
            SetupNodesIsStartField(_goal, starts);
            nodesRead.SetData(graph.ComputeNodes);
            InitNodes(_goal.ComputeIndex);

            (int groupsX, int groupsY) = GetGroupsNumber(plannerKernel, width, height); 
            
            shader.SetInt("groupsX", groupsX);
            shader.SetInt("groupsY", groupsY);

            shader.SetBuffer(plannerKernel, nodesReadID, nodesRead);
            shader.SetBuffer(plannerKernel, nodesWriteID, nodesWrite);
            shader.SetBuffer(plannerKernel, "edgesVert", edgesVert);
            shader.SetBuffer(plannerKernel, "edgesHoriz", edgesHoriz);
            shader.SetBuffer(plannerKernel, "edgesLeftDiag", edgesLeftDiag);
            shader.SetBuffer(plannerKernel, "edgesRightDiag", edgesRightDiag);
            shader.SetBuffer(plannerKernel, "exitFlagBuffer", exitFlagBuffer);
            shader.SetBuffer(plannerKernel, "maxAgentsGPerGroup", maxAgentsGPerGroupBuffer);

            GetClosestStepsDistanceNode(_goal, starts, out int closestNodeStepsDistance);
            int minIterations = closestNodeStepsDistance + 2;

            int[] exitFlagData = { 1 };
            const int exitFlagCheckPeriod = 10;
            int iteration = 0;
            while (true)
            {
                iteration++;
                if (iteration >= 2000)
                {
                    Debug.LogError("Path finding takes too long. Breaking loop...");
                    break;
                }

                if (iteration >= minIterations)
                {
                    if (iteration % exitFlagCheckPeriod == 0 || iteration == minIterations)
                    {
                        exitFlagData[0] = 1;
                        exitFlagBuffer.SetData(exitFlagData);
                    }
                }

                (nodesRead, nodesWrite) = (nodesWrite, nodesRead);
                shader.SetBuffer(plannerKernel, nodesReadID, nodesRead);
                shader.SetBuffer(plannerKernel, nodesWriteID, nodesWrite);

                shader.Dispatch(plannerKernel, groupsX, groupsY, 1);

                if (iteration >= minIterations)
                {
                    if (iteration % exitFlagCheckPeriod == 0 || iteration == minIterations)
                    {
                        exitFlagBuffer.GetData(exitFlagData);
                        if (exitFlagData[0] != 0)
                            break;
                    }
                }
            }
            //Debug.Log("Iterations: " + iteration);

            var result = new ComputeNode[nodesWrite.count];
            nodesWrite.GetData(result);
            return result;
        }

        void InitNodes(int goalIndex)
        {
            shader.SetInt(goalIndexID, goalIndex);
            shader.SetBuffer(initKernel, nodesReadID, nodesRead);
            shader.SetBuffer(initKernel, nodesWriteID, nodesWrite);

            (int groupsX, int groupsY) = GetGroupsNumber(initKernel, width + 2, height + 2);

            shader.Dispatch(initKernel, groupsX, groupsY, 1);
        }

        void SetupNodesIsStartField(Node _goal, params Node[] starts)
        {
            graph.ComputeNodes[_goal.ComputeIndex].IsStart = false;
            foreach (var _start in starts)
            {
                graph.ComputeNodes[_start.ComputeIndex].IsStart = true;
            }
        }
    }

    private static Node[] ReconstructPath(ComputeNode[] nodesStartToGoal, ComputeNode[] nodesGoalToStart, int i, int j, Graph graph, Node goal)
    {
        var pathNodes = new List<Node> { ComputeToNode(i, j) };
        var (prevGuideI, prevGuideJ) = (i, j);
        var (guideI, guideJ) = NextCompute(i, j);
        var (prevI, prevJ) = (i, j);
        var (goalI, goalJ) = (goal.ComputeIndexI, goal.ComputeIndexJ);
        while (guideI != goalI || guideJ != goalJ)
        {
            if (pathNodes.Count > 100000)
            {
                pathNodes.RemoveRange(100, pathNodes.Count - 100);
                Debug.LogError("Something went wrong");
                pathNodes.RemoveAll(x => x == null);
                return pathNodes.ToArray();
            }

            var (nextGuideI, nextGuideJ) = NextCompute(guideI, guideJ);

            float minForwardG = GetComputeAt(nodesStartToGoal, nextGuideI, nextGuideJ).G;
            float maxForwardG = GetComputeAt(nodesStartToGoal, prevGuideI, prevGuideJ).G;
            float minBackwardG = GetComputeAt(nodesGoalToStart, prevGuideI, prevGuideJ).G;
            float maxBackwardG = GetComputeAt(nodesGoalToStart, nextGuideI, nextGuideJ).G;

            var visited = new HashSet<(int, int)>() { (guideI, guideJ) };
            var similarCostNodeIndices = new List<(int i, int j)> { (guideI, guideJ) };

            Vector2Int sumPos = new Vector2Int(guideI, guideJ);

            SimilarCostNodesSearch(guideI, guideJ);
            void SimilarCostNodesSearch(int compI, int compJ)
            {
                for (int dir = 0; dir < 8; dir++)
                {
                    var shift = Graph.DirectionToShift(dir);
                    if (float.IsInfinity(graph.GetComputeEdgeForComputeNode(compI, compJ, shift.di, shift.dj)))
                        continue;

                    var (otherI, otherJ) = (compI + shift.di, compJ + shift.dj);
                    if (visited.Contains((otherI, otherJ)))
                        continue;

                    visited.Add((otherI, otherJ));
                    ComputeNode otherForward = GetComputeAt(nodesStartToGoal, otherI, otherJ);
                    ComputeNode otherBackward = GetComputeAt(nodesGoalToStart, otherI, otherJ);

                    if (otherForward.G >= minForwardG && otherForward.G < maxForwardG &&
                        otherBackward.G > minBackwardG && otherBackward.G <= maxBackwardG)
                    {
                        similarCostNodeIndices.Add((otherI, otherJ));
                        sumPos += new Vector2Int(otherI, otherJ);
                        SimilarCostNodesSearch(otherI, otherJ);
                    }
                }
            }

            Vector2 prevPos = new Vector2(prevI, prevJ);
            Vector2 averagePos = (Vector2)sumPos / similarCostNodeIndices.Count;
            int nextI = -1;
            int nextJ = -1;
            float minSqrDist = float.PositiveInfinity;
            foreach (var nodeIndex in similarCostNodeIndices)
            {
                float sqrDist = (averagePos - new Vector2(nodeIndex.i, nodeIndex.j)).sqrMagnitude;
                if (sqrDist < minSqrDist)
                {
                    minSqrDist = sqrDist;
                    nextI = nodeIndex.i;
                    nextJ = nodeIndex.j;
                }
            }

            if (nextI == -1)
            {
                throw new Exception("There are no similarCostNodeIndices found");
            }

            int intermediateI = prevI;
            int intermediateJ = prevJ;

            int deltaI = nextI - intermediateI;
            int deltaJ = nextJ - intermediateJ;
            while (Mathf.Abs(deltaI) > 1 || Mathf.Abs(deltaJ) > 1)
            {
                intermediateI += Math.Sign(deltaI);
                intermediateJ += Math.Sign(deltaJ);
                deltaI = nextI - intermediateI;
                deltaJ = nextJ - intermediateJ;
                pathNodes.Add(ComputeToNode(intermediateI, intermediateJ));
            }

            pathNodes.Add(ComputeToNode(nextI, nextJ));

            (prevI, prevJ) = (nextI, nextJ);
            (prevGuideI, prevGuideJ) = (guideI, guideJ);
            (guideI, guideJ) = (nextGuideI, nextGuideJ);
        }
        pathNodes.Add(goal);
        return pathNodes.ToArray();


        Node ComputeToNode(int compI, int compJ)
        {
            return graph.Nodes[compI - 1][compJ - 1];
        }

        ComputeNode GetComputeAt(ComputeNode[] array, int compI, int compJ)
        {
            return array[compI + compJ * (graph.Nodes.Length + 2)];
        }

        (int nextI, int nextJ) NextCompute(int compI, int compJ)
        {
            (int di, int dj)[] indexShifts =
            {
                (-1, -1), (0, -1), (1, -1),
                (-1, 0), (1, 0),
                (-1, 1), (0, 1), (1, 1)
            };
            ComputeNode comp = GetComputeAt(nodesStartToGoal, compI, compJ);
            var shift = indexShifts[comp.DirectionNext];
            return (compI + shift.di, compJ + shift.dj);
        }

        //Node prevGuide = current;
        //Node guide = current.CameFrom1;
        //while (guide != goal)
        //{
        //    if (pathNodes.Count > 100000)
        //    {
        //        throw new Exception("Something went wrong");
        //    }

        //    Node nextGuide = guide.CameFrom1;

        //    float minG1 = nextGuide.G1;
        //    float maxG1 = prevGuide.G1;
        //    float minG2 = prevGuide.G2;
        //    float maxG2 = nextGuide.G2;

        //    var visited = new HashSet<Node> { guide };
        //    var similarCostNodes = new List<Node> { guide };
        //    Vector2 averagePos = guide.Position;

        //    Rec(guide);
        //    void Rec(Node center)
        //    {
        //        foreach (var edge in center.IncidentEdges)
        //        {
        //            Node other = edge.GetOppositeNode(center);
        //            if (visited.Contains(other))
        //                continue;

        //            visited.Add(other);

        //            if (other.G1 >= minG1 && other.G1 <= maxG1 &&
        //                other.G2 >= minG2 && other.G2 <= maxG2)
        //            {
        //                similarCostNodes.Add(other);
        //                averagePos += other.Position;
        //                Rec(other);
        //            }
        //        }
        //    }

        //    averagePos /= similarCostNodes.Count;
        //    Node next = null;
        //    float minSqrDist = float.PositiveInfinity;
        //    foreach (var node in similarCostNodes)
        //    {
        //        float sqrDist = (averagePos - node.Position).sqrMagnitude;
        //        if (sqrDist < minSqrDist)
        //        {
        //            minSqrDist = sqrDist;
        //            next = node;
        //        }
        //    }
        //    pathNodes.Add(next);

        //    prevGuide = guide;
        //    guide = guide.CameFrom1;
        //}
    }

    private (int groupsX, int groupsY) GetGroupsNumber(int plannerKernel, int width, int height)
    {
        uint groupSizeX, groupSizeY, groupSizeZ;
        shader.GetKernelThreadGroupSizes(plannerKernel, out groupSizeX, out groupSizeY, out groupSizeZ);
        int groupsX = Mathf.CeilToInt((float)width / groupSizeX);
        int groupsY = Mathf.CeilToInt((float)height / groupSizeY);
        return (groupsX, groupsY);
    }

    private static Node GetClosestStepsDistanceNode(Node origin, Node[] others, out int minDistance)
    {
        Node closest = null;
        minDistance = int.MaxValue;
        foreach (var other in others)
        {
            int distance = GetNodeStepsDistance(origin, other);
            if (distance < minDistance)
            {
                minDistance = distance;
                closest = other;
            }
        }
        return closest;
    }

    private static int GetNodeStepsDistance(Node a, Node b)
    {
        int deltaI = a.ComputeIndexI - b.ComputeIndexI;
        int deltaJ = a.ComputeIndexJ - b.ComputeIndexJ;
        return Mathf.Max(Mathf.Abs(deltaI), Mathf.Abs(deltaJ));
    }

    private ComputeBuffer NewFloatsBuffer(float[] values)
    {
        var buffer = new ComputeBuffer(values.Length, sizeof(float));
        buffer.SetData(values);
        return buffer;
    }

    private ComputeBuffer NewNodesBuffer(int length)
    {
        var buffer = new ComputeBuffer(length, ComputeNode.StructSize);
        return buffer;
    }

    private ComputeBuffer NewIntBuffer(int value)
    {
        var buffer = new ComputeBuffer(1, sizeof(int));
        buffer.SetData(new int[] { value });
        return buffer;
    }

    private ComputeBuffer NewMaxAgentsGPerGroupBuffer(int plannerKernel, int w, int h)
    {
        uint groupSizeX, groupSizeY, groupSizeZ;
        shader.GetKernelThreadGroupSizes(plannerKernel, out groupSizeX, out groupSizeY, out groupSizeZ);

        int groupsX = Mathf.CeilToInt((float)w / groupSizeX);
        int groupsY = Mathf.CeilToInt((float)h / groupSizeY);
        var buffer = new ComputeBuffer(groupsX * groupsY, sizeof(float));
        var data = new float[groupsX * groupsY];
        buffer.SetData(data);
        return buffer;
    }

    private void ReleaseAndDisposeBuffers(IEnumerable<ComputeBuffer> buffers)
    {
        foreach (var buffer in buffers)
        {
            buffer.Release();
            buffer.Dispose();
        }
    }
}
