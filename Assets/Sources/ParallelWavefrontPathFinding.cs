using System;
using System.Collections.Generic;
using System.Linq;
using TrailEvolutionModelling;
using UnityEngine;

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

        for (int i = 0; i < graph.Nodes.Length; i++)
        {
            for (int j = 0; j < graph.Nodes[0].Length; j++)
            {
                if (graph.Nodes[i][j] != null)
                {
                    float g = nodesStartToGoal[graph.Nodes[i][j].ComputeIndex].G;
                    if (float.IsInfinity(g))
                        graph.Nodes[i][j] = null;
                    else
                        graph.Nodes[i][j].G1 = g;
                }
            }
        }
        Node[] nodesFlattened = graph.Nodes.SelectMany(t => t).Where(t => t != null).ToArray();
        PathFinder.RedrawHeatmap(nodesFlattened);

        return ReconstructPath(nodesStartToGoal, start.ComputeIndexI, start.ComputeIndexJ);


        ComputeNode[] ComputePaths(Node _goal, params Node[] starts)
        {
            SetupNodesIsStartField(_goal, starts);
            nodesRead.SetData(graph.ComputeNodes);
            InitNodes(_goal.ComputeIndex);

            (int groupsX, int groupsY) = GetGroupsNumber(plannerKernel, width, height); 
            
            var result = new ComputeNode[nodesWrite.count];

            //shader.SetFloat(maxAgentsGID, float.PositiveInfinity);
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

            int[] exitFlagData = { 1 };
            float[] maxAgentsGPerGroup = new float[maxAgentsGPerGroupBuffer.count];
            int it = 0;
            while (true)
            {
                it++;
                if (it >= 2000)
                {
                    Debug.LogError("Path finding takes too long. Breaking loop...");
                    break;
                }
                exitFlagData[0] = 1;
                exitFlagBuffer.SetData(exitFlagData);

                (nodesRead, nodesWrite) = (nodesWrite, nodesRead);
                shader.SetBuffer(plannerKernel, nodesReadID, nodesRead);
                shader.SetBuffer(plannerKernel, nodesWriteID, nodesWrite);

                shader.Dispatch(plannerKernel, groupsX, groupsY, 1);
                
                exitFlagBuffer.GetData(exitFlagData);
                if (exitFlagData[0] != 0)
                    break;

                //float maxAgentsG = FetchAndAggregateMaxAgentsG();
                //shader.SetFloat(maxAgentsGID, maxAgentsG);
            }

            nodesWrite.GetData(result);
            return result;

            //float FetchAndAggregateMaxAgentsG()
            //{
            //    maxAgentsGPerGroupBuffer.GetData(maxAgentsGPerGroup);
            //    return Mathf.Max(maxAgentsGPerGroup);
            //}
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

        Node[] ReconstructPath(ComputeNode[] computeNodes, int computeI, int computeJ)
        {
            var pathNodes = new List<Node>();
            Node node = ComputeToNode(computeI, computeJ);
            while (node != goal)
            {
                if (pathNodes.Count > 100000)
                {
                    pathNodes.RemoveRange(100, pathNodes.Count - 100);
                    Debug.LogError("Something went wrong");
                    pathNodes.RemoveAll(x => x == null);
                    return pathNodes.ToArray();
                }
                pathNodes.Add(node);
                (computeI, computeJ) = NextCompute(computeI, computeJ);
                node = ComputeToNode(computeI, computeJ);
            }
            pathNodes.Add(goal);
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
            pathNodes.Add(goal);
            return pathNodes.ToArray();

            Node ComputeToNode(int compI, int compJ) => graph.Nodes[compI - 1][compJ - 1];
            (int nextI, int nextJ) NextCompute(int compI, int compJ)
            {
                (int di, int dj)[] indexShifts =
                {
                    (-1, -1), (0, -1), (1, -1),
                    (-1, 0), (1, 0),
                    (-1, 1), (0, 1), (1, 1)
                };
                ComputeNode comp = computeNodes[compI + compJ * (width + 2)];
                var shift = indexShifts[comp.DirectionNext];
                return (compI + shift.di, compJ + shift.dj);
            }
        }
    }

    private (int groupsX, int groupsY) GetGroupsNumber(int plannerKernel, int width, int height)
    {
        uint groupSizeX, groupSizeY, groupSizeZ;
        shader.GetKernelThreadGroupSizes(plannerKernel, out groupSizeX, out groupSizeY, out groupSizeZ);
        int groupsX = Mathf.CeilToInt((float)width / groupSizeX);
        int groupsY = Mathf.CeilToInt((float)height / groupSizeY);
        return (groupsX, groupsY);
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
