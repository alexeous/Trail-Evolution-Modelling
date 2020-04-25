using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TrailEvolutionModelling.GraphTypes;

namespace TrailEvolutionModelling
{
    [ExecuteInEditMode]
    public class GraphBuilder : MonoBehaviour
    {
        [SerializeField] WorkingBounds bounds = null;
        [SerializeField] GraphHolder target = null;
        [SerializeField] float step = 0.5f;

    //    private static (int di, int dj)[] MooreIndexShifts =
    //    {
    //    (-1, -1), (0, -1), (1, -1),
    //    (-1, 0), (1, 0),
    //    (-1, 1), (0, 1), (1, 1)
    //};

    //    private static (int di, int dj)[] HexagonalIndexShiftsEven =
    //    {
    //    (-1, 1), (0, 1), (1, 1),
    //    (-1, 0), (0, -1), (1, 0)
    //};

    //    private static (int di, int dj)[] HexagonalIndexShiftsOdd =
    //    {
    //    (-1, 0), (0, 1), (1, 0),
    //    (-1, -1), (0, -1), (1, -1)
    //};

        private void OnValidate()
        {
            if (step < 0.25f)
            {
                step = 0.25f;
            }
        }

        [ContextMenu("Build")]
        public void Build()
        {
            if (bounds == null || target == null)
                return;

            target.Graph = BuildRectangularMoore();
        }

        private Graph BuildRectangularMoore()
        {
            int w = Mathf.CeilToInt(bounds.Size.x / step);
            int h = Mathf.CeilToInt(bounds.Size.y / step);
            Vector2 min = bounds.Min;

            var graph = new Graph(w, h, min.x, min.y, step);
            BuildRectangularMooreNodes(graph);
            BuildRectangularEdges(graph);
            return graph;
        }

        //private Graph BuildHexagonal()
        //{
        //    throw new NotSupportedException("Not supported more");
        //    //var graph = new Graph();
        //    //BuildHexagonalNodes(graph);
        //    //BuildHexagonalEdges(graph);
        //    //return graph;
        //}

        private void BuildRectangularMooreNodes(Graph graph)
        {
            //var nodes = new Node[w][];
            for (int i = 0; i < graph.Width; i++)
            {
                //nodes[i] = new Node[h];
                for (int j = 0; j < graph.Height; j++)
                {
                    if (MapObject.IsPointWalkable(graph.GetNodePosition(i, j).ToVector2()))
                    {
                        graph.AddNode(i, j);
                        //nodes[i][j] = new Node(position)
                        //{
                        //    ComputeIndexI = i + 1,
                        //    ComputeIndexJ = j + 1,
                        //    ComputeIndex = i + 1 + (j + 1) * (w + 2)
                        //};
                    }
                }
            }
            //graph.Nodes = nodes;
            //graph.ComputeNodes = new ComputeNode[(w + 2) * (h + 2)];
        }

        private void BuildRectangularEdges(Graph graph)
        {
            for (int i = 0; i < graph.Width; i++)
            {
                for (int j = 0; j < graph.Height; j++)
                {
                    Node node = graph.GetNodeAtOrNull(i, j);
                    if (node == null)
                        continue;

                    bool notLastColumn = i < graph.Width - 1;
                    bool notLastRow = j < graph.Height - 1;
                    bool notFirstColumn = i != 0;

                    if (notLastColumn)
                        BuildEdge(graph, node, Direction.E);

                    if (notLastRow) {
                        BuildEdge(graph, node, Direction.S);

                        if (notLastColumn)
                            BuildEdge(graph, node, Direction.SE);

                        if (notFirstColumn)
                            BuildEdge(graph, node, Direction.SW);
                    }
                }
            }
            //for (int i = 1; i < graph.Width - 1; i++)
            //{
            //    for (int j = 1; j < graph.Height - 1; j++)
            //    {
            //        Node node = graph.GetNodeAtOrNull(i, j);
            //        if (node == null)
            //            continue;

            //        for (var dir = Direction.First; dir <= Direction.Last; dir++)
            //        {
            //            Node neighbour = graph.GetNodeNeighbourOrNull(node, dir);
            //            if (neighbour == null)
            //                continue;

                          //GetAreaAttributes(graph);
            //            if (!areaAttributes.IsWalkable)
            //                continue;

            //            float weight = areaAttributes.Weight * dir.WeightMultiplier();
            //            graph.AddEdge(node, dir, weight, areaAttributes.IsTramplable);
            //        }
            //    }
            //}

            //var nodes = graph.Nodes;
            //int w = graph.Width;
            //int h = graph.Height;

            //int edgeArraysSize = (w + 1) * (h + 1);
            //graph.ComputeEdgesHoriz = new float[edgeArraysSize];
            //graph.ComputeEdgesVert = new float[edgeArraysSize];
            //graph.ComputeEdgesLeftDiag = new float[edgeArraysSize];
            //graph.ComputeEdgesRightDiag = new float[edgeArraysSize];


            //// default. May be parallelized
            //for (int i = 0; i < edgeArraysSize; i++)
            //{
            //    graph.ComputeEdgesHoriz[i] = float.PositiveInfinity;
            //    graph.ComputeEdgesVert[i] = float.PositiveInfinity;
            //    graph.ComputeEdgesLeftDiag[i] = float.PositiveInfinity;
            //    graph.ComputeEdgesRightDiag[i] = float.PositiveInfinity;
            //}

            //for (int i = 0; i < w; i++)
            //{
            //    for (int j = 0; j < h; j++)
            //    {
            //        BuildEdgesAround(graph, nodes, i, j, MooreIndexShifts, true);
            //    }
            //}
        }

        private void BuildEdge(Graph graph, Node node, Direction direction)
        {
            if (TryGetAreaAttributes(graph, node, direction, out var area) &&
                area.IsWalkable)
            {
                float weight = area.Weight * direction.WeightMultiplier();
                graph.AddEdge(node, direction, weight, area.IsTramplable);
            }
        }

        private static bool TryGetAreaAttributes(Graph graph, Node node, Direction dir, out AreaAttributes area)
        {
            Node neighbour = graph.GetNodeNeighbourOrNull(node, dir);
            if (neighbour == null)
            {
                area = default;
                return false; 
            }

            Vector2 nodePos = graph.GetNodePosition(node).ToVector2();
            Vector2 neighbourPos = graph.GetNodePosition(neighbour).ToVector2();
            area = MapObject.GetAreaAttributes(nodePos, neighbourPos);
            return true;
        }

        //private void BuildHexagonalNodes(Graph graph)
        //{
        //    float cos30 = Mathf.Cos(Mathf.PI / 6);

        //    Vector2 min = bounds.Min + new Vector2(0, 0.5f);
        //    int w = Mathf.CeilToInt(bounds.Size.x / (step * cos30));
        //    int h = Mathf.CeilToInt(bounds.Size.y / step);

        //    var nodes = new Node[w][];

        //    for (int i = 0; i < w; i++)
        //    {
        //        for (int j = 0; j < h; j++)
        //        {
        //            float evenOddShift = i % 2 == 0 ? 0 : -0.5f;
        //            var position = min + new Vector2(i * cos30, j + evenOddShift) * step;
        //            if (MapObject.IsPointWalkable(position))
        //            {
        //                nodes[i][j] = new Node(position);
        //            }
        //        }
        //    }

        //    graph.Nodes = nodes;
        //}

        //private void BuildHexagonalEdges(Graph graph)
        //{
        //    var nodes = graph.Nodes;

        //    int w = nodes.Length;
        //    int h = nodes[0].Length;

        //    for (int i = 0; i < w; i++)
        //    {
        //        for (int j = 0; j < h; j++)
        //        {
        //            (int di, int dj)[] shifts = i % 2 == 0
        //                                            ? HexagonalIndexShiftsEven
        //                                            : HexagonalIndexShiftsOdd;

        //            BuildEdgesAround(graph, nodes, i, j, shifts, false);
        //        }
        //    }
        //}

        //private void BuildEdgesAround(Graph graph, Node[][] nodes, int i, int j, (int di, int dj)[] shifts, bool mulByDistance)
        //{
        //    //Node node = nodes[i][j];
        //    //if (node == null)
        //    //{ 
        //    //    foreach (var shift in shifts)
        //    //    {
        //    //        graph.GetComputeEdgeForNode(i, j, shift.di, shift.dj) = float.PositiveInfinity;
        //    //    }
        //    //    return;
        //    //}

        //    foreach (var shift in shifts)
        //    {
        //        ref float computeEdge = ref graph.GetComputeEdgeForNode(i, j, shift.di, shift.dj);

        //        Node otherNode = GetNodeAtOrNull(nodes, i + shift.di, j + shift.dj);
        //        if (otherNode == null)
        //        {
        //            computeEdge = float.PositiveInfinity;
        //            continue;
        //        }

        //        AreaAttributes areaAttributes = MapObject.GetAreaAttributes(node.Position, otherNode.Position);
        //        if (!areaAttributes.IsWalkable)
        //        {
        //            computeEdge = float.PositiveInfinity;
        //            continue;
        //        }

        //        float weight = areaAttributes.Weight;
        //        if (mulByDistance)
        //            weight *= Vector2.Distance(node.Position, otherNode.Position) / step;
        //        graph.AddEdge(node, otherNode, weight, areaAttributes.IsTramplable);

        //        computeEdge = weight;
        //    }
        //}

        //private static Node GetNodeAtOrNull(Node[][] nodes, int i, int j)
        //{
        //    int w = nodes.Length;
        //    int h = nodes[0].Length;
        //    if (i < 0 || j < 0 || i >= w || j >= h)
        //        return null;
        //    return nodes[i][j];
        //}
    }
}