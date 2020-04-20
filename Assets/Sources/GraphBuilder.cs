using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TrailEvolutionModelling
{
    [ExecuteInEditMode]
    public class GraphBuilder : MonoBehaviour
    {
        [SerializeField] WorkingBounds bounds = null;
        [SerializeField] GraphHolder target = null;
        [SerializeField] float step = 0.5f;

        private static (int di, int dj)[] MooreIndexShifts =
        {
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    };

        private static (int di, int dj)[] HexagonalIndexShiftsEven =
        {
        (-1, 1), (0, 1), (1, 1),
        (-1, 0), (0, -1), (1, 0)
    };

        private static (int di, int dj)[] HexagonalIndexShiftsOdd =
        {
        (-1, 0), (0, 1), (1, 0),
        (-1, -1), (0, -1), (1, -1)
    };

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

        }

        public void Build(bool moore = false)
        {
            if (bounds == null || target == null)
                return;

            target.Graph = moore ? BuildRectangularMoore() : BuildHexagonal();
        }

        private Graph BuildRectangularMoore()
        {
            var graph = new Graph();
            BuildRectangularMooreNodes(graph);
            BuildRectangularMooreEdges(graph);
            
            return graph;
        }

        private Graph BuildHexagonal()
        {
            throw new NotSupportedException("Not supported more");
            //var graph = new Graph();
            //BuildHexagonalNodes(graph);
            //BuildHexagonalEdges(graph);
            //return graph;
        }

        private void BuildRectangularMooreNodes(Graph graph)
        {
            Vector2 min = bounds.Min;
            int w = (int)(bounds.Size.x / step);
            int h = (int)(bounds.Size.y / step);

            var nodes = new Node[w][];
            for (int i = 0; i < w; i++)
            {
                nodes[i] = new Node[h];
                for (int j = 0; j < h; j++)
                {
                    var position = min + new Vector2(i, j) * step;
                    if (MapObject.IsPointWalkable(position))
                    {
                        nodes[i][j] = new Node(position)
                        {
                            ComputeIndexI = i + 1,
                            ComputeIndexJ = j + 1,
                            ComputeIndex = i + 1 + (j + 1) * (w + 2)
                        };
                    }
                }
            }
            graph.Nodes = nodes;
            graph.ComputeNodes = new ComputeNode[(w + 2) * (h + 2)];
        }

        private void BuildRectangularMooreEdges(Graph graph)
        {
            var nodes = graph.Nodes;
            int w = nodes.Length;
            int h = nodes[0].Length;

            int edgeArraysSize = (w + 1) * (h + 1);
            graph.ComputeEdgesHoriz = new float[edgeArraysSize];
            graph.ComputeEdgesVert = new float[edgeArraysSize];
            graph.ComputeEdgesLeftDiag = new float[edgeArraysSize];
            graph.ComputeEdgesRightDiag = new float[edgeArraysSize];


            // default. May be parallelized
            for (int i = 0; i < edgeArraysSize; i++)
            {
                graph.ComputeEdgesHoriz[i] = float.PositiveInfinity;
                graph.ComputeEdgesVert[i] = float.PositiveInfinity;
                graph.ComputeEdgesLeftDiag[i] = float.PositiveInfinity;
                graph.ComputeEdgesRightDiag[i] = float.PositiveInfinity;
            }

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    BuildEdgesAround(graph, nodes, i, j, MooreIndexShifts, true);
                }
            }
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

        private void BuildEdgesAround(Graph graph, Node[][] nodes, int i, int j, (int di, int dj)[] shifts, bool mulByDistance)
        {
            Node node = nodes[i][j];
            if (node == null)
            { 
                foreach (var shift in shifts)
                {
                    graph.GetComputeEdgeForNode(i, j, shift.di, shift.dj) = float.PositiveInfinity;
                }
                return;
            }

            foreach (var shift in shifts)
            {
                ref float computeEdge = ref graph.GetComputeEdgeForNode(i, j, shift.di, shift.dj);

                Node otherNode = GetNodeAtOrNull(nodes, i + shift.di, j + shift.dj);
                if (otherNode == null)
                {
                    computeEdge = float.PositiveInfinity;
                    continue;
                }

                AreaAttributes areaAttributes = MapObject.GetAreaAttributes(node.Position, otherNode.Position);
                if (!areaAttributes.IsWalkable)
                {
                    computeEdge = float.PositiveInfinity;
                    continue;
                }

                float weight = areaAttributes.Weight;
                if (mulByDistance)
                    weight *= Vector2.Distance(node.Position, otherNode.Position) / step;
                graph.AddEdge(node, otherNode, weight, areaAttributes.IsTramplable);

                computeEdge = weight;
            }
        }

        private static Node GetNodeAtOrNull(Node[][] nodes, int i, int j)
        {
            int w = nodes.Length;
            int h = nodes[0].Length;
            if (i < 0 || j < 0 || i >= w || j >= h)
                return null;
            return nodes[i][j];
        }
    }
}