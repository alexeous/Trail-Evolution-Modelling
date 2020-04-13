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
            var graph = new Graph();
            BuildHexagonalNodes(graph);
            BuildHexagonalEdges(graph);
            return graph;
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
                        nodes[i][j] = new Node(position);
                    }
                }
            }
            graph.Nodes = nodes;
            graph.ComputeNodes = new ComputeNode[w * h];
        }

        private void BuildRectangularMooreEdges(Graph graph)
        {
            var nodes = graph.Nodes;
            int w = nodes.Length;
            int h = nodes[0].Length;

            graph.ComputeEdgesHoriz = new float[(w + 1) * h];
            graph.ComputeEdgesVert = new float[w * (h + 1)];
            graph.ComputeEdgesLeftDiag = new float[(w + 1) * (h + 1)];
            graph.ComputeEdgesRightDiag = new float[(w + 1) * (h + 1)];

            graph.ComputeEdgesLeftDiag[0] = float.PositiveInfinity;
            graph.ComputeEdgesLeftDiag[graph.ComputeEdgesLeftDiag.Length - 1] = float.PositiveInfinity;

            graph.ComputeEdgesRightDiag[0] = float.PositiveInfinity;
            graph.ComputeEdgesRightDiag[graph.ComputeEdgesRightDiag.Length - 1] = float.PositiveInfinity;

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    BuildEdgesAround(graph, nodes, i, j, MooreIndexShifts, true);

                }
            }
        }

        private void BuildHexagonalNodes(Graph graph)
        {
            float cos30 = Mathf.Cos(Mathf.PI / 6);

            Vector2 min = bounds.Min + new Vector2(0, 0.5f);
            int w = Mathf.CeilToInt(bounds.Size.x / (step * cos30));
            int h = Mathf.CeilToInt(bounds.Size.y / step);

            var nodes = new Node[w][];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    float evenOddShift = i % 2 == 0 ? 0 : -0.5f;
                    var position = min + new Vector2(i * cos30, j + evenOddShift) * step;
                    if (MapObject.IsPointWalkable(position))
                    {
                        nodes[i][j] = new Node(position);
                    }
                }
            }

            graph.Nodes = nodes;
        }

        private void BuildHexagonalEdges(Graph graph)
        {
            var nodes = graph.Nodes;

            int w = nodes.Length;
            int h = nodes[0].Length;

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    (int di, int dj)[] shifts = i % 2 == 0
                                                    ? HexagonalIndexShiftsEven
                                                    : HexagonalIndexShiftsOdd;

                    BuildEdgesAround(graph, nodes, i, j, shifts, false);
                }
            }
        }

        private void BuildEdgesAround(Graph graph, Node[][] nodes, int i, int j, (int di, int dj)[] shifts, bool mulByDistance)
        {
            int w = nodes.Length;
            int h = nodes[0].Length;

            Node node = nodes[i][j];
            if (node == null)
            { 
                foreach (var shift in shifts)
                {
                    GetComputeEdge(shift) = float.PositiveInfinity;
                }
                return;
            }


            foreach (var shift in shifts)
            {
                ref float computeEdge = ref GetComputeEdge(shift);

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
            
            ref float GetComputeEdge((int di, int dj) shift)
            {
                //int newI = i + shift.di;
                //int newJ = j + shift.dj;
                int di = shift.di;
                int dj = shift.dj;
                
                if (di == -1 && dj == -1) return ref graph.ComputeEdgesLeftDiag[i + j * (w + 1)];
                if (di == 0 && dj == -1) return ref graph.ComputeEdgesVert[i + j * w];
                if (di == 1 && dj == -1) return ref graph.ComputeEdgesRightDiag[i + 1 + j * (w + 1)];

                if (di == -1 && dj == 0) return ref graph.ComputeEdgesHoriz[i + j * (w + 1)];
                if (di == 1 && dj == 0) return ref graph.ComputeEdgesHoriz[i + 1 + j * (w + 1)];

                if (di == -1 && dj == 1) return ref graph.ComputeEdgesRightDiag[i + (j + 1) * (w + 1)];
                if (di == 0 && dj == 1) return ref graph.ComputeEdgesVert[i + (j + 1) * w];
                if (di == 1 && dj == 1) return ref graph.ComputeEdgesLeftDiag[i + 1 + (j + 1) * (w + 1)];

                throw new Exception("Invalid rectangular moore shift");
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