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
        }

        private void BuildRectangularMooreEdges(Graph graph)
        {
            var nodes = graph.Nodes;
            int w = nodes.Length;
            int h = nodes[0].Length;

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
            Node node = nodes[i][j];
            if (node == null)
                return;

            foreach (var shift in shifts)
            {
                Node otherNode = GetNodeAtOrNull(nodes, i + shift.di, j + shift.dj);
                if (otherNode == null)
                    continue;

                AreaAttributes areaAttributes = MapObject.GetAreaAttributes(node.Position, otherNode.Position);
                if (!areaAttributes.IsWalkable)
                    continue;

                float weight = areaAttributes.Weight;
                if (mulByDistance)
                    weight *= Vector2.Distance(node.Position, otherNode.Position) / step;
                graph.AddEdge(node, otherNode, weight, areaAttributes.IsTramplable);
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