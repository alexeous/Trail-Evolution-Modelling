using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
    private void Build()
    {
        if (bounds == null || target == null)
            return;

        //target.Graph = BuildRectangularMoore();
        target.Graph = BuildHexagonal();
    }

    private Graph BuildRectangularMoore()
    {
        var graph = new Graph();
        Node[,] nodes = BuildRectangularMooreNodes(graph);
        BuildRectangularMooreEdges(nodes, graph);
        return graph;
    }

    private Graph BuildHexagonal()
    {
        var graph = new Graph();
        Node[,] nodes = BuildHexagonalNodes(graph);
        BuildHexagonalEdges(nodes, graph);
        return graph;
    }

    private Node[,] BuildRectangularMooreNodes(Graph graph)
    {
        Vector2 min = bounds.Min;
        int w = (int)(bounds.Size.x / step);
        int h = (int)(bounds.Size.y / step);

        var nodes = new Node[w, h];
        
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                var position = min + new Vector2(i, j) * step;
                if (MapObject.IsPointWalkable(position))
                {
                    nodes[i, j] = graph.AddNode(position);
                }
            }
        }

        return nodes;
    }

    private void BuildRectangularMooreEdges(Node[,] nodes, Graph graph)
    {
        int w = nodes.GetLength(0);
        int h = nodes.GetLength(1);

        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                BuildEdgesAround(graph, nodes, i, j, MooreIndexShifts);
            }
        }
    }

    private Node[,] BuildHexagonalNodes(Graph graph)
    {
        float cos30 = Mathf.Cos(Mathf.PI / 6);

        Vector2 min = bounds.Min + new Vector2(0, 0.5f);
        int w = (int)(bounds.Size.x / step * cos30);
        int h = (int)(bounds.Size.y / step) + 1;

        var nodes = new Node[w, h];

        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                float evenOddShift = i % 2 == 0 ? 0 : -0.5f;
                var position = min + new Vector2(i * cos30, j + evenOddShift) * step;
                if (MapObject.IsPointWalkable(position))
                {
                    nodes[i, j] = graph.AddNode(position);
                }
            }
        }

        return nodes;
    }

    private void BuildHexagonalEdges(Node[,] nodes, Graph graph)
    {
        int w = nodes.GetLength(0);
        int h = nodes.GetLength(1);

        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                (int di, int dj)[] shifts = i % 2 == 0
                                                ? HexagonalIndexShiftsEven
                                                : HexagonalIndexShiftsOdd;

                BuildEdgesAround(graph, nodes, i, j, shifts);
            }
        }
    }

    private static void BuildEdgesAround(Graph graph, Node[,] nodes, int i, int j, (int di, int dj)[] shifts)
    {
        Node node = nodes[i, j];
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

            graph.AddEdge(node, otherNode, areaAttributes.Weight, areaAttributes.IsTramplable);
        }
    }

    private static Node GetNodeAtOrNull(Node[,] nodes, int i, int j)
    {
        int w = nodes.GetLength(0);
        int h = nodes.GetLength(1);
        if (i < 0 || j < 0 || i >= w || j >= h)
            return null;
        return nodes[i, j];
    }
}
