using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class GraphBuilder : MonoBehaviour
{
    [SerializeField] WorkingBounds bounds = null;
    [SerializeField] GraphHolder target = null;
    [SerializeField] float nodeStep = 0.5f;

    private static (int di, int dj)[] MooreIndexShifts =
    {
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    };

    private void OnValidate()
    {
        if (nodeStep < 0.25f)
        {
            nodeStep = 0.25f;
        }
    }

    [ContextMenu("Build")]
    private void Build()
    {
        if (bounds == null || target == null)
            return;

        var graph = new Graph();

        Node[,] nodes = BuildNodes(graph);
        BuildEdges(nodes, graph);

        target.Graph = graph;
    }

    private Node[,] BuildNodes(Graph graph)
    {
        Vector2 min = bounds.Min;
        int w = (int)(bounds.Size.x / nodeStep);
        int h = (int)(bounds.Size.y / nodeStep);

        var nodes = new Node[w, h];
        
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                var position = min + new Vector2(i, j) * nodeStep;
                if (MapObject.IsPointWalkable(position))
                {
                    Node node = graph.AddNode(position);
                    nodes[i, j] = node;
                }
            }
        }

        return nodes;
    }

    private void BuildEdges(Node[,] nodes, Graph graph)
    {
        int w = nodes.GetLength(0);
        int h = nodes.GetLength(1);

        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                Node node = nodes[i, j];

                if (node == null)
                {
                    continue;
                }

                foreach ((int di, int dj) shift in MooreIndexShifts) 
                {
                    Node otherNode = GetNodeAtOrNull(i + shift.di, j + shift.dj);
                    if (otherNode == null)
                    {
                        continue;
                    }
                    
                    AreaAttributes areaAttributes = MapObject.GetAreaAttributes(node.Position, otherNode.Position);
                    if (!areaAttributes.IsWalkable)
                    {
                        continue;
                    }

                    graph.AddEdge(node, otherNode, areaAttributes.Weight, areaAttributes.IsTramplable);
                }
            }
        }

        Node GetNodeAtOrNull(int i, int j)
        {
            if (i < 0 || j < 0 || i >= w || j >= h)
                return null;
            return nodes[i, j];
        }
    }
}
