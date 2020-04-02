using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Priority_Queue;
using System.Runtime.CompilerServices;

public class PathFinder
{
    public static Node[] FindPath(Graph graph, Node start, Node goal)
    {
        var open = new FastPriorityQueue<Node>(graph.Nodes.Count);

        start.G = 0;
        start.F = HeuristicCost(start, goal);
        open.Enqueue(start, start.F);

        while (open.Count != 0)
        {
            Node current = open.Dequeue();
            if (current == goal)
            {
                Node[] path = ReconstructPath(current);
                Cleanup(open, graph);
                return path;
            }

            current.IsClosed = true;
            foreach (var edge in current.IncidentEdges)
            {
                Node successor = edge.GetOppositeNode(current);
                float tentativeSuccessorG = current.G + edge.Weight;
                if (tentativeSuccessorG < successor.G)
                {
                    successor.CameFrom = current;
                    successor.G = tentativeSuccessorG;
                    successor.F = tentativeSuccessorG + HeuristicCost(successor, goal);
                    if (!successor.IsClosed)
                        open.Enqueue(successor, successor.F);
                }
            }
        }

        Cleanup(open, graph);
        return null;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HeuristicCost(Node node, Node goal)
    {
        return Vector2.Distance(node.Position, goal.Position);
    }

    private static void Cleanup(FastPriorityQueue<Node> open, Graph graph)
    {
        open.Clear();
        foreach (var node in graph.Nodes)
        {
            node.IsClosed = false;
            node.G = float.PositiveInfinity;
            node.F = float.PositiveInfinity;
            node.CameFrom = null;
        }
    }

    private static Node[] ReconstructPath(Node current)
    {
        var pathReversed = new List<Node>();
        while (current != null)
        {
            pathReversed.Add(current);
            current = current.CameFrom;
        }
        pathReversed.Reverse();
        return pathReversed.ToArray();
    }
}
