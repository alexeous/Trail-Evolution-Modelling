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


        //float bound = HeuristicCost(start.Position, goal.Position);
        //var path = new Stack<Node>();
        //path.Push(start);

        //while (true)
        //{
        //    float t = Search(path, goal, 0, bound);

        //    if (t == Found)
        //    {
        //        return path.ToArray();
        //    }
        //    if (t == float.PositiveInfinity)
        //    {
        //        return null;
        //    }

        //    bound = t;
        //}
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HeuristicCost(Node node, Node goal)
    {
        return Vector2.Distance(node.Position, goal.Position);

        //float dx = Mathf.Abs(nodePos.x - goalPos.x);
        //float dy = Mathf.Abs(nodePos.y - goalPos.y);
        //return dx + dy + (Sqrt2 - 2) * Mathf.Min(dx, dy);
    }



    //private static float Search(Stack<Node> path, Node goal, float currentCost, float bound)
    //{
    //    Node node = path.Peek();
    //    float f = currentCost + HeuristicCost(node.Position, goal.Position);

    //    if (f > bound)
    //    {
    //        return f;
    //    }
    //    if (node == goal)
    //    {
    //        return Found;
    //    }

    //    float min = float.PositiveInfinity;
    //    List<Edge> incidentEdges = node.IncidentEdges;
    //    for (int i = 0; i < incidentEdges.Count; i++)
    //    {
    //        Edge edge = incidentEdges[i];
    //        Node successor = edge.GetOppositeNode(node);

    //        if (path.Contains(successor)) continue;

    //        path.Push(successor);

    //        float t = Search(path, goal, currentCost + edge.Weight, bound);
    //        if (t == Found)
    //        {
    //            return Found;
    //        }
    //        min = Mathf.Min(min, t);

    //        path.Pop();
    //    }
    //    return min;
    //}
}
