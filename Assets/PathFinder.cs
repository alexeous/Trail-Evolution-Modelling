using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathFinder
{
    private static readonly float Sqrt2 = Mathf.Sqrt(2);
    
    private const float Found = 0;

    public static Node[] FindPath(Node start, Node goal)
    {
        //return null;
        //var open = new SortedDictionary<float, Node>();
        //open.Add()

        float bound = HeuristicCost(start.Position, goal.Position);
        var path = new Stack<Node>();
        path.Push(start);

        while (true)
        {
            float t = Search(path, goal, 0, bound);

            if (t == Found)
            {
                return path.ToArray();
            }
            if (t == float.PositiveInfinity)
            {
                return null;
            }
            
            bound = t;
        }
    }

    private static float HeuristicCost(in Vector2 nodePos, in Vector2 goalPos)
    {
        return Vector2.Distance(nodePos, goalPos);

        //float dx = Mathf.Abs(nodePos.x - goalPos.x);
        //float dy = Mathf.Abs(nodePos.y - goalPos.y);
        //return dx + dy + (Sqrt2 - 2) * Mathf.Min(dx, dy);
    }
    
    private static float Search(Stack<Node> path, Node goal, float currentCost, float bound)
    {
        Node node = path.Peek();
        float f = currentCost + HeuristicCost(node.Position, goal.Position);
        
        if (f > bound)
        {
            return f;
        }
        if (node == goal)
        {
            return Found;
        }

        float min = float.PositiveInfinity;
        List<Edge> incidentEdges = node.IncidentEdges;
        for (int i = 0; i < incidentEdges.Count; i++)
        {
            Edge edge = incidentEdges[i];
            Node successor = edge.GetOppositeNode(node);
            
            if (path.Contains(successor)) continue;

            path.Push(successor);
            
            float t = Search(path, goal, currentCost + edge.Weight, bound);
            if (t == Found)
            {
                return Found;
            }
            min = Mathf.Min(min, t);

            path.Pop();
        }
        return min;
    }
}
