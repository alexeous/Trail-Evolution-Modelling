using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Priority_Queue;
using System.Runtime.CompilerServices;
using System.Linq.Expressions;

namespace TrailEvolutionModelling
{
    public class PathFinder
    {
        public static Node[] FindPath(Graph graph, Node start, Node goal, bool aStar = false)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = aStar ? AStar(graph, start, goal) : NBA(graph, start, goal);
            stopwatch.Stop();

            Debug.Log($"Path finding took {stopwatch.ElapsedMilliseconds} ms");
            return result;
        }

        private static Node[] AStar(Graph graph, Node start, Node goal)
        {
            var open = new FastPriorityQueue<Node>(graph.Nodes.Count);

            start.G1 = 0;
            start.F1 = Heuristic(start, goal, start);
            open.Enqueue(start, start.F1);

            while (open.Count != 0)
            {
                Node current = open.Dequeue();
                if (current == goal)
                {
                    Node[] path = ReconstructPath(current);
                    open.Clear();
                    CleanupGraph(graph);
                    return path;
                }

                current.IsClosed = true;
                foreach (var edge in current.IncidentEdges)
                {
                    Node successor = edge.GetOppositeNode(current);
                    float tentativeSuccessorG = current.G1 + edge.Weight;
                    if (tentativeSuccessorG < successor.G1)
                    {
                        successor.CameFrom1 = current;
                        successor.G1 = tentativeSuccessorG;
                        successor.F1 = tentativeSuccessorG + Heuristic(successor, goal, start);
                        if (!successor.IsClosed)
                            open.Enqueue(successor, successor.F1);
                    }
                }
            }

            open.Clear();
            CleanupGraph(graph);
            return null;


            Node[] ReconstructPath(Node current)
            {
                var pathReversed = new List<Node>();
                while (current != null)
                {
                    pathReversed.Add(current);
                    current = current.CameFrom1;
                }
                pathReversed.Reverse();
                return pathReversed.ToArray();
            }
        }

        private class F1Comparer : Comparer<Node>
        {
            public override int Compare(Node x, Node y) => x.F1.CompareTo(y.F1);
        }

        private class F2Comparer : Comparer<Node>
        {
            public override int Compare(Node x, Node y) => x.F2.CompareTo(y.F2);
        }

        private static Node[] NBA(Graph graph, Node start, Node goal)
        {
            var open1 = new Heap<Node>(new F1Comparer(), (node, id) => node.H1 = id);
            var open2 = new Heap<Node>(new F2Comparer(), (node, id) => node.H2 = id);

            Node minNode = null;
            float lMin = float.PositiveInfinity;

            start.G1 = 0;
            float f1 = Heuristic(start, goal, start);
            start.F1 = f1;
            open1.Add(start);

            goal.G2 = 0;
            float f2 = f1;
            goal.F2 = f2;
            open2.Add(goal);

            Node cameFrom;

            while (open1.Count > 0 && open2.Count > 0)
            {
                Search(forward: (open1.Count < open2.Count));

                //if (minNode != null)
                //    break;
            }

            Node[] path = ReconstructPath(minNode);

            CleanupGraph(graph);

            return path;

            void Search(bool forward)
            {
                var open = (forward ? open1 : open2);
                cameFrom = open.Pop();
                if (cameFrom.IsClosed)
                    return;

                cameFrom.IsClosed = true;

                float otherF = forward ? f2 : f1;
                float heuristic = forward ? Heuristic(start, cameFrom, start) : Heuristic(cameFrom, goal, start);
                if (cameFrom.F(forward) < lMin && (cameFrom.G(forward) + otherF - heuristic) < lMin)
                {
                    List<Edge> edges = cameFrom.IncidentEdges;
                    for (int i = 0; i < edges.Count; i++)
                    {
                        Visit(cameFrom, edges[i], open, forward);
                    }
                }

                if (open.Count > 0)
                {
                    if (forward)
                        f1 = open.Peek().F1;
                    else
                        f2 = open.Peek().F2;
                }
            }

            void Visit(Node thisNode, Edge edge, Heap<Node> open, bool forward)
            {
                Node other = edge.GetOppositeNode(thisNode);
                if (other.IsClosed)
                    return;

                float tentativeG = cameFrom.G(forward) + edge.Weight;
                ref float g = ref other.G(forward);
                if (tentativeG < g)
                {
                    g = tentativeG;
                    float heuristic = forward ? Heuristic(other, goal, start) : Heuristic(start, other, start);
                    other.F(forward) = tentativeG + heuristic;
                    other.CameFrom(forward) = cameFrom;

                    ref int h = ref other.H(forward);
                    if (h < 0)
                        open.Add(other);
                    else
                        open.UpdateItem(h);
                }

                float potentialMin = other.G1 + other.G2;
                if (potentialMin < lMin)
                {
                    lMin = potentialMin;
                    minNode = other;
                }
            }

            Node[] ReconstructPath(Node _minNode)
            {
                if (_minNode == null)
                    return null;

                var pathForward = new List<Node> { _minNode };
                var pathReversed = new List<Node>();

                Node parent = _minNode.CameFrom1;
                while (parent != null)
                {
                    pathForward.Add(parent);
                    parent = parent.CameFrom1;
                }

                Node child = _minNode.CameFrom2;
                while (child != null)
                {
                    pathReversed.Add(child);
                    child = child.CameFrom2;
                }

                var result = new Node[pathReversed.Count + pathForward.Count];
                int j = 0;
                for (int i = pathReversed.Count - 1; i >= 0; i--)
                    result[j++] = pathReversed[i];

                for (int i = 0; i < pathForward.Count; i++)
                    result[j++] = pathForward[i];

                return result;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Heuristic(Node node, Node goal, Node start)
        {
            Vector2 nodeToGoal = goal.Position - node.Position;
            //Vector2 startToNode = node.Position - start.Position;
            float h = nodeToGoal.magnitude;
            //return h;

            Node current = node?.CameFrom1;
            Node old = current?.CameFrom1 ?? current;
            for (int i = 0; i < 15; i++)
                old = old?.CameFrom1 ?? old;
            if (current != null)
            {
                Vector2 trend = current.Position - old.Position;
                Vector2 newDirection = node.Position - old.Position;
                float angleToGoal = Vector2.Angle(trend, nodeToGoal);
                float newAngleToGoal = Vector2.Angle(newDirection, nodeToGoal);
                if (newAngleToGoal < angleToGoal)
                {
                    h = 0;
                }
                else
                {
                    Vector2 nodeToStart = start.Position - node.Position;
                    float angleToStart = Vector2.Angle(-trend, nodeToStart);
                    float newAngleToStart = Vector2.Angle(-newDirection, nodeToStart);
                    if (newAngleToStart > angleToStart)
                    {
                        h = 0;
                    }
                }
            }

            //nodeToGoal.Normalize();
            //startToNode.Normalize();
            //float cross = nodeToGoal.x * startToNode.y - startToNode.x * nodeToGoal.y;
            //h -= h * (cross);

            ////Vector2 nodeToGoal = goal.Position - node.Position;
            ////float distanceToGoal = nodeToGoal.magnitude;

            ////Node old = current?.CameFrom;
            ////old = old?.CameFrom ?? old;
            ////old = old?.CameFrom ?? old;
            //Node current = node.CameFrom;
            //if (current != null)
            //{
            //    Vector2 currentToGoalDir = (current.Position - goal.Position).normalized;
            //    Vector2 currentToNodeDir = (current.Position - node.Position).normalized;
            //    float dot = Vector2.Dot(currentToGoalDir, currentToNodeDir);
            //    float v = 1 - (Mathf.Acos(Mathf.Max(0, dot)) / (Mathf.PI / 2));
            //    h -= h * v;
            //    //Vector2 currTrendDir = (current.Position - old.Position).normalized;
            //    //Vector2 newDir = (node.Position - current.Position).normalized;

            //    //Vector2 prevToNodeNorm = (node.Position - prev.Position).normalized;
            //    //float prevToNodeDot = Vector2.Dot(prevToNodeNorm, nodeToGoal / distanceToGoal);
            //    //prevToNodeDot = Mathf.Max(0, prevToNodeDot);
            //    //h -= h * prevToNodeDot;
            //    //h = Math.Max(0, h - 100f * Mathf.Max(0, dot));
            //}

            return h;
        }

        private static void CleanupGraph(Graph graph)
        {
            foreach (var node in graph.Nodes)
            {
                node.CleanupAfterPathSearch();
            }
        }
    }
}