using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Priority_Queue;
using UnityEngine;

namespace TrailEvolutionModelling
{
    public enum PathFindingAlgorithm { AStar, NBA, Wavefront }

    public class PathFinder
    {
        public static Node[] FindPath(Graph graph, Node start, Node goal, PathFindingAlgorithm algorithm)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            Node[] result = null;
            switch (algorithm)
            {
                case PathFindingAlgorithm.AStar:
                    result = AStar(graph, start, goal);
                    break;
                case PathFindingAlgorithm.NBA:
                    result = NBA(graph, start, goal);
                    break;
                case PathFindingAlgorithm.Wavefront:
                    result = Wavefront(graph, start, goal);
                    break;
            };
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

        public class YoureNotGonnaRecognizeThisThingAtTheFirstGlance
        {
            public
            static
            YoureNotGonnaRecognizeThisThingAtTheFirstGlance
            Yeah
            (
                object o
            )
            {
                Console.WriteLine("I am the best");
                return null;
            }
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
        public static List<List<Vector2>> gg;
        private static Node[] Wavefront(Graph graph, Node start, Node goal)
        {
            foreach (var node in graph.Nodes)
            {
                node.G1 = -1;
                node.G2 = -1;
                node.CameFrom1 = null;
            }
            goal.G1 = 0;
            start.G2 = 0;

            bool exitFlag = false;
            int iters = 0;
            while (!exitFlag)
            {
                exitFlag = true;
                Parallel.ForEach(graph.Nodes, PlannerKernel);
                if (start.G1 == -1 || goal.G2 == -1)
                    exitFlag = false;
                Parallel.ForEach(graph.Nodes, node =>
                {
                    node.G1 = node.F1;
                    node.G2 = node.F2;
                });
            }
            Node[] path = ReconstructPath(start);
            RedrawHeatmap(graph);

            Debug.Log("Iterations: " + iters);

            CleanupGraph(graph);
            var visitedEdges = new HashSet<Edge>();
            foreach (var node in path)
            {
                foreach (var edge in node.IncidentEdges)
                {
                    if (visitedEdges.Contains(edge) || !edge.IsTramplable)
                        continue;

                    edge.Weight = Mathf.Max(1.1f, edge.Weight - 0.4f);
                    visitedEdges.Add(edge);

                    Node other = edge.GetOppositeNode(node);
                    foreach (var edge2 in other.IncidentEdges)
                    {
                        if (visitedEdges.Contains(edge2) || !edge2.IsTramplable)
                            continue;

                        edge2.Weight = Mathf.Max(1.1f, edge2.Weight - 0.4f);
                        visitedEdges.Add(edge2);
                    }
                }
            }
            return path;

            void PlannerKernel(Node node)
            {
                Interlocked.Increment(ref iters);
                node.F1 = node.G1;
                if (node != goal)
                    foreach (var edge in node.IncidentEdges)
                    {
                        Node other = edge.GetOppositeNode(node);
                        var newG = other.G1 + edge.Weight;
                        if (other.G1 != -1 && (node.F1 == -1 || newG < node.F1))
                        {
                            node.CameFrom1 = other;
                            node.F1 = newG;
                            if (node.F1 < /*precalculated max of*/ start.G1)
                                exitFlag = false;
                        }
                    }

                node.F2 = node.G2;
                if (node != start)
                    foreach (var edge in node.IncidentEdges)
                    {
                        Node other = edge.GetOppositeNode(node);
                        var newG = other.G2 + edge.Weight;
                        if (other.G2 != -1 && (node.F2 == -1 || newG < node.F2))
                        {
                            node.CameFrom2 = other;
                            node.F2 = newG;
                            if (node.F2 < /*precalculated max of*/ start.G2)
                                exitFlag = false;
                        }
                    }
            }

            Node[] ReconstructPath(Node current)
            {
                if (current == goal)
                {
                    return new Node[] { current };
                }

                gg = new List<List<Vector2>>();

                var pathNodes = new List<Node> { current };
                Node prevGuide = current;
                Node guide = current.CameFrom1;
                while (guide != goal)
                {
                    if (pathNodes.Count > 100000)
                    {
                        CleanupGraph(graph);
                        throw new Exception("Something went wrong");
                    }

                    Node nextGuide = guide.CameFrom1;

                    float minG1 = nextGuide.G1;
                    float maxG1 = prevGuide.G1;
                    float minG2 = prevGuide.G2;
                    float maxG2 = nextGuide.G2;

                    var visited = new HashSet<Node> { guide };
                    var similarCostNodes = new List<Node> { guide };
                    Vector2 averagePos = guide.Position;

                    Rec(guide);
                    void Rec(Node center)
                    {
                        foreach (var edge in center.IncidentEdges)
                        {
                            Node other = edge.GetOppositeNode(center);
                            if (visited.Contains(other))
                                continue;

                            visited.Add(other);

                            if (other.G1 >= minG1 && other.G1 <= maxG1 &&
                                other.G2 >= minG2 && other.G2 <= maxG2)
                            {
                                similarCostNodes.Add(other);
                                averagePos += other.Position;
                                Rec(other);
                            }
                        }
                    }

                    gg.Add(similarCostNodes.Select(n => n.Position).ToList());

                    averagePos /= similarCostNodes.Count;
                    Node next = null;
                    float minSqrDist = float.PositiveInfinity;
                    foreach (var node in similarCostNodes)
                    {
                        float sqrDist = (averagePos - node.Position).sqrMagnitude;
                        if (sqrDist < minSqrDist)
                        {
                            minSqrDist = sqrDist;
                            next = node;
                        }
                    }
                    pathNodes.Add(next);

                    prevGuide = guide;
                    guide = guide.CameFrom1;
                }
                pathNodes.Add(goal);

                //var pathNodes = new List<Node>();
                //while (current != null)
                //{
                //    pathNodes.Add(current);
                //    if (pathNodes.Count > 100000)
                //    {
                //        CleanupGraph(graph);
                //        throw new Exception("Something went wrong");
                //    }
                //    current = current.CameFrom1;
                //}

                //Vector2 deltaPos = graph.Nodes[0].Position - graph.Nodes[1].Position;
                //float cellStep = Mathf.Max(Mathf.Abs(deltaPos.x), Mathf.Abs(deltaPos.y));

                //while (true)
                //{
                //    pathNodes.Add(current);
                //    if (pathNodes.Count > 100000)
                //    {
                //        pathNodes.RemoveRange(100, pathNodes.Count - 100);
                //        CleanupGraph(graph);
                //        Debug.LogError("Something went wrong");
                //        break;
                //    }
                //    if (current == goal)
                //        break;

                //    Vector2 vector = Vector2.zero;

                //    var visited = new HashSet<Node>();
                //    float radius = cellStep * 4;
                //    Node next = null;
                //    Rec(current, 0);

                //    void Rec(Node node, float prevSqrDist)
                //    {
                //        foreach (var edge2 in node.IncidentEdges)
                //        {
                //            Node other2 = edge2.GetOppositeNode(node);
                //            if (visited.Contains(other2))
                //                continue;

                //            if (other2 == goal)
                //            {
                //                next = other2;
                //                return;
                //            }

                //            visited.Add(other2);
                //            if (other2.CameFrom1 == null)
                //                continue;

                //            float sqrDistFromCenter = (other2.Position - current.Position).sqrMagnitude;
                //            if (sqrDistFromCenter > radius * radius || sqrDistFromCenter <= prevSqrDist)
                //                continue;

                //            float t = Mathf.Exp(-sqrDistFromCenter / (radius * radius) / 2);
                //            t /= Mathf.Sqrt(2 * Mathf.PI);
                //            vector += t * (other2.CameFrom1.Position - other2.Position).normalized;

                //            Rec(other2, sqrDistFromCenter);

                //            if (next != null)
                //                return;
                //        }
                //    }

                //    vector.Normalize();

                //    float maxDot = float.NegativeInfinity;
                //    foreach (var edge in current.IncidentEdges)
                //    {
                //        Node other = edge.GetOppositeNode(current);
                //        Vector2 dir = (other.Position - current.Position).normalized;
                //        float dot = Vector2.Dot(dir, vector);
                //        if (dot > maxDot)
                //        {
                //            maxDot = dot;
                //            next = other;
                //        }
                //    }
                //    //float min = float.PositiveInfinity;
                //    //foreach (var edge in current.IncidentEdges)
                //    //{
                //    //    Node other = edge.GetOppositeNode(current);
                //    //    if (other.G1 != -1 && other.G1 < min)
                //    //    {
                //    //        next = other;
                //    //        min = other.G1;
                //    //    }
                //    //}
                //    current = next;
                //}
                return pathNodes.ToArray();
            }
        }

        static void RedrawHeatmap(Graph graph)
        {
            float minG = graph.Nodes.Min(n => n.G1);
            float maxG = graph.Nodes.Max(n => n.G1);
            Color minColor = Color.green;
            Color maxColor = Color.red;
            Vector2 deltaPos = graph.Nodes[0].Position - graph.Nodes[1].Position;
            float step = Mathf.Max(Mathf.Abs(deltaPos.x), Mathf.Abs(deltaPos.y));
            float halfStep = step / 2;

            var mesh = new Mesh();
            var vertices = new List<Vector3>();
            var colors = new List<Color>();
            foreach (var node in graph.Nodes)
            {
                float t = (node.G1 - minG) / (maxG - minG);
                var color = node.G1 == -1 ? Color.blue : Color.Lerp(minColor, maxColor, t);
                vertices.Add(node.Position + new Vector2(-halfStep, -halfStep));
                vertices.Add(node.Position + new Vector2(+halfStep, -halfStep));
                vertices.Add(node.Position + new Vector2(+halfStep, +halfStep));
                vertices.Add(node.Position + new Vector2(-halfStep, +halfStep));

                colors.Add(color);
                colors.Add(color);
                colors.Add(color);
                colors.Add(color);
            }
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            var indices = Enumerable.Range(0, vertices.Count).ToArray();
            mesh.SetIndices(indices, MeshTopology.Quads, 0);

            var old = GameObject.Find("HEATMAP");
            if (old != null)
            {
                GameObject.DestroyImmediate(old);
            }
            var go = new GameObject("HEATMAP");
            go.transform.position = new Vector3(0, 0, -0.95f);
            var meshFilter = go.AddComponent<MeshFilter>();
            meshFilter.sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void AtomicMin(ref float ptr, float value)
        {
            float oldVal, newVal;
            do
            {
                oldVal = ptr;
                newVal = Mathf.Min(oldVal, value);
            } while (Interlocked.CompareExchange(ref ptr, oldVal, newVal) != oldVal);
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