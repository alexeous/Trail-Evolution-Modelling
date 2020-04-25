//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.Linq;
//using System.Runtime.CompilerServices;
//using System.Threading;
//using System.Threading.Tasks;
//using Priority_Queue;
//using UnityEngine;

//using Debug = UnityEngine.Debug;

//namespace TrailEvolutionModelling
//{
//    public enum PathFindingAlgorithm { AStar, NBA, Wavefront, WavefrontParallel }

//    public class PathFinder
//    {
//        public static Node[] FindPath(Graph graph, Node start, Node goal, PathFindingAlgorithm algorithm)
//        {
//            Node[] result = null;
//            switch (algorithm)
//            {
//                case PathFindingAlgorithm.AStar:
//                    result = AStar(graph, start, goal);
//                    break;
//                case PathFindingAlgorithm.NBA:
//                    result = NBA(graph, start, goal);
//                    break;
//                case PathFindingAlgorithm.Wavefront:
//                    result = Wavefront(graph, start, goal);
//                    break;
//            };
//            return result;
//        }

//        private static Node[] AStar(Graph graph, Node start, Node goal)
//        {
//            Node[] nodesFlattened = graph.Nodes.SelectMany(t => t).Where(t => t != null).ToArray();
//            var open = new FastPriorityQueue<Node>(graph.Nodes.Length * graph.Nodes[0].Length);

//            start.G1 = 0;
//            start.F1 = Heuristic(start, goal, start);
//            open.Enqueue(start, start.F1);

//            while (open.Count != 0)
//            {
//                Node current = open.Dequeue();
//                if (current == goal)
//                {
//                    Node[] path = ReconstructPath(current);
//                    open.Clear();
//                    CleanupGraph(nodesFlattened);
//                    return path;
//                }

//                current.IsClosed = true;
//                foreach (var edge in current.IncidentEdges)
//                {
//                    Node successor = edge.GetOppositeNode(current);
//                    float tentativeSuccessorG = current.G1 + edge.Weight;
//                    if (tentativeSuccessorG < successor.G1)
//                    {
//                        successor.CameFrom1 = current;
//                        successor.G1 = tentativeSuccessorG;
//                        successor.F1 = tentativeSuccessorG + Heuristic(successor, goal, start);
//                        if (!successor.IsClosed)
//                            open.Enqueue(successor, successor.F1);
//                    }
//                }
//            }

//            open.Clear();
//            CleanupGraph(nodesFlattened);
//            return null;


//            Node[] ReconstructPath(Node current)
//            {
//                var pathReversed = new List<Node>();
//                while (current != null)
//                {
//                    pathReversed.Add(current);
//                    current = current.CameFrom1;
//                }
//                pathReversed.Reverse();
//                return pathReversed.ToArray();
//            }
//        }

//        private class F1Comparer : Comparer<Node>
//        {
//            public override int Compare(Node x, Node y) => x.F1.CompareTo(y.F1);
//        }

//        private class F2Comparer : Comparer<Node>
//        {
//            public override int Compare(Node x, Node y) => x.F2.CompareTo(y.F2);
//        }

//        private static Node[] NBA(Graph graph, Node start, Node goal)
//        {
//            Node[] nodesFlattened = graph.Nodes.SelectMany(t => t).Where(t => t != null).ToArray();
//            var open1 = new Heap<Node>(new F1Comparer(), (node, id) => node.H1 = id);
//            var open2 = new Heap<Node>(new F2Comparer(), (node, id) => node.H2 = id);

//            Node minNode = null;
//            float lMin = float.PositiveInfinity;

//            start.G1 = 0;
//            float f1 = Heuristic(start, goal, start);
//            start.F1 = f1;
//            open1.Add(start);

//            goal.G2 = 0;
//            float f2 = f1;
//            goal.F2 = f2;
//            open2.Add(goal);

//            Node cameFrom;

//            while (open1.Count > 0 && open2.Count > 0)
//            {
//                Search(forward: (open1.Count < open2.Count));

//                //if (minNode != null)
//                //    break;
//            }

//            Node[] path = ReconstructPath(minNode);
            
//            CleanupGraph(nodesFlattened);

//            return path;

//            void Search(bool forward)
//            {
//                var open = (forward ? open1 : open2);
//                cameFrom = open.Pop();
//                if (cameFrom.IsClosed)
//                    return;

//                cameFrom.IsClosed = true;

//                float otherF = forward ? f2 : f1;
//                float heuristic = forward ? Heuristic(start, cameFrom, start) : Heuristic(cameFrom, goal, start);
//                if (cameFrom.F(forward) < lMin && (cameFrom.G(forward) + otherF - heuristic) < lMin)
//                {
//                    List<Edge> edges = cameFrom.IncidentEdges;
//                    for (int i = 0; i < edges.Count; i++)
//                    {
//                        Visit(cameFrom, edges[i], open, forward);
//                    }
//                }

//                if (open.Count > 0)
//                {
//                    if (forward)
//                        f1 = open.Peek().F1;
//                    else
//                        f2 = open.Peek().F2;
//                }
//            }

//            void Visit(Node thisNode, Edge edge, Heap<Node> open, bool forward)
//            {
//                Node other = edge.GetOppositeNode(thisNode);
//                if (other.IsClosed)
//                    return;

//                float tentativeG = cameFrom.G(forward) + edge.Weight;
//                ref float g = ref other.G(forward);
//                if (tentativeG < g)
//                {
//                    g = tentativeG;
//                    float heuristic = forward ? Heuristic(other, goal, start) : Heuristic(start, other, start);
//                    other.F(forward) = tentativeG + heuristic;
//                    other.CameFrom(forward) = cameFrom;

//                    ref int h = ref other.H(forward);
//                    if (h < 0)
//                        open.Add(other);
//                    else
//                        open.UpdateItem(h);
//                }

//                float potentialMin = other.G1 + other.G2;
//                if (potentialMin < lMin)
//                {
//                    lMin = potentialMin;
//                    minNode = other;
//                }
//            }

//            Node[] ReconstructPath(Node _minNode)
//            {
//                if (_minNode == null)
//                    return null;

//                var pathForward = new List<Node> { _minNode };
//                var pathReversed = new List<Node>();

//                Node parent = _minNode.CameFrom1;
//                while (parent != null)
//                {
//                    pathForward.Add(parent);
//                    parent = parent.CameFrom1;
//                }

//                Node child = _minNode.CameFrom2;
//                while (child != null)
//                {
//                    pathReversed.Add(child);
//                    child = child.CameFrom2;
//                }

//                var result = new Node[pathReversed.Count + pathForward.Count];
//                int j = 0;
//                for (int i = pathReversed.Count - 1; i >= 0; i--)
//                    result[j++] = pathReversed[i];

//                for (int i = 0; i < pathForward.Count; i++)
//                    result[j++] = pathForward[i];

//                return result;
//            }
//        }
//        public static List<List<Vector2>> gg;
//        private static Node[] Wavefront(Graph graph, Node start, Node goal)
//        {
//            Node[] nodesFlattened = graph.Nodes.SelectMany(t => t).Where(t => t != null).ToArray();
//            foreach (var node in nodesFlattened)
//            {
//                node.G1 = -1;
//                node.G2 = -1;
//                node.CameFrom1 = null;
//            }
//            goal.G1 = 0;
//            start.G2 = 0;

//            bool exitFlag = false;
//            int iters = 0;
//            while (!exitFlag)
//            {
//                exitFlag = true;
//                Parallel.ForEach(nodesFlattened, PlannerKernel);
//                if (start.G1 == -1 || goal.G2 == -1)
//                    exitFlag = false;
//                Parallel.ForEach(nodesFlattened, node =>
//                {
//                    node.G1 = node.F1;
//                    node.G2 = node.F2;
//                });
//                iters++;
//            }
//            var stopwatch = Stopwatch.StartNew();
//            Node[] path = ReconstructPath(start);
//            stopwatch.Stop();
//            Debug.Log("Path reconstruction took: " + stopwatch.ElapsedMilliseconds + " ms");
//            RedrawHeatmap(nodesFlattened);

//            Debug.Log("Iterations: " + iters);
            
//            CleanupGraph(nodesFlattened);
//            var visitedEdges = new HashSet<Edge>();
//            foreach (var node in path)
//            {
//                foreach (var edge in node.IncidentEdges)
//                {
//                    if (visitedEdges.Contains(edge) || !edge.IsTramplable)
//                        continue;

//                    edge.Weight = Mathf.Max(1.1f, edge.Weight - 0.4f);
//                    visitedEdges.Add(edge);

//                    Node other = edge.GetOppositeNode(node);
//                    foreach (var edge2 in other.IncidentEdges)
//                    {
//                        if (visitedEdges.Contains(edge2) || !edge2.IsTramplable)
//                            continue;

//                        edge2.Weight = Mathf.Max(1.1f, edge2.Weight - 0.2f);
//                        visitedEdges.Add(edge2);
//                    }
//                }
//            }
//            return path;

//            void PlannerKernel(Node node)
//            {
//                node.F1 = node.G1;
//                if (node != goal)
//                    foreach (var edge in node.IncidentEdges)
//                    {
//                        Node other = edge.GetOppositeNode(node);
//                        var newG = other.G1 + edge.Weight;
//                        if (other.G1 != -1 && (node.F1 == -1 || newG < node.F1))
//                        {
//                            node.CameFrom1 = other;
//                            node.F1 = newG;
//                            if (node.F1 < /*precalculated max of*/ start.G1)
//                                exitFlag = false;
//                        }
//                    }

//                node.F2 = node.G2;
//                if (node != start)
//                    foreach (var edge in node.IncidentEdges)
//                    {
//                        Node other = edge.GetOppositeNode(node);
//                        var newG = other.G2 + edge.Weight;
//                        if (other.G2 != -1 && (node.F2 == -1 || newG < node.F2))
//                        {
//                            node.CameFrom2 = other;
//                            node.F2 = newG;
//                            if (node.F2 < /*precalculated max of*/ start.G2)
//                                exitFlag = false;
//                        }
//                    }
//            }

//            Node[] ReconstructPath(Node current)
//            {
//                if (current == goal)
//                {
//                    return new Node[] { current };
//                }

//                gg = new List<List<Vector2>>();
//                var (prevI, prevJ) = (current.ComputeIndexI - 1, current.ComputeIndexJ - 1);
//                var pathNodes = new List<Node> { current };
//                Node prevGuide = current;
//                Node guide = current.CameFrom1;
//                while (guide != goal)
//                {
//                    if (pathNodes.Count > 100000)
//                    {
//                        CleanupGraph(nodesFlattened);
//                        throw new Exception("Something went wrong");
//                    }

//                    Node nextGuide = guide.CameFrom1;

//                    float minG1 = nextGuide.G1;
//                    float maxG1 = prevGuide.G1;
//                    float minG2 = prevGuide.G2;
//                    float maxG2 = nextGuide.G2;

//                    var visited = new HashSet<Node> { guide };
//                    var similarCostNodes = new List<Node> { guide };
//                    Vector2 averagePos = guide.Position;

//                    Rec(guide);
//                    void Rec(Node center)
//                    {
//                        foreach (var edge in center.IncidentEdges)
//                        {
//                            Node other = edge.GetOppositeNode(center);
//                            if (visited.Contains(other))
//                                continue;

//                            visited.Add(other);

//                            if (other.G1 > minG1 && other.G1 < maxG1 &&
//                                other.G2 > minG2 && other.G2 < maxG2)
//                            {
//                                similarCostNodes.Add(other);
//                                averagePos += other.Position;
//                                Rec(other);
//                            }
//                        }
//                    }

//                    gg.Add(similarCostNodes.Select(n => n.Position).ToList());

//                    averagePos /= similarCostNodes.Count;
//                    Node next = null;
//                    float minSqrDist = float.PositiveInfinity;
//                    foreach (var node in similarCostNodes)
//                    {
//                        float sqrDist = (averagePos - node.Position).sqrMagnitude;
//                        if (sqrDist < minSqrDist)
//                        {
//                            minSqrDist = sqrDist;
//                            next = node;
//                        }
//                    }
//                    int intermediateI = prevI;
//                    int intermediateJ = prevJ;
//                    int nextI = next.ComputeIndexI - 1;
//                    int nextJ = next.ComputeIndexJ - 1;

//                    int deltaI = nextI - intermediateI;
//                    int deltaJ = nextJ - intermediateJ;
//                    while (Mathf.Abs(deltaI) > 1 || Mathf.Abs(deltaJ) > 1)
//                    {
//                        intermediateI += Math.Sign(deltaI);
//                        intermediateJ += Math.Sign(deltaJ);
//                        deltaI = nextI - intermediateI;
//                        deltaJ = nextJ - intermediateJ;
//                        pathNodes.Add(graph.Nodes[intermediateI][intermediateJ]);
//                    }

//                    pathNodes.Add(next);
//                    (prevI, prevJ) = (nextI, nextJ);
//                    prevGuide = guide;
//                    guide = guide.CameFrom1;
//                }
//                pathNodes.Add(goal);
//                return pathNodes.ToArray();
//            }
//        }

//        public static void RedrawHeatmap(Node[] nodesFlattened)
//        {
//            float minG = nodesFlattened.Where(n => !float.IsInfinity(n.G1)).Min(n => n.G1);
//            float maxG = nodesFlattened.Where(n => !float.IsInfinity(n.G1)).Max(n => n.G1);
//            Color minColor = Color.green;
//            Color maxColor = Color.red;
//            Vector2 deltaPos = nodesFlattened[0].Position - nodesFlattened[1].Position;
//            float step = Mathf.Max(Mathf.Abs(deltaPos.x), Mathf.Abs(deltaPos.y));
//            float halfStep = step / 2;

//            var mesh = new Mesh();
//            var vertices = new List<Vector3>();
//            var colors = new List<Color>();
//            foreach (var node in nodesFlattened)
//            {
//                float t = (node.G1 - minG) / (maxG - minG);
//                var color = node.G1 == -1 || float.IsInfinity(node.G1) ? Color.blue : Color.Lerp(minColor, maxColor, t);
//                vertices.Add(node.Position + new Vector2(-halfStep, -halfStep));
//                vertices.Add(node.Position + new Vector2(+halfStep, -halfStep));
//                vertices.Add(node.Position + new Vector2(+halfStep, +halfStep));
//                vertices.Add(node.Position + new Vector2(-halfStep, +halfStep));

//                colors.Add(color);
//                colors.Add(color);
//                colors.Add(color);
//                colors.Add(color);
//            }
//            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
//            mesh.SetVertices(vertices);
//            mesh.SetColors(colors);
//            var indices = Enumerable.Range(0, vertices.Count).ToArray();
//            mesh.SetIndices(indices, MeshTopology.Quads, 0);

//            var old = GameObject.Find("HEATMAP");
//            if (old != null)
//            {
//                GameObject.DestroyImmediate(old);
//            }
//            var go = new GameObject("HEATMAP");
//            go.transform.position = new Vector3(0, 0, -0.95f);
//            var meshFilter = go.AddComponent<MeshFilter>();
//            meshFilter.sharedMesh = mesh;
//            var renderer = go.AddComponent<MeshRenderer>();
//            renderer.material = new Material(Shader.Find("Sprites/Default"));
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        static void AtomicMin(ref float ptr, float value)
//        {
//            float oldVal, newVal;
//            do
//            {
//                oldVal = ptr;
//                newVal = Mathf.Min(oldVal, value);
//            } while (Interlocked.CompareExchange(ref ptr, oldVal, newVal) != oldVal);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        private static float Heuristic(Node node, Node goal, Node start)
//        {
//            Vector2 nodeToGoal = goal.Position - node.Position;
//            //Vector2 startToNode = node.Position - start.Position;
//            float h = nodeToGoal.magnitude;
//            //return h;

//            Node current = node?.CameFrom1;
//            Node old = current?.CameFrom1 ?? current;
//            for (int i = 0; i < 15; i++)
//                old = old?.CameFrom1 ?? old;
//            if (current != null)
//            {
//                Vector2 trend = current.Position - old.Position;
//                Vector2 newDirection = node.Position - old.Position;
//                float angleToGoal = Vector2.Angle(trend, nodeToGoal);
//                float newAngleToGoal = Vector2.Angle(newDirection, nodeToGoal);
//                if (newAngleToGoal < angleToGoal)
//                {
//                    h = 0;
//                }
//                else
//                {
//                    Vector2 nodeToStart = start.Position - node.Position;
//                    float angleToStart = Vector2.Angle(-trend, nodeToStart);
//                    float newAngleToStart = Vector2.Angle(-newDirection, nodeToStart);
//                    if (newAngleToStart > angleToStart)
//                    {
//                        h = 0;
//                    }
//                }
//            }

//            //nodeToGoal.Normalize();
//            //startToNode.Normalize();
//            //float cross = nodeToGoal.x * startToNode.y - startToNode.x * nodeToGoal.y;
//            //h -= h * (cross);

//            ////Vector2 nodeToGoal = goal.Position - node.Position;
//            ////float distanceToGoal = nodeToGoal.magnitude;

//            ////Node old = current?.CameFrom;
//            ////old = old?.CameFrom ?? old;
//            ////old = old?.CameFrom ?? old;
//            //Node current = node.CameFrom;
//            //if (current != null)
//            //{
//            //    Vector2 currentToGoalDir = (current.Position - goal.Position).normalized;
//            //    Vector2 currentToNodeDir = (current.Position - node.Position).normalized;
//            //    float dot = Vector2.Dot(currentToGoalDir, currentToNodeDir);
//            //    float v = 1 - (Mathf.Acos(Mathf.Max(0, dot)) / (Mathf.PI / 2));
//            //    h -= h * v;
//            //    //Vector2 currTrendDir = (current.Position - old.Position).normalized;
//            //    //Vector2 newDir = (node.Position - current.Position).normalized;

//            //    //Vector2 prevToNodeNorm = (node.Position - prev.Position).normalized;
//            //    //float prevToNodeDot = Vector2.Dot(prevToNodeNorm, nodeToGoal / distanceToGoal);
//            //    //prevToNodeDot = Mathf.Max(0, prevToNodeDot);
//            //    //h -= h * prevToNodeDot;
//            //    //h = Math.Max(0, h - 100f * Mathf.Max(0, dot));
//            //}

//            return h;
//        }

//        private static void CleanupGraph(Node[] nodesFlattened)
//        {
//            foreach (var node in nodesFlattened)
//            {
//                node.CleanupAfterPathSearch();
//            }
//        }
//    }
//}