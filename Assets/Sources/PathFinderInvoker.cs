using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TrailEvolutionModelling.Drawing;
using UnityEditor;
using UnityEngine;

using Debug = UnityEngine.Debug;

namespace TrailEvolutionModelling
{
    [ExecuteInEditMode]
    public class PathFinderInvoker : LinesProvider, ILinesChangedNotifier
    {
        [SerializeField] GraphHolder graphHolder = null;
        [SerializeField] ParallelWavefrontPathFinding parallelWavefront = null;
        [SerializeField] Transform start = null;
        [SerializeField] Transform end = null;
        [SerializeField] Color pathColor = Color.red;
        [SerializeField] int ind = 0;

        private Color oldPathColor;
        private Node[] path;

        public event Action<ILinesChangedNotifier> LinesChanged;

        [ContextMenu("Invoke A*")]
        public void InvokeAStar()
        {
            FindPath(PathFindingAlgorithm.AStar);
        }

        public int count;

#if UNITY_EDITOR
        void OnDrawGizmos()
        {
            try
            {
                Handles.color = Color.white;
                foreach (var p in PathFinder.gg[ind])
                {
                    Handles.DrawSolidDisc(p, Vector3.forward, 0.2f);
                }
                count = PathFinder.gg[ind].Count;
            }
            catch { }

            try
            {
                foreach (var n in ParallelWavefrontPathFinding.neigh)
                {
                    Color color = Color.red;
                    color.a = 1 - n.F1 / 7f;
                    Handles.color = color;
                    Handles.DrawSolidDisc(n.Position, Vector3.forward, 0.2f);
                }
            }
            catch { }
        }
#endif

        public TimeSpan FindPath(PathFindingAlgorithm algorithm)
        {
            if (graphHolder == null || start == null || end == null)
            {
                return TimeSpan.Zero;
            }

            Graph graph = graphHolder.Graph;
            if (graph == null)
            {
                throw new InvalidOperationException("GraphHolder contains no Graph");
            }

            Node startNode = FindClosestNode(graph, start.position);
            Node endNode = FindClosestNode(graph, end.position);

            var stopwatch = Stopwatch.StartNew();
            if (algorithm == PathFindingAlgorithm.WavefrontParallel)
            {
                this.path = parallelWavefront.FindPath(graph, startNode, endNode);
            }
            else
            {
                this.path = PathFinder.FindPath(graph, startNode, endNode, algorithm);
            }
            if (this.path == null)
            {
                Debug.LogWarning("Path not found");
            }
            stopwatch.Stop();

            Debug.Log($"Path finding took {stopwatch.ElapsedMilliseconds} ms");

            LinesChanged?.Invoke(this);
            return stopwatch.Elapsed;
        }

        public override IEnumerable<ColoredLine> GetLines()
        {
            if (path == null || path.Length < 2)
                yield break;

            Node prev = path[0];
            foreach (var node in path.Skip(1))
            {
                Vector3 start = prev.Position;
                Vector3 end = node.Position;
                prev = node;
                yield return new ColoredLine(start, end, pathColor);
            }
        }

        private Node FindClosestNode(Graph graph, Vector2 position)
        {
            Node closest = null;
            float minSqrDist = float.PositiveInfinity;

            for (int i = 0; i < graph.Nodes.Length; i++)
                for (int j = 0; j < graph.Nodes[0].Length; j++)
                {
                    Node node = graph.Nodes[i][j];
                    if (node == null)
                        continue;

                    float sqrDist = (node.Position - position).sqrMagnitude;
                    if (sqrDist < minSqrDist)
                    {
                        minSqrDist = sqrDist;
                        closest = node;
                    }
                }

            return closest;
        }

        private void Update()
        {
            if (oldPathColor != pathColor)
            {
                oldPathColor = pathColor;

                LinesChanged?.Invoke(this);
            }
        }
    }
}