using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TrailEvolutionModelling.Drawing;
using UnityEditor;
using UnityEngine;

namespace TrailEvolutionModelling
{
    [ExecuteInEditMode]
    public class GraphHolder : LinesProvider, ILinesChangedNotifier
    {
        [SerializeField] float minWeight = 1;
        [SerializeField] float maxWeight = 2.7f;
        [SerializeField] Color minWeightColor = Color.blue;
        [SerializeField] Color maxWeightColor = Color.red;

        private Color oldMinColor;
        private Color oldMaxColor;
        private Graph graph;

        public Graph Graph
        {
            get => graph;
            set
            {
                graph = value;
                LinesChanged?.Invoke(this);
            }
        }

        public event Action<ILinesChangedNotifier> LinesChanged;

        public override IEnumerable<ColoredLine> GetLines()
        {
            if (graph?.Edges == null)
                yield break;

            foreach (var edge in graph.Edges)
            {
                Vector3 start = edge.Node1.Position;
                Vector3 end = edge.Node2.Position;
                Color color = GetEdgeColor(edge);

                yield return new ColoredLine(start, end, color);
            }
        }

        private Color GetEdgeColor(Edge edge)
        {
            float t = (edge.Weight - minWeight) / (maxWeight - minWeight);
            return Color.Lerp(minWeightColor, maxWeightColor, t);
        }

        private void Update()
        {
            if (minWeightColor != oldMinColor ||
                maxWeightColor != oldMaxColor)
            {
                oldMinColor = minWeightColor;
                oldMaxColor = maxWeightColor;

                LinesChanged?.Invoke(this);
            }
        }
    }
}