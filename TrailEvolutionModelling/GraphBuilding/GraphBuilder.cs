using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using TrailEvolutionModelling.Attractors;
using TrailEvolutionModelling.GPUProxy;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.GraphBuilding
{
    static class GraphBuilder
    {
        public static Graph Build(GraphBuilderInput input)
        {
            BoundingBox bounds = input.World.BoundingArea.Geometry.BoundingBox;
            Point min = bounds.Min;

            float step = ClampStep(input.DesiredStep, bounds, out int w, out int h);

            var graph = new Graph(w, h, (float)min.X, (float)min.Y, step);
            BuildNodes(graph, input.World);
            BuildEdges(graph, input.World);

            return graph;
        }

        public static Attractor[] CreateAttractors(Graph graph, IEnumerable<AttractorObject> attractorObjects)
        {
            var attractors = new List<Attractor>();
            foreach (var attrObj in attractorObjects)
            {
                Node closest = graph.GetClosestNode((float)attrObj.Position.X, (float)attrObj.Position.Y);
                if (closest == null)
                    throw new ArgumentException("Graph has no nodes");

                var attractor = new Attractor
                {
                    Node = closest,
                    WorkingRadius = attrObj.WorkingRadius,
                    Performance = AttractorPerformanceToNumber(attrObj.Performance),
                    IsSource = attrObj.Type == AttractorType.Universal || attrObj.Type == AttractorType.Source,
                    IsDrain = attrObj.Type == AttractorType.Universal || attrObj.Type == AttractorType.Drain
                };
                attractors.Add(attractor);
            }
            return attractors.ToArray();
        }

        static float ClampStep(float step, BoundingBox bounds, out int w, out int h)
        {
            double max = Math.Max(bounds.Width, bounds.Height);
            if (max > step * Graph.MaximumSteps)
            {
                step = (float)max / Graph.MaximumSteps;
            }
            w = Math.Min(Graph.MaximumSteps, (int)Math.Ceiling(bounds.Width / step));
            h = Math.Min(Graph.MaximumSteps, (int)Math.Ceiling(bounds.Height / step));
            return step;
        }

        static void BuildNodes(Graph graph, World world)
        {
            Parallel.For(0, graph.Width, i =>
            {
                for (int j = 0; j < graph.Height; j++)
                {
                    var point = graph.GetNodePosition(i, j).ToMapsui();
                    if (world.IsPointWalkable(point))
                    {
                        graph.AddNode(i, j);
                    }
                }
            });
        }

        static void BuildEdges(Graph graph, World world)
        {
            for (int i = 0; i < graph.Width; i++)
            {
                for (int j = 0; j < graph.Height; j++)
                {
                    Node node = graph.GetNodeAtOrNull(i, j);
                    if (node == null)
                        continue;

                    bool notLastColumn = i < graph.Width - 1;
                    bool notLastRow = j < graph.Height - 1;
                    bool notFirstColumn = i != 0;

                    if (notLastColumn)
                        BuildEdge(graph, world, node, Direction.E);

                    if (notLastRow)
                    {
                        BuildEdge(graph, world, node, Direction.S);

                        if (notLastColumn)
                            BuildEdge(graph, world, node, Direction.SE);

                        if (notFirstColumn)
                            BuildEdge(graph, world, node, Direction.SW);
                    }
                }
            }
        }

        static void BuildEdge(Graph graph, World world, Node node, Direction direction)
        {
            if (TryGetAreaAttributes(graph, world, node, direction, out var area) &&
                area.IsWalkable)
            {
                graph.AddEdge(node, direction, area.Weight, area.IsTramplable);
            }
        }

        static bool TryGetAreaAttributes(Graph graph, World world, Node node, Direction dir, out AreaAttributes area)
        {
            Node neighbour = graph.GetNodeNeighbourOrNull(node, dir);
            if (neighbour == null)
            {
                area = default;
                return false;
            }

            Point nodePos = graph.GetNodePosition(node).ToMapsui();
            Point neighbourPos = graph.GetNodePosition(neighbour).ToMapsui();
            area = world.GetAreaAttributesInLine(nodePos, neighbourPos);
            return true;
        }

        static float AttractorPerformanceToNumber(AttractorPerformance performance)
        {
            float normal = 0.033333f;   // people per virtual second
            switch (performance)
            {
                case AttractorPerformance.Normal:
                    return normal;
                case AttractorPerformance.High:
                    return normal * 2;
                default:
                    throw new NotSupportedException($"Unknown {nameof(AttractorPerformance)} value");
            }
        }
    }
}
