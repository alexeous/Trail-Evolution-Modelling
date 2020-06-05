using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using Mapsui.Projection;
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
            Point origin = bounds.Min;

            float step = ClampStep(input.DesiredStep, bounds, out int w, out int h);
            float stepMeters = CalcStepMeters(step, origin);

            var graph = new Graph(w, h, (float)origin.X, (float)origin.Y, step, stepMeters);
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

                if (attractors.Exists(a => a.Node == closest))
                    continue;

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

        private static float CalcStepMeters(float step, Point origin)
        {
            const double earthRadius = 6371000; //meters
            const double deg2rad = Math.PI / 180;

            Point offseted = origin + new Point(step, 0);
            double lat = deg2rad * SphericalMercator.ToLonLat(origin.X, origin.Y).X;
            double lon1 = deg2rad * SphericalMercator.ToLonLat(origin.X, origin.Y).X;
            double lon2 = deg2rad * SphericalMercator.ToLonLat(offseted.X, offseted.Y).X;
            double deltaLon = lon2 - lon1;

            return (float)(2 * earthRadius * Math.Asin(Math.Abs(Math.Sin(deltaLon / 2) * Math.Cos(lat / 2))));
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
                if (area.IsTramplable)
                {
                    graph.TramplableEdgesNumber++;
                }
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

        static Random random = new Random();

        static float AttractorPerformanceToNumber(AttractorPerformance performance)
        {
            float normal = ((float)random.NextDouble() * 0.1f + 0.95f) * 0.1f;   // people per virtual second
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
