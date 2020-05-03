using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using TrailEvolutionModelling.GraphTypes;

namespace TrailEvolutionModelling.GraphBuilding
{
    static class GraphBuilder
    {
        private static readonly int MaxSteps = 500;

        public static Graph Build(GraphBuilderInput input)
        {
            BoundingBox bounds = input.BoundingArea.Geometry.BoundingBox;
            Point min = bounds.Min;

            float step = ClampStep(input.DesiredStep, bounds);
            int w = (int)Math.Ceiling(bounds.Width / step);
            int h = (int)Math.Ceiling(bounds.Height / step);

            var graph = new Graph(w, h, (float)min.X, (float)min.Y, step);


            return null;
        }

        static float ClampStep(float step, BoundingBox bounds)
        {
            double max = Math.Max(bounds.Width, bounds.Height);
            if (max > step * MaxSteps)
            {
                step = (float)(max / MaxSteps);
            }
            return step;
        }

        // TODO
        //static void BuildNodes(Graph graph)
        //{
        //    for (int i = 0; i < graph.Width; i++)
        //    {
        //        for (int j = 0; j < graph.Height; j++)
        //        {
        //            if (MapObject.IsPointWalkable(graph.GetNodePosition(i, j)))
        //            {
        //                graph.AddNode(i, j);
        //            }
        //        }
        //    }
        //}
    }
}
