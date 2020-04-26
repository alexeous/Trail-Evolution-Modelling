using Mapsui.Geometries;
using Mapsui.Providers;
using System;
using System.Collections.Generic;

namespace TrailEvolutionModelling.MapObjects
{
    partial class PolygonEditing
    {
        private class InsertionPreviewFeature : Feature
        {
            private readonly IList<Point> vertices;
            public Point Vertex { get; private set; }
            public int Index { get; private set; }

            public InsertionPreviewFeature(Polygon polygon, Point vertex, int index)
            {
                if (polygon == null)
                {
                    throw new ArgumentNullException(nameof(polygon));
                }
                vertices = polygon.Vertices;
                Update(vertex, index);
            }

            public void Update(Point vertex, int index)
            {
                if (index < 0 || index >= vertices.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }
                Geometry = Vertex = vertex ?? throw new ArgumentNullException(nameof(vertex));
                Index = index;
            }
        }
    }
}
