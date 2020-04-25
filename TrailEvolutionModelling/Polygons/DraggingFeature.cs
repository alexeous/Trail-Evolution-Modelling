using Mapsui.Geometries;
using Mapsui.Providers;
using System;
using System.Collections.Generic;

namespace TrailEvolutionModelling.Polygons
{
    partial class PolygonEditing
    {
        private class DraggingFeature : Feature
        {
            private readonly IList<Point> vertices;
            private Point vertex;

            public Point Vertex
            {
                get => vertex;
                set
                {
                    int index = vertices.IndexOf(Vertex);
                    if (index != -1)
                    {
                        vertices.RemoveAt(index);
                        vertices.Insert(index, value);
                    }
                    vertex = value;
                    Geometry = value;
                }
            }

            public DraggingFeature(Polygon polygon, Point vertex)
            {
                if (polygon == null)
                {
                    throw new ArgumentNullException(nameof(polygon));
                }

                this.vertex = vertex ?? throw new ArgumentNullException(nameof(vertex));
                this.vertices = polygon.Vertices;

                Geometry = vertex;
            }
        }
    }
}
