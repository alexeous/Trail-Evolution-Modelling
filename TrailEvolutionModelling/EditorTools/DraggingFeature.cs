using Mapsui.Geometries;
using Mapsui.Providers;
using System;
using System.Collections.Generic;
using TrailEvolutionModelling.MapObjects;
using Polygon = TrailEvolutionModelling.MapObjects.Polygon;

namespace TrailEvolutionModelling.EditorTools
{
    class DraggingFeature : Feature
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

        public DraggingFeature(MapObject mapObject, Point vertex)
        {
            if (mapObject == null)
            {
                throw new ArgumentNullException(nameof(mapObject));
            }

            this.vertex = vertex ?? throw new ArgumentNullException(nameof(vertex));
            this.vertices = mapObject.Vertices;

            Geometry = vertex;
        }
    }
}
