using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.Geometries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MapsuiPolygon = Mapsui.Geometries.Polygon;

namespace TrailEvolutionModelling.MapObjects
{
    public class Polygon : MapObject
    {
        public override IList<Point> Vertices => MapsuiPolygon.ExteriorRing.Vertices;

        private MapsuiPolygon MapsuiPolygon => (MapsuiPolygon)Geometry;

        public Polygon()
        {
            Geometry = new MapsuiPolygon(new LinearRing());
            Styles = new List<IStyle>();
            Styles.Add(new VectorStyle
            {
                Fill = new Brush(new Color(240, 20, 20, 70)),
                Outline = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 2
                }
            });
        }

        protected override void InitGeometryFromText(string geometryText)
        {
            if (string.IsNullOrWhiteSpace(geometryText))
            {
                throw new ArgumentException("geometryText is null or blank");
            }

            MapsuiPolygon mapsuiPolygon;
            try
            {
                var geometry = Mapsui.Geometries.Geometry.GeomFromText(geometryText) as MapsuiPolygon;
                mapsuiPolygon = geometry as MapsuiPolygon;
                if (mapsuiPolygon == null)
                {
                    throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText));
                }
            }
            catch (Exception ex)
            {
                throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText), ex);
            }

            Geometry = mapsuiPolygon;
        }
    }
}
