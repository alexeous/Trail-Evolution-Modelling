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
        public override bool AreVerticesLooped => true;
        public override string DisplayedName => AreaType?.DisplayedName ?? "<Polygon>";

        private MapsuiPolygon MapsuiPolygon => (MapsuiPolygon)Geometry;

        public override double Distance(Point p)
        {
            return MapsuiPolygon.Distance(p);
        }

        protected override IGeometry CreateGeometry()
        {
            return new MapsuiPolygon(new LinearRing());
        }

        protected override void InitGeometryFromText(string geometryText)
        {
            if (string.IsNullOrWhiteSpace(geometryText))
            {
                throw new ArgumentException($"{nameof(geometryText)} is null or blank");
            }

            MapsuiPolygon mapsuiPolygon;
            try
            {
                var geometry = Mapsui.Geometries.Geometry.GeomFromText(geometryText);
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

        protected override VectorStyle CreateHighlighedStyle()
        {
            return new VectorStyle
            {
                Fill = new Brush(Color.Transparent),
                Outline = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 7
                }
            };
        }
    }
}
