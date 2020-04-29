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
    }
}
