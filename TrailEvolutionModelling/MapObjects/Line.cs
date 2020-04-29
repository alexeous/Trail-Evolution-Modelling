using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using Mapsui.Styles;

namespace TrailEvolutionModelling.MapObjects
{
    class Line : MapObject
    {
        public override IList<Point> Vertices => LineString.Vertices;

        private LineString LineString => (LineString)Geometry;

        protected override IGeometry CreateGeometry()
        {
            return new LineString();
        }

        protected override void InitGeometryFromText(string geometryText)
        {
            if (string.IsNullOrWhiteSpace(geometryText))
            {
                throw new ArgumentException($"{nameof(geometryText)} is null or blank");
            }

            LineString lineString;
            try
            {
                var geometry = Mapsui.Geometries.Geometry.GeomFromText(geometryText);
                lineString = geometry as LineString;
                if (lineString == null)
                {
                    throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText));
                }
            }
            catch (Exception ex)
            {
                throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText), ex);
            }

            Geometry = lineString;
        }

        protected override VectorStyle CreateHighlighedStyle()
        {
            return new VectorStyle
            {
                Line = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 7
                }
            };
        }
    }
}
