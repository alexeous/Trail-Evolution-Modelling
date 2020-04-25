using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.Geometries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MapsuiPolygon = Mapsui.Geometries.Polygon;

namespace TrailEvolutionModelling.Polygons
{
    class Polygon : Feature
    {
        private static readonly IStyle highlightedStyle = CreateHighlighedStyle();

        private readonly MapsuiPolygon mapsuiPolygon;
        private bool isHighlighted;

        public IList<Point> Vertices => mapsuiPolygon.ExteriorRing.Vertices;

        public string GeometryText => mapsuiPolygon.AsText();

        public bool IsHighlighted
        {
            get => isHighlighted;
            set
            {
                if (value == isHighlighted) return;

                isHighlighted = value;
                if(isHighlighted)
                {
                    Styles.Add(highlightedStyle);
                }
                else
                {
                    Styles.Remove(highlightedStyle);
                }
            }
        }

        public string ObjectKindName => "Препятствие";

        public Polygon(IEnumerable<Point> vertices)
            : this(new MapsuiPolygon(new LinearRing(vertices)))
        {
        }

        private Polygon(MapsuiPolygon mapsuiPolygon)
        {
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
            Geometry = this.mapsuiPolygon = mapsuiPolygon;
        }

        public static Polygon FromGeomText(string geometryText)
        {
            if (string.IsNullOrWhiteSpace(geometryText))
            {
                throw new ArgumentException("geometryText is null or blank");
            }
            MapsuiPolygon mapsuiPolygon = null;
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
            return new Polygon(mapsuiPolygon);
        }

        private static VectorStyle CreateHighlighedStyle()
        {
            return new VectorStyle
            {
                Fill = new Brush(new Color(240, 240, 20, 70)),
                Outline = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 2
                }
            };
        }
    }
}
