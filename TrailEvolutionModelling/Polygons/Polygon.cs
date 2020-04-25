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
                    // Style.Equals() is not symmetric, i. e.
                    // style1.Equals(style2) could return false whilst
                    // style2.Equals(style1) could return true.
                    // That causes bugs.
                    bool contains = StylesList.Any(s => ReferenceEquals(s, highlightedStyle));
                    if (!contains)
                    {
                        StylesList.Add(highlightedStyle);
                    }
                }
                else
                {
                    StylesList.RemoveAll(s => ReferenceEquals(s, highlightedStyle));
                }
            }
        }

        // Introduced to have an opportunity to remove items by using a predicate
        // because Style.Equals() is bugged.
        private List<IStyle> StylesList => Styles as List<IStyle>;

        public Polygon(IEnumerable<Point> vertices)
            : this(new MapsuiPolygon(new LinearRing(vertices)))
        {
        }

        private Polygon(MapsuiPolygon mapsuiPolygon)
        {
            Styles = new List<IStyle>();
            Geometry = this.mapsuiPolygon = mapsuiPolygon;
        }

        public static Polygon FromGeomText(string geometryText)
        {
            if (string.IsNullOrWhiteSpace(geometryText))
            {
                throw new ArgumentException("geometryText is null or blank");
            }
            try
            {
                var geometry = Mapsui.Geometries.Geometry.GeomFromText(geometryText) as MapsuiPolygon;
                if (geometry is MapsuiPolygon mp)
                {
                    return new Polygon(mp);
                }
                else
                {
                    throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText));
                }
            }
            catch (Exception ex)
            {
                throw new ArgumentException($"Geometry text is invalid: {geometryText}", nameof(geometryText), ex);
            }
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
