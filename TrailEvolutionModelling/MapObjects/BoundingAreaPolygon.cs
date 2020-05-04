using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using TrailEvolutionModelling.Util;
using MapsuiPolygon = Mapsui.Geometries.Polygon;

namespace TrailEvolutionModelling.MapObjects
{
    public class BoundingAreaPolygon : Polygon
    {
        public override string DisplayedName => "Рабочая область";

        public override double Distance(Point p)
        {
            return ((MapsuiPolygon)Geometry).ExteriorRing.GetLineString().Distance(p);
        }

        public override bool IntersectsLine(Point start, Point end)
        {
            return GeometryIntersections.DoesLineIntersectPolyline(start, end, Vertices);
        }
    }
}
