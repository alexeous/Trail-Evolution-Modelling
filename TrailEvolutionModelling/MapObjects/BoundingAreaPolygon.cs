using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
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
    }
}
