using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using Mapsui.Projection;

namespace TrailEvolutionModelling.Util
{
    static class PointExtensions
    {
        public static Point OffsetMeters(this Point point, double dx, double dy)
        {
            const double EarthRadius = 6_371_000;
            Point lonLat = SphericalMercator.ToLonLat(point.X, point.Y);
            var newLon = lonLat.X + (dx / EarthRadius) * (180 / Math.PI) / Math.Cos(lonLat.Y * Math.PI / 180);
            var newLat = lonLat.Y + (dy / EarthRadius) * (180 / Math.PI);
            return SphericalMercator.FromLonLat(newLon, newLat);
        }
    }
}
