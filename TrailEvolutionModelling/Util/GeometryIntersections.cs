using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using Mapsui.Geometries.Utilities;

namespace TrailEvolutionModelling.Util
{
    class GeometryIntersections
    {
        public static bool DoesLineIntersectPolyline(Point start, Point end, IList<Point> vertices)
        {
            for (int i = 1; i < vertices.Count; i++)
            {
                Point segmentStart = vertices[i - 1];
                Point segmentEnd = vertices[i];
                double distance = CGAlgorithms.DistanceLineLine(start, end, segmentStart, segmentEnd);
                if (distance <= 0.001)
                {
                    return true;
                }
            }
            return false;
        }
    }
}
