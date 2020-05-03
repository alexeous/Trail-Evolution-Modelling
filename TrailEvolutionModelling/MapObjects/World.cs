using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Mapsui.Geometries;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public class World
    {
        public BoundingAreaPolygon BoundingArea { get; set; }

        [XmlArrayItem(typeof(Line))]
        [XmlArrayItem(typeof(Polygon))]
        public MapObject[] MapObjects { get; set; }

        public bool IsPointWalkable(Point point)
        {
            if (!BoundingArea.Geometry.Contains(point))
                return false;

            for (int i = 0; i < MapObjects.Length; i++)
            {
                MapObject obj = MapObjects[i];
                if (!obj.AreaType.Attributes.IsWalkable && obj.Geometry.Contains(point))
                    return false;
            }

            return true;
        }

        public AreaAttributes GetAreaAttributes(Point nodePos, Point neighbourPos)
        {
            
        }
    }
}
