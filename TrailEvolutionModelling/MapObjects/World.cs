using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Mapsui.Geometries;
using TrailEvolutionModelling.Attractors;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public class World
    {
        public BoundingAreaPolygon BoundingArea { get; set; }

        [XmlArrayItem(typeof(Line))]
        [XmlArrayItem(typeof(Polygon))]
        public MapObject[] MapObjects { get; set; }
        public AttractorObject[] AttractorObjects { get; set; }

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

        public AreaAttributes GetAreaAttributesInLine(Point nodePos, Point neighbourPos)
        {
            if (BoundingArea.IntersectsLine(nodePos, neighbourPos))
                return AreaAttributes.Unwalkable;

            float maxWeight = 0;
            AreaAttributes resultAttribtues = default;
            foreach (var mapObj in MapObjects)
            {
                if (mapObj.IntersectsLine(nodePos, neighbourPos))
                {
                    AreaAttributes attributes = mapObj.AreaType.Attributes;
                    if (!attributes.IsWalkable)
                        return AreaAttributes.Unwalkable;

                    if (attributes.Weight > maxWeight)
                    {
                        maxWeight = attributes.Weight;
                        resultAttribtues = attributes;
                    }
                }
            }

            if (maxWeight == 0)
            {
                resultAttribtues = AreaTypes.Default.Attributes;
            }
            return resultAttribtues;
        }
    }
}
