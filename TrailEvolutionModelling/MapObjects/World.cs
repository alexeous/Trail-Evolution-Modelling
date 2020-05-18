using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Mapsui.Geometries;
using TrailEvolutionModelling.Attractors;
using TrailEvolutionModelling.MapObjects.SpatialIndexing;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public class World
    {
        public BoundingAreaPolygon BoundingArea { get; set; }

        [XmlArrayItem(typeof(Line))]
        [XmlArrayItem(typeof(Polygon))]
        public MapObject[] MapObjects
        {
            get => mapObjects;
            set
            {
                mapObjects = value;
                RebuildSpatialIndex();
            }
        }

        public AttractorObject[] AttractorObjects { get; set; }

        private MapObject[] mapObjects;
        private RTreeMemoryIndex<MapObject> spatialIndex;

        private ThreadLocal<HashSet<MapObject>> mapObjBuffers;

        public World()
        {
            mapObjBuffers = new ThreadLocal<HashSet<MapObject>>(() => new HashSet<MapObject>());
        }

        public bool IsPointWalkable(Point point)
        {
            if (!BoundingArea.Geometry.Contains(point))
                return false;

            if (mapObjects.Length == 0)
                return true;

            var mapObjBuffer = mapObjBuffers.Value;
            mapObjBuffer.Clear();
            spatialIndex.GetNonAlloc(point, mapObjBuffer);
            foreach (var obj in mapObjBuffer)
            {
                if (!obj.AreaType.Attributes.IsWalkable && obj.Geometry.Contains(point))
                    return false;
            }

            return true;
        }

        public AreaAttributes GetAreaAttributesInLine(Point nodePos, Point neighbourPos)
        {
            if (BoundingArea.IntersectsLine(nodePos, neighbourPos))
                return AreaAttributes.Unwalkable;

            if (mapObjects.Length == 0)
                return AreaTypes.Default.Attributes;

            var mapObjBuffer = mapObjBuffers.Value;
            mapObjBuffer.Clear();
            spatialIndex.GetNonAlloc(new BoundingBox(nodePos, neighbourPos), mapObjBuffer);
            float maxWeight = 0;
            AreaAttributes resultAttribtues = default;
            foreach (var mapObj in mapObjBuffer)
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

        private void RebuildSpatialIndex()
        {
            spatialIndex = new RTreeMemoryIndex<MapObject>();
            foreach (var mapObj in mapObjects)
            {
                spatialIndex.Add(mapObj.Geometry.BoundingBox, mapObj);
            }
        }
    }
}
