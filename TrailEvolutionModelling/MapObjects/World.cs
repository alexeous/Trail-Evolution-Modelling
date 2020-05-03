using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public class World
    {
        public BoundingAreaPolygon BoundingArea { get; set; }

        [XmlArrayItem(typeof(Line))]
        [XmlArrayItem(typeof(Polygon))]
        public MapObject[] MapObjects { get; set; }
    }
}
