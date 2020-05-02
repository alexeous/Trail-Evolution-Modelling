using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Mapsui.Geometries;
using TrailEvolutionModelling.MapObjects;
using Polygon = TrailEvolutionModelling.MapObjects.Polygon;

namespace TrailEvolutionModelling.Files
{
    [Serializable]
    public class SaveFile
    {
        public Polygon BoundingArea { get; set; }

        [XmlArrayItem(typeof(Line))]
        [XmlArrayItem(typeof(Polygon))]
        public MapObject[] MapObjects { get; set; }

        public BoundingBox Viewport { get; set; }
    }
}
