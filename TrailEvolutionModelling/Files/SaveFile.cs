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
        public World World { get; set; }
        public BoundingBox Viewport { get; set; }
        public Trampledness Trampledness { get; set; }
    }
}
