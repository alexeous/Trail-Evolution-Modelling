using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using Mapsui.Styles;
using TrailEvolutionModelling.Styles;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public class AreaType
    {
        public AreaAttributes Attributes { get; set; }
        
        [XmlElement(typeof(LineStyle))]
        [XmlElement(typeof(PolygonStyle))]
        public MapObjectStyle Style { get; set; }
        
        [XmlAttribute]
        public string Name { get; set; }
        
        [XmlAttribute]
        public string DisplayedName { get; set; }

        public AreaType() { }
    }
}
