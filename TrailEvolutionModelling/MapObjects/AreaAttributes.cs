using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public struct AreaAttributes
    {
        [XmlAttribute]
        public bool IsWalkable { get; set; }
        
        [XmlAttribute]
        public float Weight { get; set; }
        
        [XmlAttribute]
        public bool IsTramplable { get; set; }

        public static AreaAttributes Unwalkable => new AreaAttributes
        {
            IsWalkable = false,
            IsTramplable = false
        };
    }
}
