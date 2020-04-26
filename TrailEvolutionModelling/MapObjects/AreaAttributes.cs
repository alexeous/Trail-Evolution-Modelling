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
    struct AreaAttributes : IXmlSerializable
    {
        public bool IsWalkable { get; set; }
        public float Weight { get; set; }
        public bool IsTramplable { get; set; }

        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            throw new NotImplementedException();
        }

        public void WriteXml(XmlWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
