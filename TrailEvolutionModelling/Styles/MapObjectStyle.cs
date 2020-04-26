using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using Mapsui.Styles;

namespace TrailEvolutionModelling.Styles
{
    public abstract class MapObjectStyle : VectorStyle, IXmlSerializable
    {
        public abstract Color Color { get; set; }
        public abstract PenStyle PenStyle { get; set; }

        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            reader.MoveToContent();
            
            Color = Color.FromString(reader.GetAttribute("Color"));
            PenStyle = (PenStyle)Enum.Parse(typeof(PenStyle), reader.GetAttribute("PenStyle"));
        }

        public void WriteXml(XmlWriter writer)
        {
            var color = System.Drawing.Color.FromArgb(Color.R, Color.G, Color.B);
            string hex = System.Drawing.ColorTranslator.ToHtml(color);

            writer.WriteAttributeString("Color", hex);
            writer.WriteAttributeString("PenStyle", PenStyle.ToString());
        }
    }
}
