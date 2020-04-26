using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using Mapsui.Geometries;
using Mapsui.Providers;
using Mapsui.Styles;

namespace TrailEvolutionModelling.MapObjects
{
    abstract class MapObject : Feature, IXmlSerializable
    {
        public abstract IList<Point> Vertices { get; }
        

        [field:NonSerialized]
        public Highlighter Highlighter { get; }


        public MapObject()
        {
            Highlighter = new Highlighter(this, CreateHighlighedStyle());
        }

        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            throw new NotImplementedException();
        }

        public void WriteXml(XmlWriter writer)
        {
            throw new NotImplementedException();
        }

        protected abstract void InitGeometryFromText(string text);

        private static VectorStyle CreateHighlighedStyle()
        {
            return new VectorStyle
            {
                //Fill = new Brush(new Color(240, 240, 20, 70)),
                Fill = new Brush(),
                Outline = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 7
                }
            };
        }
    }
}
