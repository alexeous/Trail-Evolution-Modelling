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
    public abstract class MapObject : Feature, IXmlSerializable
    {
        public abstract IList<Point> Vertices { get; }
        public abstract bool AreVerticesLooped { get; }
        public abstract string DisplayedName { get; }
        public abstract int MinimumVertices { get; }

        public Highlighter Highlighter { get; }
        public bool IsVerticesNumberValid => Vertices != null && Vertices.Count >= MinimumVertices;

        private IStyle mainStyle;
        private AreaType areaType;

        public AreaType AreaType
        {
            get => areaType;
            set
            {
                if (mainStyle != null)
                {
                    Styles.Remove(mainStyle);
                }      
                
                areaType = value;
                mainStyle = areaType?.Style;

                if (mainStyle != null)
                {
                    Styles.Add(mainStyle);
                }
            }
        }


        public MapObject()
        {
            Geometry = CreateGeometry();
            Highlighter = new Highlighter(this, CreateHighlighedStyle());
            Styles = new List<IStyle>();
        }

        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            string areaTypeName = reader.GetAttribute("AreaType");
            if (areaTypeName != null)
                AreaType = AreaTypes.GetByName(areaTypeName);

            reader.MoveToContent();
            string geomText = reader.ReadElementContentAsString();
            InitGeometryFromText(geomText);
        }

        public void WriteXml(XmlWriter writer)
        {
            if (AreaType?.Name != null)
                writer.WriteAttributeString("AreaType", AreaType.Name);
            
            writer.WriteString(ConvertGeometryToText());
        }

        public abstract double Distance(Point p);
        protected abstract IGeometry CreateGeometry();
        protected abstract void InitGeometryFromText(string geometryText);
        protected abstract string ConvertGeometryToText();
        protected abstract VectorStyle CreateHighlighedStyle();
    }
}
