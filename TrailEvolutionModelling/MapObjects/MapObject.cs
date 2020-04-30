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
        
        public Highlighter Highlighter { get; }

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
            throw new NotImplementedException();
        }

        public void WriteXml(XmlWriter writer)
        {
            throw new NotImplementedException();
        }

        public abstract double Distance(Point p);
        protected abstract IGeometry CreateGeometry();
        protected abstract void InitGeometryFromText(string geometryText);
        protected abstract VectorStyle CreateHighlighedStyle();
    }
}
