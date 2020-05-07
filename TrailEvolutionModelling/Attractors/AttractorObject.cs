using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing.Text;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Documents.Serialization;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using Mapsui.Geometries;
using Mapsui.Providers;
using Mapsui.Styles;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.Attractors
{
    public enum AttractorType { Universal, Source, Drain }
    public enum AttractorPerformance { Normal, High }
    
    public class AttractorObject : Feature, IMapObject, IXmlSerializable
    {
        public static readonly float DefaultWorkingRadius = 1000;

        public float WorkingRadius { get; set; }
        public Point Position
        {
            get => Geometry as Point ?? new Point();
            set => Geometry = value;
        }

        public AttractorPerformance Performance
        {
            get => performance;
            set
            {
                performance = value;
                UpdateStyle();
            }
        }

        public AttractorType Type
        {
            get => type;
            set
            {
                type = value;
                UpdateStyle();
            }
        }

        public string DisplayedName => GetBaseDisplayedName() + GetPerformancePostfix();

        public Highlighter Highlighter { get; }

        private AttractorPerformance performance;
        private AttractorType type;
        private VectorStyle style;


        public AttractorObject()
        {
            Highlighter = new Highlighter(this, CreateHighlightedStyle());
            style = new VectorStyle();
            Styles.Add(style);
            UpdateStyle();
        }

        public Color GetColor()
        {
            switch (type)
            {
                case AttractorType.Universal: return Color.FromArgb(255, 230, 30, 230);
                case AttractorType.Source: return Color.FromArgb(255, 230, 230, 30);
                case AttractorType.Drain: return Color.FromArgb(255, 30, 30, 230);
                default: throw new NotSupportedException("Unknown AttractorType");
            }
        }


        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            Position = (Point)Mapsui.Geometries.Geometry.GeomFromText(reader.GetAttribute("Position"));
            WorkingRadius = float.Parse(reader.GetAttribute("WorkingRadius"));
            Performance = (AttractorPerformance)Enum.Parse(typeof(AttractorPerformance), reader.GetAttribute("Performance"));
            Type = (AttractorType)Enum.Parse(typeof(AttractorType), reader.GetAttribute("Type"));

            reader.ReadStartElement();
        }

        public void WriteXml(XmlWriter writer)
        {
            writer.WriteAttributeString("Position", Position.AsText());
            writer.WriteAttributeString("WorkingRadius", WorkingRadius.ToString());
            writer.WriteAttributeString("Performance", Performance.ToString());
            writer.WriteAttributeString("Type", Type.ToString());
        }


        public double Distance(Point p)
        {
            return Position.Distance(p);
        }
        
        private void UpdateStyle()
        {
            style.Fill.Color = GetColor();
            style.Outline.Color = GetColor().Scale(0.5f);
            style.Outline.Width = GetOutlineWidth();
        }

        private double GetOutlineWidth() => Performance == AttractorPerformance.High ? 8 : 1.5;

        private string GetBaseDisplayedName()
        {
            switch (Type)
            {
                case AttractorType.Universal: return "Универсальная точка притяжения";
                case AttractorType.Source: return "Источник";
                case AttractorType.Drain: return "Сток";
                default: throw new NotSupportedException($"Unknown {nameof(AttractorType)}");
            }
        }

        private string GetPerformancePostfix()
        {
            switch (Performance)
            {
                case AttractorPerformance.Normal: return "";
                case AttractorPerformance.High: return "+";
                default: throw new NotSupportedException($"Unknown {nameof(AttractorPerformance)}");
            }
        }

        private static VectorStyle CreateHighlightedStyle()
        {
            return new VectorStyle
            {
                Fill = new Brush(Color.Transparent),
                Outline = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 7
                }
            };
        }
    }
}
