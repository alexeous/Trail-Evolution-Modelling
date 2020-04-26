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
    public class PolygonStyle : MapObjectStyle
    {
        private const double OutlineThickness = 2;
        private const double OutlineColorDimFactor = 0.5;

        public override Color Color
        {
            get => Fill.Color;
            set
            {
                Fill.Color = value;
                Outline.Color = new Color((int)(value.R * OutlineColorDimFactor),
                                          (int)(value.G * OutlineColorDimFactor),
                                          (int)(value.B * OutlineColorDimFactor));
            }
        }

        public override PenStyle PenStyle {
            get => Outline.PenStyle;
            set => Outline.PenStyle = value;
        }

        public PolygonStyle()
        {
            Outline.Width = OutlineThickness;
        }

        public PolygonStyle(int r, int g, int b)
            : this()
        {
            Color = new Color(r, g, b);
        }
    }
}
