using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Styles;

namespace TrailEvolutionModelling.Styles
{
    public class LineStyle : MapObjectStyle
    {
        private const double OutlineThickness = 3.5;

        public override Color Color
        {
            get => Line.Color;
            set => Line.Color = value;
        }

        public override PenStyle PenStyle
        {
            get => Line.PenStyle;
            set => Line.PenStyle = value;
        }

        public LineStyle()
        {
            Fill = new Brush(Color.Transparent);

            Line.Width = OutlineThickness;
        }

        public LineStyle(int r, int g, int b)
            : this()
        {
            Color = new Color(r, g, b);
        }
    }
}
