using Mapsui.Layers;
using Mapsui.Styles;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.Polygons
{
    class PolygonLayer : WritableLayer
    {
        public PolygonLayer()
        {
            Name = "Polygons";
            Style = CreateLayerStyle();
        }

        private static IStyle CreateLayerStyle()
        {
            return new VectorStyle
            {
                Fill = new Brush(new Color(240, 20, 20, 70)),
                Outline = new Pen
                {
                    Color = new Color(240, 20, 20),
                    Width = 2
                }
            };
        }
    }
}
