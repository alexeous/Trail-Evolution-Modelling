using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui;
using Mapsui.Layers;
using Mapsui.Providers;
using Mapsui.Rendering;
using Mapsui.Rendering.Skia.SkiaStyles;
using Mapsui.Styles;
using SkiaSharp;
using TrailEvolutionModelling.Styles;

namespace TrailEvolutionModelling.Layers
{
    class BoundingAreaLayer : WritableLayer
    {
        public BoundingAreaLayer()
        {
            Name = "Bounding Area Layer";
            Style = new BoundingAreaStyle();
        }
    }
}
