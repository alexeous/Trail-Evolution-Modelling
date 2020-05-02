using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Layers;
using Mapsui.Styles;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.Layers
{
    class DraggingLayer : WritableLayer
    {
        public static readonly float Scale = 0.6f;
        private const string ImagePath = "Dragging.png";

        public DraggingLayer()
        {
            this.Style = new ImageStyle
            {
                BitmapId = BitmapResources.GetBitmapIdForEmbeddedResourceRelative(ImagePath),
                SymbolScale = Scale
            };
        }
    }
}
