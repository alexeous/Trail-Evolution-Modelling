using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Layers;
using Mapsui.UI.Wpf;
using TrailEvolutionModelling.MapObjects;

namespace TrailEvolutionModelling.EditorTools
{
    class LineTool : MapObjectTool<Line>
    {
        public LineTool(MapControl mapControl, WritableLayer targetLayer)
            : base(mapControl, targetLayer)
        {
        }

        protected override bool IsFinalResultAcceptable(Line finalResult)
        {
            return finalResult.IsVerticesNumberValid;
        }

        protected override Line CreateNewMapObject()
        {
            return new Line();
        }
    }
}
