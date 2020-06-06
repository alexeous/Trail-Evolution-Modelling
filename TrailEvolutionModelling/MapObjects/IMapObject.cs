using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mapsui.Geometries;
using Mapsui.UI.Wpf;

namespace TrailEvolutionModelling.MapObjects
{
    public interface IMapObject
    {
        string DisplayedName { get; }
        Highlighter Highlighter { get; }

        double Distance(Point p);
        bool IsMouseOver(Point worldPos, MapControl mapControl);
    }
}
