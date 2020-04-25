using System;
using Mapsui.Layers;

namespace TrailEvolutionModelling.Util
{
    static class ILayerExtensions
    {
        public static void Refresh(this ILayer layer)
        {
            layer.DataHasChanged();
            //layer.ViewChanged(true, layer.Envelope, resolution: 1);
        }
    }
}
