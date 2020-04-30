using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.UI.Wpf;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Util;
using Polygon = TrailEvolutionModelling.MapObjects.Polygon;

namespace TrailEvolutionModelling.EditorTools
{
    class BoundingAreaTool : Tool
    {
        private WritableLayer boundingAreaLayer;
        private BoundingAreaPolygonTool polygonTool;
        
        public Polygon BoundingArea { get; private set; }

        public BoundingAreaTool(MapControl mapControl, WritableLayer boundingAreaLayer)
        {
            this.boundingAreaLayer = boundingAreaLayer;

            polygonTool = new BoundingAreaPolygonTool(mapControl, boundingAreaLayer);
        }

        public bool Remove()
        {
            if (BoundingArea == null)
                return false;

            boundingAreaLayer.TryRemove(BoundingArea);
            boundingAreaLayer.Refresh();
            BoundingArea = null;
            return true;
        }

        protected override void BeginImpl()
        {
            if (BoundingArea != null)
            {
                End();
                return;
            }
            polygonTool.Begin();
        }

        protected override void EndImpl()
        {            
            polygonTool.End();

            BoundingArea = polygonTool.Result;
        }

        private class BoundingAreaPolygonTool : MapObjectTool<BoundingAreaPolygon>
        {
            public BoundingAreaPolygonTool(MapControl mapControl, WritableLayer targetLayer)
            : base(mapControl, targetLayer)
            {
            }

            protected override bool IsFinalResultAcceptable(BoundingAreaPolygon finalResult)
            {
                return finalResult.IsVerticesNumberValid;
            }

            protected override BoundingAreaPolygon CreateNewMapObject()
            {
                return new BoundingAreaPolygon();
            }
        }
    }
}