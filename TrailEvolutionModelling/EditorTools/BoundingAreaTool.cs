using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.UI.Wpf;
using TrailEvolutionModelling.Layers;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Util;
using Polygon = TrailEvolutionModelling.MapObjects.Polygon;

namespace TrailEvolutionModelling.EditorTools
{
    class BoundingAreaTool : Tool
    {
        private BoundingAreaLayer boundingAreaLayer;
        private BoundingAreaPolygonTool polygonTool;
        private BoundingAreaPolygon boundingArea;

        public BoundingAreaPolygon BoundingArea
        {
            get => boundingArea;
            set
            {
                if (value != boundingArea)
                {
                    boundingAreaLayer.TryRemove(boundingArea);
                    boundingArea = value;
                    if (boundingArea != null)
                        boundingAreaLayer.Add(boundingArea);
                    
                    boundingAreaLayer.Refresh();
                }
            }
        }

        public BoundingAreaTool(MapControl mapControl, BoundingAreaLayer boundingAreaLayer)
        {
            this.boundingAreaLayer = boundingAreaLayer;

            polygonTool = new BoundingAreaPolygonTool(mapControl, boundingAreaLayer);
        }

        public bool Remove()
        {
            if (BoundingArea == null)
                return false;

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