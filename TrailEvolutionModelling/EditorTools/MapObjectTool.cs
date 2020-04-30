using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Mapsui.Layers;
using Mapsui.Providers;
using Mapsui.UI.Wpf;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.EditorTools
{
    abstract class MapObjectTool<T> : Tool where T : MapObject
    {
        public AreaType AreaType { get; set; }

        protected MapControl MapControl { get; }
        protected WritableLayer TargetLayer { get; }
        protected T CurrentDrawnObject { get; set; }


        private System.Windows.Point mouseDownPos;
        private Mapsui.Geometries.Point previewPoint;


        public MapObjectTool(MapControl mapControl, WritableLayer targetLayer)
        {
            this.MapControl = mapControl;
            this.TargetLayer = targetLayer;
        }

        protected override void BeginImpl()
        {
            MapControl.Cursor = Cursors.Pen;

            SubscribeMouseEvents();
        }

        protected override void EndImpl()
        {
            UnsubscribeMouseEvents();

            T result = CurrentDrawnObject;

            MapControl.Cursor = Cursors.Arrow;

            if (CurrentDrawnObject != null)
            {
                if (previewPoint != null)
                {
                    CurrentDrawnObject.Vertices.Remove(previewPoint);
                }
                if (CurrentDrawnObject.Vertices.Count <= 1)
                {
                    TargetLayer.TryRemove(CurrentDrawnObject);
                    result = null;
                }
            }
            Update();

            CurrentDrawnObject = null;
            previewPoint = null;
        }

        private void OnLeftMouseDown(object sender, MouseButtonEventArgs e)
        {
            mouseDownPos = e.GetPosition(MapControl);
        }

        private void OnLeftMouseUp(object sender, MouseButtonEventArgs e)
        {
            const double tolerance = 5;
            var pos = e.GetPosition(MapControl);
            if (Math.Abs(pos.X - mouseDownPos.X) > tolerance ||
                Math.Abs(pos.Y - mouseDownPos.Y) > tolerance)
            {
                return;
            }
            // previewPoint is already in currentPolygon.
            // Setting it null makes OnMouseMove not remove it from currentPolygon
            // thus it stays persistently
            previewPoint = null;
        }

        private void OnMouseMove(object sender, MouseEventArgs e)
        {
            if (CurrentDrawnObject == null)
            {
                CurrentDrawnObject = CreateNewMapObject();
                CurrentDrawnObject.AreaType = AreaType;
                TargetLayer.Add(CurrentDrawnObject);
            }
            if (previewPoint != null)
            {
                CurrentDrawnObject.Vertices.Remove(previewPoint);
            }
            previewPoint = GetGlobalPointFromEvent(e);
            CurrentDrawnObject.Vertices.Add(previewPoint);

            Update();
        }

        private Mapsui.Geometries.Point GetGlobalPointFromEvent(MouseEventArgs e)
        {
            var screenPosition = e.GetPosition(MapControl).ToMapsui();
            var globalPosition = MapControl.Viewport.ScreenToWorld(screenPosition);
            return globalPosition;
        }

        private void Update()
        {
            TargetLayer.Refresh();
        }

        private void SubscribeMouseEvents()
        {
            this.MapControl.MouseLeftButtonDown += OnLeftMouseDown;
            this.MapControl.MouseLeftButtonUp += OnLeftMouseUp;
            this.MapControl.MouseMove += OnMouseMove;
        }

        private void UnsubscribeMouseEvents()
        {
            this.MapControl.MouseLeftButtonDown -= OnLeftMouseDown;
            this.MapControl.MouseLeftButtonUp -= OnLeftMouseUp;
            this.MapControl.MouseMove -= OnMouseMove;
        }

        protected abstract T CreateNewMapObject();
    }
}
