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
    abstract class MapObjectTool<T> where T : MapObject
    {
        protected MapControl MapControl { get; }
        protected WritableLayer TargetLayer { get; }
        protected AreaType CurrentAreaType { get; set; }
        protected T CurrentDrawnObject { get; set; }

        public bool IsInDrawingMode { get; private set; }

        private System.Windows.Point mouseDownPos;
        private Mapsui.Geometries.Point previewPoint;
        private Feature previewPointFeature;


        public MapObjectTool(MapControl mapControl, WritableLayer polygonLayer)
        {
            this.MapControl = mapControl;
            this.TargetLayer = polygonLayer;

            this.MapControl.MouseLeftButtonDown += OnLeftMouseDown;
            this.MapControl.MouseLeftButtonUp += OnLeftMouseUp;
            this.MapControl.MouseMove += OnMouseMove;
        }

        public void BeginDrawing(AreaType areaType)
        {
            if (IsInDrawingMode)
            {
                return;
            }

            IsInDrawingMode = true;
            CurrentAreaType = areaType;
            MapControl.Cursor = Cursors.Pen;
        }


        private void OnLeftMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!IsInDrawingMode)
            {
                return;
            }
            mouseDownPos = e.GetPosition(MapControl);
        }

        private void OnLeftMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (!IsInDrawingMode)
            {
                return;
            }
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
            if (!IsInDrawingMode)
            {
                return;
            }

            if (CurrentDrawnObject == null)
            {
                CurrentDrawnObject = CreateNewMapObject();
                CurrentDrawnObject.AreaType = CurrentAreaType;
                TargetLayer.Add(CurrentDrawnObject);
            }
            if (previewPoint != null)
            {
                CurrentDrawnObject.Vertices.Remove(previewPoint);
            }
            previewPoint = GetGlobalPointFromEvent(e);
            CurrentDrawnObject.Vertices.Add(previewPoint);

            if (previewPointFeature == null)
            {
                TargetLayer.Add(previewPointFeature = new Feature { Geometry = previewPoint });
            }
            previewPointFeature.Geometry = previewPoint;

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

        public T EndDrawing()
        {
            if (!IsInDrawingMode)
            {
                return null;
            }

            IsInDrawingMode = false;
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
            if (previewPointFeature != null)
            {
                TargetLayer.TryRemove(previewPointFeature);
            }
            Update();

            CurrentDrawnObject = null;
            previewPoint = null;
            previewPointFeature = null;

            return result;
        }

        protected abstract T CreateNewMapObject();
    }
}
