using Mapsui;
using Mapsui.Layers;
using Mapsui.Projection;
using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.UI.Wpf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.MapObjects
{
    class PolygonTool
    {
        private MapControl mapControl;
        private WritableLayer polygonLayer;

        private AreaType currentAreaType;
        private System.Windows.Point mouseDownPos;
        private Polygon currentPolygon;
        private Mapsui.Geometries.Point previewPoint;
        private Feature previewPointFeature;

        public bool IsInDrawingMode { get; private set; }

        public PolygonTool(MapControl mapControl, WritableLayer polygonLayer)
        {
            this.mapControl = mapControl;
            this.polygonLayer = polygonLayer;

            this.mapControl.MouseLeftButtonDown += OnLeftMouseDown;
            this.mapControl.MouseLeftButtonUp += OnLeftMouseUp;
            this.mapControl.MouseMove += OnMouseMove;
        }

        private void OnLeftMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (!IsInDrawingMode)
            {
                return;
            }
            mouseDownPos = e.GetPosition(mapControl);
        }

        private void OnLeftMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (!IsInDrawingMode)
            {
                return;
            }
            const double tolerance = 5;
            var pos = e.GetPosition(mapControl);
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
            
            if (currentPolygon == null)
            {
                currentPolygon = new Polygon();
                currentPolygon.AreaType = currentAreaType;
                polygonLayer.Add(currentPolygon);
            }
            if (previewPoint != null)
            {
                currentPolygon.Vertices.Remove(previewPoint);
            }
            previewPoint = GetGlobalPointFromEvent(e);
            currentPolygon.Vertices.Add(previewPoint);

            if (previewPointFeature == null)
            {
                polygonLayer.Add(previewPointFeature = new Feature { Geometry = previewPoint });
            }
            previewPointFeature.Geometry = previewPoint;

            Update();
        }

        private Mapsui.Geometries.Point GetGlobalPointFromEvent(MouseEventArgs e)
        {
            var screenPosition = e.GetPosition(mapControl).ToMapsui();
            var globalPosition = mapControl.Viewport.ScreenToWorld(screenPosition);
            return globalPosition;
        }

        private void Update()
        {
            polygonLayer.Refresh();
        }

        public void BeginDrawing(AreaType areaType)
        {
            if (IsInDrawingMode)
            {
                return;
            }

            IsInDrawingMode = true;
            currentAreaType = areaType;
            mapControl.Cursor = Cursors.Pen;
        }

        public Polygon EndDrawing()
        {
            if (!IsInDrawingMode)
            {
                return null;
            }

            IsInDrawingMode = false;
            Polygon result = currentPolygon;

            mapControl.Cursor = Cursors.Arrow;

            if (currentPolygon != null)
            {
                if (previewPoint != null)
                {
                    currentPolygon.Vertices.Remove(previewPoint);
                }
                if (currentPolygon.Vertices.Count <= 1)
                {
                    polygonLayer.TryRemove(currentPolygon);
                    result = null;
                }
            }
            if (previewPointFeature != null)
            {
                polygonLayer.TryRemove(previewPointFeature);
            }
            Update();

            currentPolygon = null;
            previewPoint = null;
            previewPointFeature = null;

            return result;
        }
    }
}
