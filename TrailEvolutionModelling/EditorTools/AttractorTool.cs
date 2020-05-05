using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.UI.Wpf;
using TrailEvolutionModelling.Attractors;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.EditorTools
{
    class AttractorTool : Tool
    {
        public AttractorType AttractorType { get; set; }
        public bool IsLarge { get; set; }

        public AttractorObject Result { get; private set; }

        private MapControl mapControl;
        private WritableLayer targetLayer;
        private System.Windows.Point mouseDownPos;

        public AttractorTool(MapControl mapControl, WritableLayer targetLayer)
        {
            this.mapControl = mapControl;
            this.targetLayer = targetLayer;
        }

        protected override void BeginImpl()
        {
            mapControl.Cursor = Cursors.Pen;
            SubscribeMouseEvents();
        }

        protected override void EndImpl()
        {
            UnsubscribeMouseEvents();
            mapControl.Cursor = Cursors.Arrow;
            targetLayer.Refresh();
        }

        private void SubscribeMouseEvents()
        {
            this.mapControl.MouseLeftButtonDown += OnLeftMouseDown;
            this.mapControl.MouseLeftButtonUp += OnLeftMouseUp;
        }

        private void UnsubscribeMouseEvents()
        {
            this.mapControl.MouseLeftButtonDown -= OnLeftMouseDown;
            this.mapControl.MouseLeftButtonUp -= OnLeftMouseUp;
        }

        private void OnLeftMouseDown(object sender, MouseButtonEventArgs e)
        {
            mouseDownPos = e.GetPosition(mapControl);
        }

        private void OnLeftMouseUp(object sender, MouseButtonEventArgs e)
        {
            const double tolerance = 5;
            var pos = e.GetPosition(mapControl);
            if (Math.Abs(pos.X - mouseDownPos.X) > tolerance ||
                Math.Abs(pos.Y - mouseDownPos.Y) > tolerance)
            {
                return;
            }

            PlaceAttractorObject(mapControl.Viewport.ScreenToWorld(pos.ToMapsui()));
            End();
        }

        private void PlaceAttractorObject(Point point)
        {
            Result = new AttractorObject
            {
                Position = point,
                WorkingRadius = AttractorObject.DefaultWorkingRadius,
                Type = AttractorType,
                IsLarge = IsLarge
            };
            targetLayer.Add(Result);
        }
    }
}
