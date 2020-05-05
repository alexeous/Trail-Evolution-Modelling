using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Input;
using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.UI.Wpf;
using TrailEvolutionModelling.Attractors;
using TrailEvolutionModelling.Layers;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling.EditorTools
{
    class AttractorEditing : Tool
    {
        private class AttractorDraggingFeature : Feature
        {
            public AttractorDraggingFeature(AttractorObject attractor) : base(attractor)
            {
                Geometry = attractor.Position;
                Styles.Add(new ImageStyle
                {
                    BitmapId = BitmapResources.GetBitmapIdForEmbeddedResourceRelative("Dragging.png"),
                    SymbolScale = 0.8f
                });
            }
        }

        public AttractorObject TargetObject { get; set; }
        private MapControl mapControl;
        private WritableLayer targetLayer;
        private WritableLayer draggingLayer;
        private AttractorDraggingFeature draggingFeature;
        private Point draggingOffset;
        private AttractorObject editedObject;

        public AttractorEditing(MapControl mapControl, WritableLayer targetLayer)
        {
            this.mapControl = mapControl;
            this.targetLayer = targetLayer;

            draggingLayer = new WritableLayer();
            this.mapControl.Map.Layers.Add(draggingLayer);
        }

        protected override void BeginImpl()
        {
            editedObject = TargetObject;

            if (editedObject == null)
            {
                throw new InvalidOperationException($"{nameof(TargetObject)} was not set");
            }

            draggingLayer.Add(new AttractorDraggingFeature(editedObject));
            draggingLayer.Refresh();

            SubscribeMouseEvents();
        }

        protected override void EndImpl()
        {
            UnsubscribeMouseEvents();

            draggingLayer.Clear();
            draggingLayer.Refresh();
            editedObject = null;
        }

        private void OnPreviewLeftMouseDown(object sender, MouseButtonEventArgs e)
        {
            Point mouseScreenPoint = e.GetPosition(mapControl).ToMapsui();
            var draggingFeature = GetFeaturesAtScreenPoint(mouseScreenPoint)
                .OfType<AttractorDraggingFeature>().FirstOrDefault();

            if (draggingFeature != null)
            {
                // Preventing map panning
                e.Handled = true;

                this.draggingFeature = draggingFeature;
                Point mouseWorldPoint = ScreenPointToGlobal(mouseScreenPoint);
                draggingOffset = mouseWorldPoint - (Point)draggingFeature.Geometry;
                return;
            }
        }


        private void OnPreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (draggingFeature != null)
            {
                Point mousePoint = ScreenPointToGlobal(e.GetPosition(mapControl).ToMapsui());
                draggingFeature.Geometry = editedObject.Position = mousePoint - draggingOffset;
                draggingLayer.Refresh();
                targetLayer.Refresh();
            }
        }

        private void OnPreviewLeftMouseUp(object sender, MouseButtonEventArgs e)
        {
            draggingFeature = null;
        }

        private IEnumerable<IFeature> GetFeaturesAtScreenPoint(Point point)
        {
            var worldPoint = ScreenPointToGlobal(point);
            double extentReduction = DraggingLayer.Scale / 2;
            double resolution = mapControl.Viewport.Resolution * extentReduction;
            var features = draggingLayer.GetFeaturesInView(new BoundingBox(worldPoint, worldPoint), resolution);
            return features;
        }

        private void SubscribeMouseEvents()
        {
            this.mapControl.PreviewMouseLeftButtonDown += OnPreviewLeftMouseDown;
            this.mapControl.PreviewMouseMove += OnPreviewMouseMove;
            this.mapControl.PreviewMouseLeftButtonUp += OnPreviewLeftMouseUp;
        }

        private void UnsubscribeMouseEvents()
        {
            this.mapControl.PreviewMouseLeftButtonDown -= OnPreviewLeftMouseDown;
            this.mapControl.PreviewMouseMove -= OnPreviewMouseMove;
            this.mapControl.PreviewMouseLeftButtonUp -= OnPreviewLeftMouseUp;
        }

        private Point ScreenPointToGlobal(Point screenPoint)
        {
            return mapControl.Viewport.ScreenToWorld(screenPoint);
        }
    }
}
