using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
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
using Xceed.Wpf.Toolkit;
using Point = Mapsui.Geometries.Point;
using MapsuiPolygon = Mapsui.Geometries.Polygon;

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

        private class WorkingRadiusFeature : Feature 
        {
            public AttractorObject Attractor { get; set; }

            private VectorStyle style;

            public WorkingRadiusFeature(AttractorObject attractor)
            {
                Attractor = attractor;

                style = new VectorStyle();
                Styles.Add(style);

                Update();
            }

            public void Update()
            {
                style.Line.Width = 2;
                Color color = Attractor.GetColor();
                style.Line.Color = color.Scale(0.5f);
                style.Fill.Color = color;

                UpdateCirclePolygon();
            }

            private void UpdateCirclePolygon()
            {
                // Create new circle
                var centerX = Attractor.Position.X;
                var centerY = Attractor.Position.Y;
                var radius = Attractor.WorkingRadius;
                var increment = 2;
                var exteriorRing = new LinearRing();

                for (double angle = 0; angle < 360; angle += increment)
                {
                    var angleRad = angle / 180.0 * Math.PI;
                    exteriorRing.Vertices.Add(new Point(radius * Math.Sin(angleRad) + centerX, radius * Math.Cos(angleRad) + centerY));
                }

                Geometry = new MapsuiPolygon(exteriorRing);
            }
        }

        public AttractorObject TargetObject { get; set; }

        private MapControl mapControl;
        private WritableLayer targetLayer;
        
        private UIElement editingUIRoot;
        private ComboBox typeComboBox;
        private ComboBox performanceComboBox;
        private IntegerUpDown workingRadiusUpDown;

        private AttractorObject editedObject;
        private WritableLayer draggingLayer;
        private AttractorDraggingFeature draggingFeature;
        private Point draggingOffset;
        private WritableLayer workingRadiusLayer;
        private WorkingRadiusFeature workingRadiusFeature;

        public AttractorEditing(MapControl mapControl, WritableLayer targetLayer, UIElement editingUIRoot, ComboBox typeComboBox, ComboBox performanceComboBox, IntegerUpDown workingRadiusUpDown)
        {
            this.mapControl = mapControl;
            this.targetLayer = targetLayer;
            this.editingUIRoot = editingUIRoot;
            this.typeComboBox = typeComboBox;
            this.performanceComboBox = performanceComboBox;
            this.workingRadiusUpDown = workingRadiusUpDown;

            InitLayers();
        }

        protected override void BeginImpl()
        {
            editedObject = TargetObject;

            if (editedObject == null)
            {
                throw new InvalidOperationException($"{nameof(TargetObject)} was not set");
            }

            CreateDraggingFeature();
            CreateWorkingRadiusFeature();
            FillEditingUIControls();
            SubscribeUIEvents();
            SubscribeMouseEvents();
            editingUIRoot.Visibility = Visibility.Visible;
        }

        protected override void EndImpl()
        {
            editingUIRoot.Visibility = Visibility.Collapsed;
            UnsubscribeMouseEvents();
            UnsubscribeUIEvents();
            ClearLayers();
            editedObject = null;
        }

        private void ClearLayers()
        {
            draggingFeature = null;
            draggingLayer.Clear();
            draggingLayer.Refresh();
            
            workingRadiusFeature = null;
            workingRadiusLayer.Clear();
            workingRadiusLayer.Refresh();
        }

        private void InitLayers()
        {
            draggingLayer = new WritableLayer();
            workingRadiusLayer = new WritableLayer { Opacity = 0.3 };
            this.mapControl.Map.Layers.Add(workingRadiusLayer);
            this.mapControl.Map.Layers.Add(draggingLayer);
        }

        private void CreateDraggingFeature()
        {
            draggingLayer.Add(new AttractorDraggingFeature(editedObject));
            draggingLayer.Refresh();
        }

        private void CreateWorkingRadiusFeature()
        {
            workingRadiusLayer.Add(workingRadiusFeature = new WorkingRadiusFeature(editedObject));
            workingRadiusLayer.Refresh();
        }

        private void FillEditingUIControls()
        {
            typeComboBox.SelectedItem = typeComboBox.Items.Cast<ComboBoxItem>().FirstOrDefault(it => editedObject.Type.Equals(it.Tag));
            performanceComboBox.SelectedValue = performanceComboBox.Items.Cast<ComboBoxItem>().FirstOrDefault(it => editedObject.Performance.Equals(it.Tag));
            workingRadiusUpDown.Value = (int)editedObject.WorkingRadius;
        }

        private void SubscribeUIEvents()
        {
            typeComboBox.SelectionChanged += OnTypeComboBoxSelectionChanged;
            performanceComboBox.SelectionChanged += OnPerformanceComboBoxSelectionChanged;
            workingRadiusUpDown.ValueChanged += OnWorkingRadiusUpDownValueChanged;
        }

        private void UnsubscribeUIEvents()
        {
            typeComboBox.SelectionChanged -= OnTypeComboBoxSelectionChanged;
            performanceComboBox.SelectionChanged -= OnPerformanceComboBoxSelectionChanged;
            workingRadiusUpDown.ValueChanged -= OnWorkingRadiusUpDownValueChanged;
        }

        private void SubscribeMouseEvents()
        {
            mapControl.PreviewMouseLeftButtonDown += OnPreviewLeftMouseDown;
            mapControl.PreviewMouseMove += OnPreviewMouseMove;
            mapControl.PreviewMouseLeftButtonUp += OnPreviewLeftMouseUp;
        }

        private void UnsubscribeMouseEvents()
        {
            mapControl.PreviewMouseLeftButtonDown -= OnPreviewLeftMouseDown;
            mapControl.PreviewMouseMove -= OnPreviewMouseMove;
            mapControl.PreviewMouseLeftButtonUp -= OnPreviewLeftMouseUp;
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
                Refresh();
            }
        }

        private void OnPreviewLeftMouseUp(object sender, MouseButtonEventArgs e)
        {
            draggingFeature = null;
        }

        private IEnumerable<IFeature> GetFeaturesAtScreenPoint(Point point)
        {
            var worldPoint = ScreenPointToGlobal(point);
            double extentReduction = DraggingLayer.Scale / 3.5f;
            double resolution = mapControl.Viewport.Resolution * extentReduction;
            var features = draggingLayer.GetFeaturesInView(new BoundingBox(worldPoint, worldPoint), resolution);
            return features;
        }

        private Point ScreenPointToGlobal(Point screenPoint)
        {
            return mapControl.Viewport.ScreenToWorld(screenPoint);
        }

        private void OnTypeComboBoxSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            editedObject.Type = (AttractorType)((ComboBoxItem)typeComboBox.SelectedItem).Tag;

            Refresh();
        }

        private void OnPerformanceComboBoxSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            editedObject.Performance = (AttractorPerformance)((ComboBoxItem)performanceComboBox.SelectedItem).Tag;

            Refresh();
        }

        private void OnWorkingRadiusUpDownValueChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            editedObject.WorkingRadius = workingRadiusUpDown.Value ?? 0;

            Refresh();
        }

        private void Refresh()
        {
            workingRadiusFeature.Update();
            workingRadiusLayer.Refresh();
            draggingLayer.Refresh();
            targetLayer.Refresh();
        }
    }
}
