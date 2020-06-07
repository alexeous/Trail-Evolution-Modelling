using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Mapsui;
using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.UI.Wpf;
using Mapsui.Utilities;
using Microsoft.Win32;
using TrailEvolutionModelling.Attractors;
using TrailEvolutionModelling.EditorTools;
using TrailEvolutionModelling.Files;
using TrailEvolutionModelling.GPUProxy;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.Layers;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Styles;
using TrailEvolutionModelling.Util;
using Point = Mapsui.Geometries.Point;

namespace TrailEvolutionModelling
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private MapObjectLayer mapObjectLayer;
        private BoundingAreaLayer boundingAreaLayer;
        private WritableLayer attractorLayer;
        private WritableLayer edgeLayer;
        private RasterizingLayer edgeRasterizingLayer;

        private PolygonTool polygonTool;
        private LineTool lineTool;
        private BoundingAreaTool boundingAreaTool;
        private MapObjectEditing mapObjectEditing;
        private AttractorTool attractorTool;
        private AttractorEditing attractorEditing;
        private Tool[] allTools;

        private System.Windows.Point mouseDownPos;

        private Thread computationThread;
        private TrailsComputation computation;

        private Trampledness trampledness;

        private XmlSaverLoader<SaveFile> saver;

        private bool unsavedChanges = false;

        private Button[] MapObjectButtons => new[]
        {
            buttonPavedPath,
            buttonCarRoad,
            buttonVegetation,
            buttonWalkthroughableFence,
            buttonBuilding,
            buttonFence,
            buttonWater,
            buttonOtherUnwalkthroughable
        };

        public MainWindow()
        {
            InitializeComponent();
            InitMapObjectButtons();
            InitAttractorButtonsTags();
            InitAttractorEditingControlsTags();

            mapControl.Map.Layers.Add(OpenStreetMap.CreateTileLayer());
            mapControl.Renderer.StyleRenderers.Add(typeof(BoundingAreaStyle), new BoundingAreaRenderer());
        }

        private void InitMapObjectButtons()
        {
            foreach (var button in MapObjectButtons)
            {
                string text = GetAreaTypeFromTag(button).DisplayedName;
                button.Content = new TextBlock
                {
                    Text = text,
                    TextWrapping = TextWrapping.Wrap,
                    TextAlignment = TextAlignment.Left
                };
            }
        }

        private void InitAttractorButtonsTags()
        {
            buttonAttractorUniversal.Tag = AttractorType.Universal;
            buttonAttractorSource.Tag = AttractorType.Source;
            buttonAttractorDrain.Tag = AttractorType.Drain;
            buttonAttractorUniversalLarge.Tag = AttractorType.Universal;
            buttonAttractorSourceLarge.Tag = AttractorType.Source;
            buttonAttractorDrainLarge.Tag = AttractorType.Drain;
        }

        private void InitAttractorEditingControlsTags()
        {
            itemAttractorTypeUniversal.Tag = AttractorType.Universal;
            itemAttractorTypeSource.Tag = AttractorType.Source;
            itemAttractorTypeDrain.Tag = AttractorType.Drain;

            itemAttractorPerfNormal.Tag = AttractorPerformance.Normal;
            itemAttractorPerfHigh.Tag = AttractorPerformance.High;
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            base.OnKeyDown(e);
            if (e.Key == Key.Escape)
            {
                EndAllTools();
            }
        }

        private void OnWindowLoaded(object sender, RoutedEventArgs e)
        {
            InitLayers();
            InitializeMapControl();
            InitTools();
            InitSaver();

            RefreshButtons();
            ZoomToPoint(new Point(9231625, 7402608));

            checkBoxShowTrampledness.IsChecked = true;
        }

        private void InitLayers()
        {
            mapObjectLayer = new MapObjectLayer();
            boundingAreaLayer = new BoundingAreaLayer();
            attractorLayer = new WritableLayer();

            edgeLayer = new WritableLayer();
            edgeRasterizingLayer = new RasterizingLayer(edgeLayer,
                delayBeforeRasterize: 0, renderResolutionMultiplier: 1,
                rasterizer: null, overscanRatio: 2);
        }

        private void InitializeMapControl()
        {
            var resolutions = mapControl.Map.Resolutions;
            mapControl.Map.Limiter.ZoomLimits = new Mapsui.UI.MinMax(
                resolutions.Min() / 8, resolutions.Max());

            mapControl.Map.Layers.Add(OpenStreetMap.CreateTileLayer());

            mapControl.Map.Layers.Add(mapObjectLayer);
            mapControl.Map.Layers.Add(attractorLayer);
            mapControl.Map.Layers.Add(boundingAreaLayer);
            mapControl.Map.Layers.Add(edgeRasterizingLayer);

            mapControl.MouseLeftButtonDown += OnMapLeftDown;
            mapControl.MouseLeftButtonUp += OnMapLeftUp;
            mapControl.MouseRightButtonDown += OnMapRightClick;
        }

        private void InitTools()
        {
            polygonTool = new PolygonTool(mapControl, mapObjectLayer);
            lineTool = new LineTool(mapControl, mapObjectLayer);
            mapObjectEditing = new MapObjectEditing(mapControl, mapObjectLayer);
            boundingAreaTool = new BoundingAreaTool(mapControl, boundingAreaLayer);
            attractorTool = new AttractorTool(mapControl, attractorLayer);
            attractorEditing = new AttractorEditing(mapControl, attractorLayer,
                gridAttractorEditing, comboBoxAttractorType, comboBoxAttractorPerformance,
                upDownAttractorRadius);

            allTools = new Tool[] {
                polygonTool, lineTool, mapObjectEditing,
                boundingAreaTool, attractorTool, attractorEditing
            };

            foreach (var tool in allTools)
            {
                tool.OnBegin += OnToolBegin;
                tool.OnEnd += OnToolEnd;
            }
        }

        private void InitSaver()
        {
            saver = new XmlSaverLoader<SaveFile>("tem");
        }

        private void ZoomToPoint(Point center)
        {
            var extent = new Point(1000, 1000);
            mapControl.ZoomToBox(center - extent, center + extent);
        }

        private void OnToolBegin(object sender, EventArgs e)
        {
            unsavedChanges = true;
        }

        private void OnToolEnd(object sender, EventArgs e)
        {
            RefreshButtons();
        }

        private ContextMenu CreateMapObjectContextMenu(IMapObject iMapObject)
        {
            var contextMenu = new ContextMenu();
            contextMenu.Items.Add(CreateModifyItem());
            contextMenu.Items.Add(CreateRemoveItem());
            if (iMapObject is MapObject mapObj && !(iMapObject is BoundingAreaPolygon)) {
                contextMenu.Items.Add(CreateConvertItem(mapObj)); 
            }
            return contextMenu;

            ///////// BUTTONS CREATION LOCAL FUNCS
            MenuItem CreateModifyItem()
            {
                var item = new MenuItem
                {
                    Header = "Редактировать",
                    Icon = BitmapResources.LoadImage("EditPolygon.png")
                };
                item.Click += (s, e) =>
                {
                    UnhighlightAllMapObjects();
                    EndAllTools();
                    StartMapObjectEditing(iMapObject);
                };
                return item;
            }

            MenuItem CreateRemoveItem()
            {
                var item = new MenuItem
                {
                    Header = "Удалить",
                    Icon = BitmapResources.LoadImage("Delete.png")
                };
                item.Click += (s, e) =>
                {
                    if (iMapObject is MapObject mapObject)
                    {
                        if (iMapObject == boundingAreaTool.BoundingArea)
                        {
                            boundingAreaTool.Remove();
                            RefreshButtons();
                        }
                        else
                        {
                            mapObjectLayer.TryRemove(mapObject);
                        }
                    }
                    else if (iMapObject is AttractorObject attractor)
                    {
                        attractorLayer.TryRemove(attractor);
                        RefreshButtons();
                    }
                    RefreshLayers();
                };
                return item;
            }

            MenuItem CreateConvertItem(MapObject mapObject)
            {
                var parent = new MenuItem
                {
                    Header = "Преобразовать в..."
                };
                var convertOptions = AreaTypes.All
                    .Where(x => x.GeometryType == mapObject.AreaType.GeometryType);
                foreach (var areaType in convertOptions)
                {
                    var item = new MenuItem
                    {
                        Header = areaType.DisplayedName
                    };
                    item.Click += (s, e) =>
                    {
                        mapObject.AreaType = areaType;
                        RefreshLayers();
                    };
                    parent.Items.Add(item);
                }
                return parent;
            }
            ///////// END BUTTONS CREATION LOCAL FUNCS
        }

        private void StartMapObjectEditing(IMapObject iMapObject)
        {
            if (iMapObject is MapObject mapObject)
            {
                mapObjectEditing.TargetObject = mapObject;
                mapObjectEditing.Begin();
            }
            else if (iMapObject is AttractorObject attractor)
            {
                attractorEditing.TargetObject = attractor;
                attractorEditing.Begin();
            }
        }

        private void OnMapLeftUp(object sender, MouseButtonEventArgs e)
        {
            const double tolerance = 5;
            var pos = e.GetPosition(mapControl);
            if (Math.Abs(pos.X - mouseDownPos.X) > tolerance ||
                Math.Abs(pos.Y - mouseDownPos.Y) > tolerance)
            {
                return;
            }

            UnhighlightAllMapObjects();
            if (allTools.All(t => !t.IsActive))
            {
                var clickScreenPos = e.GetPosition(mapControl).ToMapsui();
                TrySelectMapObjectAt(clickScreenPos, StartMapObjectEditing);
            }
        }

        private void OnMapRightClick(object sender, MouseButtonEventArgs e)
        {
            UnhighlightAllMapObjects();

            bool anyToolWasAcitve = EndAllTools();
            if (anyToolWasAcitve)
            {
                return;
            }

            var clickScreenPos = e.GetPosition(mapControl).ToMapsui();
            TrySelectMapObjectAt(clickScreenPos, OnMapObjectRightClick);
        }

        private void TrySelectMapObjectAt(Point clickScreenPos, Action<IMapObject> actionOnSelect)
        {
            var clickWorldPos = mapControl.Viewport.ScreenToWorld(clickScreenPos);
            IEnumerable<IMapObject> mapObjects = GetFeaturesAt(clickWorldPos);

            int count = mapObjects.Count();
            if (count == 0)
            {
                return;
            }
            if (count == 1)
            {
                actionOnSelect(mapObjects.First());
            }
            else
            {
                var contextMenu = new ContextMenu();
                contextMenu.Items.Add(new Label
                {
                    Content = "Выберите объект:",
                    IsEnabled = false
                });
                foreach (var mapObject in mapObjects)
                {
                    var item = new MenuItem
                    {
                        Header = mapObject.DisplayedName
                    };
                    item.GotFocus += (s, ee) => { mapObject.Highlighter.IsHighlighted = true; RefreshLayers(); };
                    item.LostFocus += (s, ee) => { mapObject.Highlighter.IsHighlighted = false; RefreshLayers(); };
                    item.Click += (s, ee) => actionOnSelect(mapObject);
                    contextMenu.Items.Add(item);
                }
                contextMenu.IsOpen = true;
            }
        }

        private IEnumerable<IMapObject> GetFeaturesAt(Point point)
        {
            var boundingBox = new BoundingBox(point, point);
            return mapObjectLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution)
                .Concat(boundingAreaLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution))
                .Concat(attractorLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution))
                .OfType<IMapObject>()
                .Where(p => p.IsMouseOver(point, mapControl));
        }

        private bool EndAllTools()
        {
            bool any = false;
            foreach (var tool in allTools)
            {
                any |= tool.End();
            }

            return any;
        }

        private void OnMapLeftDown(object sender, MouseButtonEventArgs e)
        {
            mouseDownPos = e.GetPosition(mapControl);
        }

        private void OnMapObjectRightClick(IMapObject mapObject)
        {
            mapObject.Highlighter.IsHighlighted = true;
            RefreshLayers();

            var contextMenu = CreateMapObjectContextMenu(mapObject);
            contextMenu.IsOpen = true;
        }

        private void UnhighlightAllMapObjects()
        {
            var mapObjs = mapControl.Map.Layers
                .OfType<WritableLayer>()
                .SelectMany(layer => layer.GetFeatures())
                .OfType<IMapObject>();
            foreach (var mapObject in mapObjs)
            {
                mapObject.Highlighter.IsHighlighted = false;
            }
            RefreshLayers();
        }

        private void OnPolygonDrawClick(object sender, RoutedEventArgs e)
        {
            EndAllTools();

            polygonTool.AreaType = GetAreaTypeFromTag(sender);
            polygonTool.Begin();
        }

        private void OnLineDrawClick(object sender, RoutedEventArgs e)
        {
            EndAllTools();

            lineTool.AreaType = GetAreaTypeFromTag(sender);
            lineTool.Begin();
        }

        private void OnBoundingAreaToolClicked(object sender, RoutedEventArgs e)
        {
            EndAllTools();

            boundingAreaTool.Begin();
        }

        private void RefreshButtons()
        {
            RefreshBoundingAreaToolButton();
            RefreshStartButton();
        }

        private void RefreshBoundingAreaToolButton()
        {
            buttonBoundingArea.IsEnabled = boundingAreaTool.BoundingArea == null;
        }

        private void RefreshStartButton()
        {
            buttonStart.IsEnabled = boundingAreaTool.BoundingArea != null &&
                attractorLayer.GetFeatures().Count() > 1;
        }

        private void RefreshLayers()
        {
            foreach (var layer in mapControl.Map.Layers)
            {
                layer.Refresh();
            }
        }

        private static AreaType GetAreaTypeFromTag(object element)
        {
            string areaTypeName = (string)((FrameworkElement)element).Tag;
            return AreaTypes.GetByName(areaTypeName);
        }

        private void OnNewFileClick(object sender, RoutedEventArgs e)
        {
            if (ConfirmOverrideUnsavedChanges())
                NewFile();
        }

        private void OnOpenFileClick(object sender, RoutedEventArgs e)
        {
            SaveFile save = saver.Load();
            if (save != null)
                if (ConfirmOverrideUnsavedChanges())
                    LoadFromSaveFile(save);
        }

        private void OnSaveFileClick(object sender, RoutedEventArgs e)
        {
            saver.Save(PrepareSaveFile());
            unsavedChanges = false;
        }

        private void OnSaveFileAsClick(object sender, RoutedEventArgs e)
        {
            saver.SaveAs(PrepareSaveFile());
            unsavedChanges = false;
        }


        private void OnExportImageClick(object sender, RoutedEventArgs e)
        {
            ExportImage();
        }

        private void NewFile()
        {
            ClearAll();
            unsavedChanges = false;
        }

        private bool ConfirmOverrideUnsavedChanges()
        {
            if (!unsavedChanges)
                return true;

            const string text = "Несохранённые изменения могут быть утеряны. Продолжить?";
            const string caption = "Внимание! Несохранённые изменения!";
            var result = MessageBox.Show(text, caption, MessageBoxButton.YesNo, MessageBoxImage.Warning);
            return result == MessageBoxResult.Yes;
        }

        private void LoadFromSaveFile(SaveFile saveFile)
        {
            ClearAll();

            boundingAreaTool.BoundingArea = saveFile.World.BoundingArea;
            mapObjectLayer.AddRange(saveFile.World.MapObjects);
            attractorLayer.AddRange(saveFile.World.AttractorObjects);

            this.trampledness = saveFile.Trampledness;
            DrawTrampledness();
            checkBoxShowTrampledness.IsChecked = true;

            mapControl.ZoomToBox(saveFile.Viewport.TopLeft, saveFile.Viewport.BottomRight);

            RefreshLayers();
            RefreshButtons();

            unsavedChanges = false;
        }

        private void ClearAll()
        {
            EndAllTools();
            ClearTrampledness();
            boundingAreaTool.BoundingArea = null;
            mapObjectLayer.Clear();
            attractorLayer.Clear();
            RefreshLayers();
            RefreshButtons();
            mapControl.Refresh();
        }

        private SaveFile PrepareSaveFile()
        {
            return new SaveFile
            {
                World = GetWorld(),
                Viewport = mapControl.Viewport.Extent,
                Trampledness = trampledness
            };
        }

        private void ExportImage()
        {
            var dialog = new SaveFileDialog
            {
                Filter = "png|*.png"
            };
            if (dialog.ShowDialog() != true)
                return;


            IReadOnlyViewport viewport;
            BoundingBox boundingBox = boundingAreaTool.BoundingArea?.Geometry?.BoundingBox;
            if (boundingBox != null)
            {
                viewport = new Viewport
                {
                    Center = boundingBox.Centroid,
                    Width = boundingBox.Width,
                    Height = boundingBox.Height
                };
            }
            else
            {
                viewport = mapControl.Viewport;
            }

            try
            {
                ILayer[] layers = { mapObjectLayer, edgeLayer };
                using (var bitmap = mapControl.Renderer.RenderToBitmapStream(viewport, layers, null, 3))
                {
                    File.WriteAllBytes(dialog.FileName, bitmap.ToArray());
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("При экспорте изображения возникла ошибка:\n" + ex.Message, "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void OnStartClick(object sender, RoutedEventArgs e)
        {
            EndAllTools();
            buttonStart.IsEnabled = false;

            World world = GetWorld();
            computation = new TrailsComputation(world);
            computation.ProgressChanged += (_s, _e) => Dispatcher.Invoke(
                () => {
                    if (computation != null)
                        textBoxComputationStage.Text = computation.CurrentStage;
                }
            );
            computation.CanGiveUnripeResult += (_s, _e) => Dispatcher.Invoke(
                () => buttonGiveUnripeResult.Visibility = Visibility.Visible
            );

            computationThread = new Thread(() =>
            {
                try
                {
                    Dispatcher.Invoke(ShowComputationsIsOnGrid);
                    TrailsComputationsOutput output = computation.Run();
                    trampledness = new Trampledness(output.Graph);
                    unsavedChanges = true;
                    Dispatcher.Invoke(DrawTrampledness);
                }
                catch (IsolatedAttractorsException)
                {
                    MessageBox.Show("Обнаружены изолированные точки", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                catch (ThreadAbortException)
                {
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.ToString(), "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    Dispatcher.Invoke(() => {
                        gridComputationIsOn.Visibility = Visibility.Collapsed;
                        buttonStart.IsEnabled = true;
                    });
                    computation = null;
                    computationThread = null;
                }
            });
            computationThread.IsBackground = true;
            computationThread.Start();
        }

        private void ShowComputationsIsOnGrid()
        {
            buttonGiveUnripeResult.Visibility = Visibility.Collapsed;
            gridComputationIsOn.Visibility = Visibility.Visible;
        }

        private void ClearTrampledness()
        {
            trampledness = null;
            edgeLayer.Clear();
        }

        private void DrawTrampledness()
        {
            if (trampledness == null)
                return;

            edgeLayer.Clear();

            Color minCol = Color.Red;
            Color maxCol = Color.FromArgb(255, 0, 255, 0);

            foreach (var edge in trampledness)
            {
                Point pos1 = new Point(edge.X1, edge.Y1);
                Point pos2 = new Point(edge.X2, edge.Y2);
                float t = edge.Trampledness;
                Color color = Color.FromArgb(255, Lerp(minCol.R, maxCol.R, t), Lerp(minCol.G, maxCol.G, t), Lerp(minCol.B, maxCol.B, t));
                edgeLayer.Add(new Feature
                {
                    Geometry = new LineString(new[] { pos1, pos2 }),
                    Styles = new[]
                    {
                        new VectorStyle { Line = new Pen(color, 1) }
                    }
                });
            }

            edgeLayer.Refresh();
            mapControl.Refresh();
            checkBoxShowTrampledness.IsChecked = true;

            int Lerp(int a, int b, float t)
            {
                t = Math.Max(0, Math.Min(t, 1));
                return (int)(a * (1 - t) + b * t);
            }
        }

        private void OnAttractorButtonClick(object sender, RoutedEventArgs e)
        {
            EndAllTools();
            attractorTool.AttractorType = (AttractorType)((FrameworkElement)sender).Tag;
            attractorTool.AttractorPerformance = AttractorPerformance.Normal;
            attractorTool.Begin();
        }

        private void OnLargeAttractorButtonClick(object sender, RoutedEventArgs e)
        {
            EndAllTools();
            attractorTool.AttractorType = (AttractorType)((FrameworkElement)sender).Tag;
            attractorTool.AttractorPerformance = AttractorPerformance.High;
            attractorTool.Begin();
        }

        private World GetWorld()
        {
            return new World
            {
                BoundingArea = boundingAreaTool.BoundingArea,
                MapObjects = mapObjectLayer.GetFeatures().OfType<MapObject>().ToArray(),
                AttractorObjects = attractorLayer.GetFeatures().OfType<AttractorObject>().ToArray()
            };
        }

        private void OnCancelComputationClick(object sender, RoutedEventArgs e)
        {
            if (MessageBox.Show("Вы уверены, что хотите прервать вычисления? Операция необратима.",
                "Отменить вычисления?", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
            {
                computationThread?.Abort();
            }
        }

        private void OnGiveUnripeResultClick(object sender, RoutedEventArgs e)
        {
            if (MessageBox.Show("Вы уверены, что хотите получить незаконченный результат? " +
                "Продолжить вычисления с данного места будет невозможно.",
                "Выдать результат досрочно?", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
            {
                var localComp = computation;
                if (localComp != null)
                {
                    localComp.GiveUnripeResultFlag = true;
                }
                computationThread?.Abort();
            }
        }

        private void OnShowTramplednessToggled(object sender, RoutedEventArgs e)
        {
            if (checkBoxShowTrampledness.IsChecked == true)
            {
                if (!mapControl.Map.Layers.Contains(edgeRasterizingLayer))
                    mapControl.Map.Layers.Add(edgeRasterizingLayer);
            }
            else
            {
                mapControl.Map.Layers.Remove(edgeRasterizingLayer);
            }
        }
    }
}
