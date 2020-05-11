using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.UI.Wpf;
using Mapsui.Utilities;
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

        private PolygonTool polygonTool;
        private LineTool lineTool;
        private BoundingAreaTool boundingAreaTool;
        private MapObjectEditing mapObjectEditing;
        private AttractorTool attractorTool;
        private AttractorEditing attractorEditing;
        private Tool[] allTools;

        private Thread computationThread;

        private XmlSaverLoader<SaveFile> saver;

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

            saver.Path = "../../map.tem";
            LoadFromSaveFile(saver.ReadFromFile(saver.Path));
        }

        private void InitLayers()
        {
            mapObjectLayer = new MapObjectLayer();
            boundingAreaLayer = new BoundingAreaLayer();
            attractorLayer = new WritableLayer();
        }

        private void InitializeMapControl()
        {
            mapControl.Map.Layers.Add(OpenStreetMap.CreateTileLayer());

            mapControl.Map.Layers.Add(mapObjectLayer);
            mapControl.Map.Layers.Add(attractorLayer);
            mapControl.Map.Layers.Add(boundingAreaLayer);

            mapControl.MouseLeftButtonDown += OnMapLeftClick;
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

        private ContextMenu CreateMapObjectContextMenu(IMapObject iMapObject)
        {
            var contextMenu = new ContextMenu();
            contextMenu.Items.Add(CreateModifyItem());
            contextMenu.Items.Add(CreateRemoveItem());
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
            ///////// END BUTTONS CREATION LOCAL FUNCS
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
            var clickWorldPos = mapControl.Viewport.ScreenToWorld(clickScreenPos);
            IEnumerable<IMapObject> mapObjects = GetFeaturesAt(clickWorldPos);

            int count = mapObjects.Count();
            if (count == 0)
            {
                return;
            }
            if (count == 1)
            {
                OnMapObjectRightClick(mapObjects.First());
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
                    item.Click += (s, ee) => OnMapObjectRightClick(mapObject);
                    contextMenu.Items.Add(item);
                }
                contextMenu.IsOpen = true;
            }
        }

        private IEnumerable<IMapObject> GetFeaturesAt(Point point)
        {
            double tolerance = mapControl.Viewport.Resolution * 5;

            var boundingBox = new BoundingBox(point, point);
            return mapObjectLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution)
                .Concat(boundingAreaLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution))
                .Concat(attractorLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution))
                .OfType<IMapObject>()
                .Where(p => p.Distance(point) <= tolerance);
        }

        private bool EndAllTools()
        {
            bool any = false;
            foreach (var tool in allTools)
            {
                any |= tool.End();
            }

            RefreshButtons();

            return any;
        }

        private void OnMapLeftClick(object sender, MouseButtonEventArgs e)
        {
            UnhighlightAllMapObjects();
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
            buttonStart.IsEnabled = boundingAreaTool.BoundingArea != null;
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

        private void OnOpenFileClick(object sender, RoutedEventArgs e)
        {
            SaveFile save = saver.Load();
            if (save != null)
                LoadFromSaveFile(save);
        }

        private void OnSaveFileClick(object sender, RoutedEventArgs e)
        {
            saver.Save(PrepareSaveFile());
        }

        private void OnSaveFileAsClick(object sender, RoutedEventArgs e)
        {
            saver.SaveAs(PrepareSaveFile());
        }

        private void LoadFromSaveFile(SaveFile saveFile)
        {
            boundingAreaTool.BoundingArea = saveFile.World.BoundingArea;
            mapObjectLayer.Clear();
            mapObjectLayer.AddRange(saveFile.World.MapObjects);
            attractorLayer.Clear();
            attractorLayer.AddRange(saveFile.World.AttractorObjects);
            mapControl.ZoomToBox(saveFile.Viewport.TopLeft, saveFile.Viewport.BottomRight);

            RefreshLayers();
            RefreshButtons();
        }

        private SaveFile PrepareSaveFile()
        {
            return new SaveFile
            {
                World = GetWorld(),
                Viewport = mapControl.Viewport.Extent
            };
        }

        private void OnStartClick(object sender, RoutedEventArgs e)
        {
            World world = GetWorld();
            var computation = new TrailsComputation(world);
            computation.ProgressChanged += (_s, _e) => Dispatcher.Invoke(
                () => textBoxComputationStage.Text = computation.CurrentStage
            );

            computationThread = new Thread(() =>
            {
                try
                {
                    Dispatcher.Invoke(() => gridComputationIsOn.Visibility = Visibility.Visible);
                    TrailsComputationsOutput output = computation.Run();
                    Graph graph = output.Graph;

                    Dispatcher.Invoke(() =>
                    {
                        var edgeLayer = new WritableLayer();

                        Color minCol = Color.FromArgb(255, 0, 255, 0);
                        Color maxCol = Color.Red;
                        const float minW = 1f;
                        const float maxW = 3.4f;

                        foreach (var edge in graph.Edges)
                        {
                            Point pos1 = graph.GetNodePosition(edge.Node1).ToMapsui();
                            Point pos2 = graph.GetNodePosition(edge.Node2).ToMapsui();
                            float t = (edge.Weight - edge.Trampledness - minW) / (maxW - minW);
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
                        mapControl.Map.Layers.Add(edgeLayer);

                        int Lerp(int a, int b, float t)
                        {
                            t = Math.Max(0, Math.Min(t, 1));
                            return (int)(a * (1 - t) + b * t);
                        }
                    });
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
                    Dispatcher.Invoke(() => gridComputationIsOn.Visibility = Visibility.Collapsed);
                }
            });
            computationThread.IsBackground = true;
            computationThread.Start();


        }

        private void OnAttractorButtonClick(object sender, RoutedEventArgs e)
        {
            attractorTool.AttractorType = (AttractorType)((FrameworkElement)sender).Tag;
            attractorTool.AttractorPerformance = AttractorPerformance.Normal;
            attractorTool.Begin();
        }

        private void OnLargeAttractorButtonClick(object sender, RoutedEventArgs e)
        {
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
            computationThread?.Abort();
        }
    }
}
