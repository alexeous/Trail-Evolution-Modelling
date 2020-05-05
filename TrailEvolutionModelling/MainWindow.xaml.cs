using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using Mapsui.Geometries;
using Mapsui.Layers;
using Mapsui.Projection;
using Mapsui.Providers;
using Mapsui.Styles;
using Mapsui.UI.Wpf;
using Mapsui.Utilities;
using TrailEvolutionModelling.Attractors;
using TrailEvolutionModelling.EditorTools;
using TrailEvolutionModelling.Files;
using TrailEvolutionModelling.GPUProxy;
using TrailEvolutionModelling.GraphBuilding;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.Layers;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Styles;
using TrailEvolutionModelling.Util;
using Point = Mapsui.Geometries.Point;
using Polygon = TrailEvolutionModelling.MapObjects.Polygon;

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
        private Tool[] allTools;

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
            mapObjectLayer = new MapObjectLayer();
            boundingAreaLayer = new BoundingAreaLayer();
            attractorLayer = new WritableLayer();
            InitializeMapControl();
            //polygonLayer.AddRange(polygonStorage.Polygons);
            InitTools();
            InitSaver();

            RefreshButtons();
            ZoomToPoint(new Point(9231625, 7402608));
        }

        private void InitTools()
        {
            polygonTool = new PolygonTool(mapControl, mapObjectLayer);
            lineTool = new LineTool(mapControl, mapObjectLayer);
            boundingAreaTool = new BoundingAreaTool(mapControl, boundingAreaLayer);
            mapObjectEditing = new MapObjectEditing(mapControl, mapObjectLayer);
            attractorTool = new AttractorTool(mapControl, attractorLayer);
            
            allTools = new Tool[] { 
                polygonTool, lineTool, mapObjectEditing,
                boundingAreaTool, attractorTool
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

        private void InitializeMapControl()
        {
            mapControl.Map.Layers.Add(OpenStreetMap.CreateTileLayer());

            mapControl.Map.Layers.Add(mapObjectLayer);
            mapControl.Map.Layers.Add(attractorLayer);
            mapControl.Map.Layers.Add(boundingAreaLayer);

            mapControl.MouseLeftButtonDown += OnMapLeftClick;
            mapControl.MouseRightButtonDown += OnMapRightClick;
        }

        private ContextMenu CreateMapObjectContextMenu(MapObject mapObject)
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
                    mapObjectEditing.TargetObject = mapObject;
                    mapObjectEditing.Begin();
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
                    if (mapObject == boundingAreaTool.BoundingArea)
                    {
                        boundingAreaTool.Remove();
                        RefreshButtons();
                    }
                    else
                    {
                        mapObjectLayer.TryRemove(mapObject);
                        mapObjectLayer.Refresh();
                    }
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
            IEnumerable<MapObject> mapObjects = GetMapObjectsAt(clickWorldPos);

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
                    item.GotFocus += (s, ee) => { mapObject.Highlighter.IsHighlighted = true; mapObjectLayer.Refresh(); };
                    item.LostFocus += (s, ee) => { mapObject.Highlighter.IsHighlighted = false; mapObjectLayer.Refresh(); };
                    item.Click += (s, ee) => OnMapObjectRightClick(mapObject);
                    contextMenu.Items.Add(item);
                }
                contextMenu.IsOpen = true;
            }
        }

        private IEnumerable<MapObject> GetMapObjectsAt(Point point)
        {
            const double tolerance = 15;

            var boundingBox = new BoundingBox(point, point);
            return mapObjectLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution)
                .Concat(boundingAreaLayer.GetFeaturesInView(boundingBox, mapControl.Viewport.Resolution))
                .OfType<MapObject>()
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

        private void OnMapObjectRightClick(MapObject mapObject)
        {
            mapObject.Highlighter.IsHighlighted = true;
            mapObjectLayer.Refresh();

            var contextMenu = CreateMapObjectContextMenu(mapObject);
            contextMenu.IsOpen = true;
        }

        private void UnhighlightAllMapObjects()
        {
            var mapObjs = mapObjectLayer.GetFeatures().Concat(boundingAreaLayer.GetFeatures()).OfType<MapObject>();
            foreach (var mapObject in mapObjs)
            {
                mapObject.Highlighter.IsHighlighted = false;
            }
            mapObjectLayer.Refresh();
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
            var graph = GraphBuilder.Build(new GraphBuilderInput
            {
                World = GetWorld(),
                DesiredStep = 50f
            });

            var layer = new WritableLayer();
            layer.Style = new SymbolStyle
            {
                Fill = new Brush(Color.Blue),
                SymbolType = SymbolType.Ellipse,
                SymbolScale = 0.1f
            };
            for (int i = 0; i < graph.Width; i++)
            {
                for (int j = 0; j < graph.Height; j++)
                {
                    Node node = graph.GetNodeAtOrNull(i, j);
                    if (node == null)
                        continue;

                    Point pos = graph.GetNodePosition(node).ToMapsui();
                    layer.Add(new Feature
                    {
                        Geometry = pos
                    });
                }
            }
            mapControl.Map.Layers.Add(layer);

            var edgeLayer = new WritableLayer();

            Color minCol = Color.FromArgb(255, 0, 255, 0);
            Color maxCol = Color.Red;
            const float minW = 1f;
            const float maxW = 3.4f;

            foreach (var edge in graph.Edges)
            {
                Point pos1 = graph.GetNodePosition(edge.Node1).ToMapsui();
                Point pos2 = graph.GetNodePosition(edge.Node2).ToMapsui();
                float t = (edge.Weight - minW) / (maxW - minW);
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
                return (int)(a * (1 - t) + b * t);
            }
        }

        private void OnAttractorButtonClick(object sender, RoutedEventArgs e)
        {
            string tag = (string)((FrameworkElement)sender).Tag;
            attractorTool.AttractorType = (AttractorType)Enum.Parse(typeof(AttractorType), tag);
            attractorTool.IsLarge = false;
            attractorTool.Begin();
        }

        private void OnLargeAttractorButtonClick(object sender, RoutedEventArgs e)
        {
            string tag = (string)((FrameworkElement)sender).Tag;
            attractorTool.AttractorType = (AttractorType)Enum.Parse(typeof(AttractorType), tag);
            attractorTool.IsLarge = true;
            attractorTool.Begin();
        }

        private World GetWorld()
        {
            return new World
            {
                BoundingArea = boundingAreaTool.BoundingArea,
                MapObjects = mapObjectLayer.GetFeatures().OfType<MapObject>().ToArray()
            };
        }
    }
}
