using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using BruTile.Predefined;
using Mapsui.Layers;
using Mapsui.Utilities;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.GPUProxy;

namespace TrailEvolutionModelling
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            mapControl.Map.Layers.Add(OpenStreetMap.CreateTileLayer());
            
            var input = new TrailsComputationsInput
            {
                Attractors = new Attractor[0]
            };
            TrailsComputationsOutput output = TrailsGPUProxy.ComputeTrails(input);
            MessageBox.Show(output.Graph.Height.ToString()); ;
        }
    }
}
