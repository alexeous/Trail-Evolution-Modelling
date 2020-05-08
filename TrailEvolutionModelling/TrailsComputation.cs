using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TrailEvolutionModelling.GPUProxy;
using TrailEvolutionModelling.GraphBuilding;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.MapObjects;

namespace TrailEvolutionModelling
{
    class TrailsComputation
    {
        public event EventHandler ProgressChanged;
        public string CurrentStage { get; private set; }

        private World world;

        public TrailsComputation(World world)
        {
            this.world = world;
        }

        public void Run()
        {
            ReportProgress("Построение графа");
            Graph graph = BuildGraph();
            Attractor[] attractors = GraphBuilder.CreateAttractors(graph, world.AttractorObjects);

            ReportProgress("Симуляция движения пешеходов");
            var computationsInput = new TrailsComputationsInput
            {
                Graph = graph,
                Attractors = attractors
            };
            TrailsComputationsOutput output = TrailsGPUProxy.ComputeTrails(computationsInput);
        }

        private void ReportProgress(string message)
        {
            CurrentStage = message;
            ProgressChanged?.Invoke(this, EventArgs.Empty);
        }

        private Graph BuildGraph()
        {
            return GraphBuilder.Build(new GraphBuilderInput
            {
                World = world,
                DesiredStep = 2.4f
            });
        }
    }
}
