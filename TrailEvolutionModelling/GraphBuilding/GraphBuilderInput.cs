using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TrailEvolutionModelling.MapObjects;

namespace TrailEvolutionModelling.GraphBuilding
{
    class GraphBuilderInput
    {
        public World World { get; set; }
        public float DesiredStep { get; set; }
    }
}
