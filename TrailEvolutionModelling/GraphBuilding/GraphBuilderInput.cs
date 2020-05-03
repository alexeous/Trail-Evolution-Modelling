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
        public BoundingAreaPolygon BoundingArea { get; set; }
        public MapObject[] MapObjects { get; set; }
        public float DesiredStep { get; set; }
    }
}
