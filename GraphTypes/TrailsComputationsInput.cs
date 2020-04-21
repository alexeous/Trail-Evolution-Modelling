using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GraphTypes
{
    [Serializable]
    public class TrailsComputationsInput
    {
        public Graph Graph { get; set; }
        public Attractor[] Attractors { get; set; }
    }
}
