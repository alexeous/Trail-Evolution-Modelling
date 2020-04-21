using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GraphTypes
{
    [Serializable]
    public class Attractor
    {
        public Node Node { get; set; }
        public bool IsSource { get; set; }
        public bool IsDrain { get; set; }
        public float Performance { get; set; }
        public float WorkingRadius { get; set; }
    }
}
