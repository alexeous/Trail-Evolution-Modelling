using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    [Serializable]
    public abstract class Request
    {
        internal int ID { get; set; }
    }
}
