using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator.Internal
{
    [Serializable]
    abstract class Request
    {
        public abstract Response Process();
    }
}
