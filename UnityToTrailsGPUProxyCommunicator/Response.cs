using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator.Internal
{
    [Serializable]
    class Response
    {
        public object Result { get; private set; }

        public Response(object result)
        {
            Result = result;
        }
    }
}
