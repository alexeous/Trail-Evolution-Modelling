using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    [Serializable]
    public class ResultResponse : Response
    {
        public object Result { get; private set; }

        public ResultResponse(Request request, object result)
            : base(request)
        {
            Result = result;
        }
    }
}
