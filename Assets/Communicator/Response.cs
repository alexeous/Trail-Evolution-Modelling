using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    [Serializable]
    public abstract class Response
    {
        internal int RequestID { get; }

        protected Response() { }

        public Response(Request request)
        {
            RequestID = request.ID;
        }
    }
}
