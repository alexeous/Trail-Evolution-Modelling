using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    [Serializable]
    public class ErrorResponse : Response
    {
        public Exception Exception { get; }

        private ErrorResponse() { }

        public ErrorResponse(Request request, Exception exception)
            : base(request)
        {
            Exception = exception;
        }
    }
}
