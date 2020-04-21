using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{

    [Serializable]
    public class ComputationsException : Exception
    {
        public ComputationsException() { }
        public ComputationsException(string message) : base(message) { }
        public ComputationsException(string message, Exception inner) : base(message, inner) { }
        protected ComputationsException(
          System.Runtime.Serialization.SerializationInfo info,
          System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}
