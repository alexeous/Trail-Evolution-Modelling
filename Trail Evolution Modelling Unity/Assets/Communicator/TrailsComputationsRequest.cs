using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TrailEvolutionModelling.GPUProxyCommunicator;
using TrailEvolutionModelling.GraphTypes;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    [Serializable]
    public class TrailsComputationsRequest : Request
    {
        public TrailsComputationsInput ComputationsInput { get; private set; }

        private TrailsComputationsRequest() { }

        public TrailsComputationsRequest(TrailsComputationsInput computationsInput)
        {
            ComputationsInput = computationsInput;
        }
    }
}
