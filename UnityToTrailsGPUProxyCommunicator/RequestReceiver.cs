using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class RequestReceiver
    {
        private Stream stream;
        private BinaryFormatter formatter;

        public RequestReceiver(Stream stream)
        {
            this.stream = stream;
            formatter = new BinaryFormatter();
        }

        public Task<Request> ReceiveAsync()
        {
            return Task.Run(Receive);
        }

        public Request Receive()
        {
            return (Request)formatter.Deserialize(stream);
        }
    }
}
