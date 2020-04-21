using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class RequestSender
    {
        private static int requestIdCounter;

        private Stream stream;
        private BinaryFormatter formatter;

        public RequestSender(Stream stream)
        {
            this.stream = stream;
            formatter = new BinaryFormatter();
        }

        public void Send(Request request)
        {
            lock (stream)
            {
                AssignNextID(request);
                formatter.Serialize(stream, request);
                stream.Flush();
            }
        }

        private void AssignNextID(Request request)
        {
            request.ID = requestIdCounter++;
        }
    }
}
