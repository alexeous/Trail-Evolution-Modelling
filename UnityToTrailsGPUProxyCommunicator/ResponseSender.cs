using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class ResponseSender
    {
        private Stream stream;
        private BinaryFormatter formatter;

        public ResponseSender(Stream stream)
        {
            this.stream = stream;
            formatter = new BinaryFormatter();
        }

        public void Send(Response response)
        {
            lock (formatter)
            {
                formatter.Serialize(stream, response);
                stream.Flush();
            }
        }
    }
}
