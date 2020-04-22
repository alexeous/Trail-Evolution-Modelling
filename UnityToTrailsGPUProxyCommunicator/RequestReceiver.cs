using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class RequestReceiver : IDisposable
    {
        private Stream stream;
        private BufferedDeserializer deserializer;

        public RequestReceiver(Stream stream)
        {
            this.stream = stream;
            deserializer = new BufferedDeserializer(stream);
        }

        public Task<Request> ReceiveAsync()
        {
            return Task.Run(Receive);
        }

        public Request Receive()
        {
            lock (stream) 
            {
                return deserializer.Deserialize<Request>();
            }
        }

        #region IDisposable Support
        private bool disposedValue = false; // Для определения избыточных вызовов

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    deserializer?.Dispose();
                }
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }
        #endregion
    }
}
