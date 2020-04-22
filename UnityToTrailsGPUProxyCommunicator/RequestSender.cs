using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class RequestSender : IDisposable
    {
        private static int requestIdCounter;

        private Stream stream;
        private BufferedSerializer serializer;

        public RequestSender(Stream stream)
        {
            this.stream = stream;
            serializer = new BufferedSerializer(stream);
        }

        public Task SendAsync(Request request)
        {
            return Task.Run(() =>
            {
                Send(request);
            });
        }

        public void Send(Request request)
        {
            lock (stream)
            {
                AssignNextID(request);
                serializer.Serialize(request);
            }
        }

        private void AssignNextID(Request request)
        {
            request.ID = requestIdCounter++;
        }

        #region IDisposable Support
        private bool disposedValue = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    serializer?.Dispose();
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
