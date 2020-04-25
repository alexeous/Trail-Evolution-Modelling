using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using Ceras;
using Ceras.Helpers;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class RequestSender : IDisposable
    {
        private static int requestIdCounter;

        private object syncObj = new object();
        private Stream stream;
        private CerasSerializer serializer;

        public RequestSender(Stream stream)
        {
            this.stream = stream;
            serializer = new CerasSerializer(new SerializerConfig { DefaultTargets = TargetMember.All });
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
            lock (syncObj)
            {
                AssignNextID(request);
                serializer.WriteToStream(stream, request);
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
                    //serializer?.Dispose();
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
