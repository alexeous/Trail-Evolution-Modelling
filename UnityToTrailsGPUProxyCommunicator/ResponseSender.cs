﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class ResponseSender : IDisposable
    {
        private Stream stream;
        private BufferedSerializer serializer;

        public ResponseSender(Stream stream)
        {
            this.stream = stream;
            serializer = new BufferedSerializer(stream);
        }

        public Task SendAsync(Response response)
        {
            return Task.Run(() => Send(response));
        }

        public void Send(Response response)
        {
            lock (stream)
            {
                serializer.Serialize(response);
            }
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
