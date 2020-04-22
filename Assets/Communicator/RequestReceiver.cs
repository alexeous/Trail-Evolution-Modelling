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
    public class RequestReceiver : IDisposable
    {
        private object syncObj = new object();
        private Stream stream;
        private CerasSerializer deserializer;

        public RequestReceiver(Stream stream)
        {
            this.stream = stream;
            deserializer = new CerasSerializer(new SerializerConfig { DefaultTargets = TargetMember.All });
        }

        public async Task<Request> ReceiveAsync()
        {
            return (Request)await deserializer.ReadFromStream(stream);
        }

        //public Request Receive()
        //{
        //    lock (syncObj)
        //    {
        //        return (Request)deserializer.ReadFromStream(stream);
        //    }
        //}

        #region IDisposable Support
        private bool disposedValue = false; // Для определения избыточных вызовов

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    //deserializer?.Dispose();
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
