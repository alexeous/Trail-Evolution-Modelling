using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class BufferedSerializer : IDisposable
    {
        private MemoryStream memoryStream;
        private BinaryWriter writer;
        private BinaryFormatter formatter;

        public BufferedSerializer(Stream stream)
        {
            memoryStream = new MemoryStream();
            writer = new BinaryWriter(stream);
            formatter = new BinaryFormatter();
        }

        public void Serialize(object obj)
        {
            formatter.Serialize(memoryStream, obj);
            writer.Write(memoryStream.Position);
            writer.Flush();
            memoryStream.CopyTo(writer.BaseStream);
            memoryStream.Position = 0;
            writer.BaseStream.Flush();
        }

        #region IDisposable Support
        private bool disposedValue = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    memoryStream?.Dispose();
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
