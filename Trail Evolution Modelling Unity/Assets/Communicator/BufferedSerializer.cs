using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using Ceras;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class BufferedSerializer : IDisposable
    {
        private MemoryStream memoryStream;
        private BinaryWriter writer;
        private CerasSerializer serializer;
        private byte[] buffer;

        public BufferedSerializer(Stream stream)
        {
            memoryStream = new MemoryStream();
            writer = new BinaryWriter(stream);
            serializer = new CerasSerializer();
        }

        public void Serialize<T>(T obj)
        {
            //memoryStream.Position = 0;
            int length = serializer.Serialize(obj, ref buffer);

            //formatter.Serialize(memoryStream, );
            //writer.Write(memoryStream.Position);
            writer.Write(length);
            writer.Write(buffer, 0, length);
            writer.Flush();
            //memoryStream.Position = 0;
            //memoryStream.CopyTo(writer.BaseStream);
            //memoryStream.Position = 0;
            //writer.BaseStream.Flush();
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
