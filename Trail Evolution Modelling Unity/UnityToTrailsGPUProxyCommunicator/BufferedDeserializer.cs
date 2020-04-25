using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Ceras;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class BufferedDeserializer : IDisposable
    {
        private MemoryStream memoryStream;
        private BinaryReader reader;
        private CerasSerializer serializer;
        private byte[] buffer;

        public BufferedDeserializer(Stream stream)
        {
            memoryStream = new MemoryStream();
            reader = new BinaryReader(stream);
            //formatter = new BinaryFormatter();
            buffer = new byte[32768];
        }

        public T Deserialize<T>()
        {
            int size = reader.ReadInt32();
            memoryStream.Position = 0;
            CopyStream(reader.BaseStream, memoryStream, size);
            memoryStream.Position = 0;
            var bytes = new byte[size];
            memoryStream.Read(bytes, 0, size);
            T result = serializer.Deserialize<T>(bytes);
            //var result = (T)formatter.Deserialize(memoryStream);
            memoryStream.Position = 0;
            return result;
        }

        private void CopyStream(Stream input, Stream output, int bytes)
        {
            int read;
            while (bytes > 0 && (read = input.Read(buffer, 0, Math.Min(buffer.Length, bytes))) > 0)
            {
                output.Write(buffer, 0, read);
                bytes -= read;
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
