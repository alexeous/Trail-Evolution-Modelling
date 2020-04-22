using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Ceras;
using Ceras.Helpers;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class ResponseReceiver : IDisposable
    {
        private Stream stream;
        private CerasSerializer deserializer;
        private Thread responseThread;
        private ConcurrentDictionary<int, TaskCompletionSource<Response>> responseWaiters;

        public ResponseReceiver(Stream stream)
        {
            this.stream = stream;
            deserializer = new CerasSerializer(new SerializerConfig { DefaultTargets = TargetMember.All });
            responseWaiters = new ConcurrentDictionary<int, TaskCompletionSource<Response>>();
            StartReceiveThread();
        }

        public void CancelAll(Exception exception = null)
        {
            exception = exception ?? new TaskCanceledException();
            foreach (var tcs in responseWaiters.Values)
            {
                if (tcs.Task.IsCompleted)
                    continue;
                tcs.SetException(exception);
            }
        }

        public async Task<T> ReceiveResultAsync<T>(Request request)
        {
            ResultResponse response = await ReceiveAsync(request);
            return (T)response.Result;
        }

        public async Task<ResultResponse> ReceiveAsync(Request request)
        {
            var tcs = responseWaiters.GetOrAdd(request.ID, _ => new TaskCompletionSource<Response>());
            Response response = await tcs.Task;
            responseWaiters.TryRemove(request.ID, out _);
            if (response is ErrorResponse error)
            {
                throw new ComputationsException("An error occurred while performing computations", 
                    error.Exception);
            }
            return (ResultResponse)response;
        }

        private void StartReceiveThread()
        {
            responseThread = new Thread(ResponseReadingProc);
            responseThread.IsBackground = true;
            responseThread.Start();
        }

        private void ResponseReadingProc()
        {
            try
            {
                while (true)
                {
                    //var response = deserializer.Deserialize<Response>();
                    var obj = deserializer.ReadFromStream(stream).GetAwaiter().GetResult();
                    var response = (Response)obj;
                    int requestID = response.RequestID;
                    var tcs = responseWaiters.GetOrAdd(requestID, _ => new TaskCompletionSource<Response>());
                    tcs.SetResult(response);
                }
            }
            catch (Exception ex)
            {
                CancelAll(ex);
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
                    responseThread?.Abort();
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
