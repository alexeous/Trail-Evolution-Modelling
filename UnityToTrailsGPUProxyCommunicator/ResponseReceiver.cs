using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    static class TaskExtensions
    {
        public static Task<T> WaitAsync<T>(this Task<T> @this, CancellationToken cancellationToken)
        {
            if (@this == null)
                throw new ArgumentNullException(nameof(@this));

            if (!cancellationToken.CanBeCanceled)
                return @this;
            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled<T>(cancellationToken);
            return DoWaitAsync<T>(@this, cancellationToken);
        }

        private static async Task<T> DoWaitAsync<T>(Task<T> task, CancellationToken cancellationToken)
        {
            var tcs = new TaskCompletionSource<T>();
            using (cancellationToken.Register(() => tcs.TrySetCanceled(cancellationToken)))
                return await await Task.WhenAny(task, tcs.Task).ConfigureAwait(false);
        }
    }

    class ResponseReceiver : IDisposable
    {
        private Stream stream;
        private BufferedDeserializer deserializer;
        private CancellationTokenSource cancellationSource;
        private Thread responseThread;
        private ConcurrentDictionary<int, TaskCompletionSource<Response>> responseWaiters;

        public ResponseReceiver(Stream stream)
        {
            this.stream = stream;
            deserializer = new BufferedDeserializer(stream);
            cancellationSource = new CancellationTokenSource();
            responseWaiters = new ConcurrentDictionary<int, TaskCompletionSource<Response>>();
            StartReceiveThread();
        }

        public void CancelAll()
        {
            cancellationSource.Cancel();
        }

        public async Task<T> ReceiveResultAsync<T>(Request request)
        {
            ResultResponse response = await ReceiveAsync(request).WaitAsync(cancellationSource.Token);
            return (T)response.Result;
        }

        public async Task<ResultResponse> ReceiveAsync(Request request)
        {
            var tcs = responseWaiters.GetOrAdd(request.ID, _ => new TaskCompletionSource<Response>());
            Response response = await tcs.Task;
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
            while (true)
            {
                var response = deserializer.Deserialize<Response>();
                int requestID = response.RequestID;
                var tcs = responseWaiters.GetOrAdd(requestID, _ => new TaskCompletionSource<Response>());
                tcs.SetResult(response);
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
                    cancellationSource?.Dispose();
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
