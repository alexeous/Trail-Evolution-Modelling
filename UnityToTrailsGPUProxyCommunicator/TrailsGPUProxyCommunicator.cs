using System;
using System.ComponentModel.Design.Serialization;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Net;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Linq;
using TrailEvolutionModelling.GraphTypes;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class TrailsGPUProxyCommunicator : IDisposable
    {
        public event EventHandler<ProcessErrorEventArgs> ProcessError;

        private Process process;
        private TcpListener tcpListener;
        private TcpClient tcpClient;
        private NetworkStream stream;
        private RequestSender requestSender;
        private ResponseReceiver responseReceiver;
        private CancellationTokenSource cancellationSource;
        private bool processClosedViaDispose;

        public TrailsGPUProxyCommunicator(string executablePath)
        {
            if (string.IsNullOrWhiteSpace(executablePath))
            {
                throw new ArgumentException("Path is null or blank");
            }

            try
            {
                //toExePipe = new AnonymousPipeServerStream(PipeDirection.Out, HandleInheritability.Inheritable);
                //fromExePipe = new AnonymousPipeServerStream(PipeDirection.In, HandleInheritability.Inheritable);

                //string toExeHandle = toExePipe.GetClientHandleAsString();
                //string fromExeHandle = fromExePipe.GetClientHandleAsString();
                //string processArgs = $"{toExeHandle} {fromExeHandle}";

                tcpListener = new TcpListener(IPAddress.Loopback, 0);
                int port = (tcpListener.LocalEndpoint as IPEndPoint).Port;

                process = new Process();
                process.StartInfo.FileName = executablePath;
                process.StartInfo.Arguments = port.ToString();
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.StandardErrorEncoding = Encoding.Unicode;
                process.EnableRaisingEvents = true;
                process.Exited += OnProcessExited;

                cancellationSource = new CancellationTokenSource();
            }
            catch (Exception ex)
            {
                Dispose(true);
                throw new Exception("Failed to start GPU proxy communicator", ex);
            }
        }

        public void Start()
        {
            tcpListener.Start();
            process.Start();

            Task<TcpClient> acceptTask = tcpListener.AcceptTcpClientAsync();
            try
            {
                acceptTask.Wait(1000, cancellationSource.Token);
            }
            catch (OperationCanceledException)
            {
                return;
            }

            if (!acceptTask.IsCompleted)
            {
                throw new Exception("Waiting too long for child process to connect");
            }
            tcpClient = acceptTask.Result;
            stream = tcpClient.GetStream();

            requestSender = new RequestSender(stream);
            responseReceiver = new ResponseReceiver(stream);
        }


        public async Task<TrailsComputationsOutput> ComputeAsync(TrailsComputationsInput input)
        {
            if (process == null)
                throw new InvalidOperationException("Process was already closed");

            var request = new TrailsComputationsRequest(input);
            await requestSender.SendAsync(request);
            return await responseReceiver.ReceiveResultAsync<TrailsComputationsOutput>(request);
        }

        #region IDisposable Support
        private bool disposedValue = false; // Для определения избыточных вызовов

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    processClosedViaDispose = true;
                    stream?.Dispose();
                    tcpClient?.Dispose();
                    requestSender?.Dispose();
                    process?.Close();
                }
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }
        #endregion

        private void OnProcessExited(object sender, EventArgs e)
        {
            if (processClosedViaDispose)
                return;

            int exitCode = process.ExitCode;
            string errorMsg = process.StandardError.ReadToEnd();
            process = null;

            ProcessError?.Invoke(this, new ProcessErrorEventArgs(exitCode, errorMsg));

            cancellationSource?.Cancel();
            responseReceiver.CancelAll();

            Dispose(true);
        }
    }
}
