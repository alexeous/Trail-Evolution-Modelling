using System;
using System.ComponentModel.Design.Serialization;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Runtime.Serialization.Formatters.Binary;
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
        private AnonymousPipeServerStream toExePipe;
        private AnonymousPipeServerStream fromExePipe;
        private RequestSender requestSender;
        private ResponseReceiver responseReceiver;
        private bool processClosedViaDispose;

        public TrailsGPUProxyCommunicator(string executablePath)
        {
            if (string.IsNullOrWhiteSpace(executablePath))
            {
                throw new ArgumentException("Path is null or blank");
            }

            try
            {
                toExePipe = new AnonymousPipeServerStream(PipeDirection.Out, HandleInheritability.Inheritable);
                fromExePipe = new AnonymousPipeServerStream(PipeDirection.In, HandleInheritability.Inheritable);

                string toExeHandle = toExePipe.GetClientHandleAsString();
                string fromExeHandle = fromExePipe.GetClientHandleAsString();
                string processArgs = $"{toExeHandle} {fromExeHandle}";

                process = new Process();
                process.StartInfo.FileName = executablePath;
                process.StartInfo.Arguments = processArgs;
                process.StartInfo.UseShellExecute = false;
                process.Exited += OnProcessExited;
                process.Start();

                requestSender = new RequestSender(toExePipe);
                responseReceiver = new ResponseReceiver(fromExePipe);

                toExePipe.DisposeLocalCopyOfClientHandle();
                fromExePipe.DisposeLocalCopyOfClientHandle();
            }
            catch (Exception ex)
            {
                Dispose(true);
                throw new Exception("Failed to start GPU proxy communicator", ex);
            }
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
                    toExePipe?.Dispose();
                    fromExePipe?.Dispose();
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

            responseReceiver.CancelAll();

            int exitCode = process.ExitCode;
            string errorMsg = process.StandardError.ReadToEnd();
            process = null;

            ProcessError?.Invoke(this, new ProcessErrorEventArgs(exitCode, errorMsg));
            
            Dispose(true);
        }
    }
}
