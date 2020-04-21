using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;
using TrailEvolutionModelling.GraphTypes;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class TrailsGPUProxyCommunicator : IDisposable
    {
        public event EventHandler<ProcessErrorEventArgs> ProcessError;

        private Process process;
        private AnonymousPipeServerStream toExePipe;
        private AnonymousPipeServerStream fromExePipe;
        private BinaryFormatter formatter;
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

                toExePipe.DisposeLocalCopyOfClientHandle();
                fromExePipe.DisposeLocalCopyOfClientHandle();

                formatter = new BinaryFormatter();
            }
            catch (Exception ex)
            {
                Dispose(true);
                throw ex;
            }
        }

        public Task<TrailsComputationsOutput> Compute(TrailsComputationsInput input)
        {
            if (process == null)
                throw new InvalidOperationException("Process was already closed");


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
                    process?.Close();
                    process?.Dispose();
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

            toExePipe?.Dispose();
            fromExePipe?.Dispose();

            int exitCode = process.ExitCode;
            string errorMsg = process.StandardError.ReadToEnd();
            process = null;

            ProcessError?.Invoke(this, new ProcessErrorEventArgs(exitCode, errorMsg));
        }
    }
}
