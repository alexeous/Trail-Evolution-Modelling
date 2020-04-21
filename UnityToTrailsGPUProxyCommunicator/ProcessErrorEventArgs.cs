using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    public class ProcessErrorEventArgs : EventArgs
    {
        public int ExitCode { get; }
        public string ErrorMessage { get; }

        internal ProcessErrorEventArgs(int exitCode, string message)
        {
            ExitCode = exitCode;
            ErrorMessage = message;
        }
    }
}
