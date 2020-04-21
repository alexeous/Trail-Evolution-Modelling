using System;
using System.Collections.Generic;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.GPUProxy;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class TrailsGPUProxyCommunicatorClient : IDisposable
    {
        private AnonymousPipeClientStream fromCommunicatorPipe;
        private AnonymousPipeClientStream toCommunicatorPipe;
        private RequestReceiver requestReceiver;
        private ResponseSender responseSender;

        public TrailsGPUProxyCommunicatorClient(string fromHandle, string toHandle)
        {
            try
            {
                fromCommunicatorPipe = new AnonymousPipeClientStream(PipeDirection.In, fromHandle);
                toCommunicatorPipe = new AnonymousPipeClientStream(PipeDirection.Out, toHandle);

                requestReceiver = new RequestReceiver(fromCommunicatorPipe);
                responseSender = new ResponseSender(toCommunicatorPipe);
            }
            catch (Exception ex)
            {
                Dispose(true);
                throw ex;
            }
        }

        public void Run()
        {
            while (true)
            {
                Request request = requestReceiver.Receive();
                ProcessRequestAsync(request).ContinueWith(task =>
                {
                    Response response = task.Result;
                    responseSender.Send(response);
                });
            }
        }

        private Task<Response> ProcessRequestAsync(Request request)
        {
            return Task.Run<Response>(() =>
            {
                try
                {
                    if (request is TrailsComputationsRequest trailsComputationsRequest)
                    {
                        TrailsComputationsInput input = trailsComputationsRequest.ComputationsInput;

                        TrailsComputationsOutput output = TrailsGPUProxy.ComputeTrails(input);
                        var response = new ResultResponse(request, output);
                        return response;
                    }
                    throw new ArgumentException("Unknown request type");
                }
                catch (Exception ex)
                {
                    return new ErrorResponse(request, ex);
                }
            });
        }

        #region IDisposable Support
        private bool disposedValue = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    toCommunicatorPipe?.Dispose();
                    fromCommunicatorPipe?.Dispose();
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
