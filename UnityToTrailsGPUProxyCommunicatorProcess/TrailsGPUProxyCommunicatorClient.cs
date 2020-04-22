using System;
using System.Collections.Generic;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.GPUProxy;
using System.Net.Sockets;
using System.Net;
using System.IO;

namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class TrailsGPUProxyCommunicatorClient : IDisposable
    {
        private TcpClient tcpClient;
        private NetworkStream stream;
        private RequestReceiver requestReceiver;
        private ResponseSender responseSender;

        public TrailsGPUProxyCommunicatorClient(string portString)
        {
            try
            {
                int port = int.Parse(portString);
                tcpClient = new TcpClient();
                tcpClient.Connect(IPAddress.Loopback, port);
                stream = tcpClient.GetStream();

                requestReceiver = new RequestReceiver(stream);
                responseSender = new ResponseSender(stream);
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
                Request request = requestReceiver.ReceiveAsync().GetAwaiter().GetResult();
                Response response = ProcessRequest(request);
                responseSender.Send(response);
            }
        }

        private Response ProcessRequest(Request request)
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
        }

        #region IDisposable Support
        private bool disposedValue = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    stream?.Dispose();
                    tcpClient?.Dispose();
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
