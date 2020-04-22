using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;


namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class Program
    {
        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.Unicode;

            if (args.Length != 1)
            {
                Console.Error.WriteLine("There must be exactly 1 argument: port");
                return -1;
            }
            try
            {
                using (var client = new TrailsGPUProxyCommunicatorClient(args[0]))
                {
                    client.Run();
                }
            } catch (Exception ex)
            {
                Console.Error.WriteLine(ex.ToString());
                return -2;
            }

            return 0;
        }
    }
}
