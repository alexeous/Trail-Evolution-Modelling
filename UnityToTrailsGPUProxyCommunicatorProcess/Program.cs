using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace TrailEvolutionModelling.GPUProxyCommunicator
{
    class Program
    {
        static int Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.Error.WriteLine("There must be exactly 2 args: handles of pipes");
                return -1;
            }
            try
            {
                using (var client = new TrailsGPUProxyCommunicatorClient(args[0], args[1]))
                {
                    client.Run();
                }
            } catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return -2;
            }

            return 0;
        }
    }
}
