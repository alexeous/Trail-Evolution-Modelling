using System;
using System.Buffers.Text;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Documents;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using Mapsui.Geometries;
using TrailEvolutionModelling.GPUProxy;
using TrailEvolutionModelling.GraphTypes;
using TrailEvolutionModelling.MapObjects;
using TrailEvolutionModelling.Util;

namespace TrailEvolutionModelling
{
    [StructLayout(LayoutKind.Sequential, Pack = 0)]
    public struct TramplednessEdge
    {
        public float X1, Y1;
        public float X2, Y2;
        public float Trampledness;

        public static readonly int Size = Marshal.SizeOf<TramplednessEdge>();

        public TramplednessEdge(float x1, float y1, float x2, float y2, float trampledness)
        {
            X1 = x1;
            Y1 = y1;
            X2 = x2;
            Y2 = y2;
            Trampledness = trampledness;
        }

        public TramplednessEdge(Point point1, Point point2, float trampledness)
        {
            X1 = (float)point1.X;
            Y1 = (float)point1.Y;
            X2 = (float)point2.X;
            Y2 = (float)point2.Y;
            Trampledness = trampledness;
        }

        public void WriteTo(BinaryWriter writer)
        {
            writer.Write(X1);
            writer.Write(Y1);
            writer.Write(X2);
            writer.Write(Y2);
            writer.Write(Trampledness);
        }

        public static TramplednessEdge ReadFrom(BinaryReader reader)
        {
            var result = new TramplednessEdge();
            result.X1 = reader.ReadSingle();
            result.Y1 = reader.ReadSingle();
            result.X2 = reader.ReadSingle();
            result.Y2 = reader.ReadSingle();
            result.Trampledness = reader.ReadSingle();
            return result;
        }
    }

    public class Trampledness : IEnumerable<TramplednessEdge>, IXmlSerializable
    {

        private List<TramplednessEdge> edges = new List<TramplednessEdge>();

        public Trampledness() 
        { 
        }

        public Trampledness(Graph graph)
        {
            float minW = TrailsGPUProxy.MinimumTramplableWeight;
            float maxW = AreaTypes.Default.Attributes.Weight;

            foreach (var edge in graph.Edges)
            {
                if (edge.Trampledness == 0)
                    continue;

                Point pos1 = graph.GetNodePosition(edge.Node1).ToMapsui();
                Point pos2 = graph.GetNodePosition(edge.Node2).ToMapsui();
                float newWeight = edge.Weight - edge.Trampledness;
                float t = (newWeight - minW) / (edge.Weight - minW);
                Add(new TramplednessEdge(pos1, pos2, t));
            }
        }

        public void Clear() => edges.Clear();

        public void Add(TramplednessEdge edge) => edges.Add(edge);

        public IEnumerator<TramplednessEdge> GetEnumerator() => edges.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => edges.GetEnumerator();


        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            reader.MoveToContent();
            int length = int.Parse(reader.GetAttribute("Length"));
            if (length % TramplednessEdge.Size != 0)
            {
                throw new ArgumentException($"Invalid trampledness edges legnth: {length}. " +
                    $"Must be divisible by {TramplednessEdge.Size}");
            }
            
            var bytes = new byte[length];
            reader.ReadElementContentAsBase64(bytes, 0, length);

            edges.Clear();
            using (var memStream = new MemoryStream(bytes))
            using (var binReader = new BinaryReader(memStream))
            {
                while (memStream.Position != memStream.Length)
                {
                    edges.Add(TramplednessEdge.ReadFrom(binReader));
                }
            }
        }

        public void WriteXml(XmlWriter writer)
        {
            using (var memStream = new MemoryStream())
            using (var binWriter = new BinaryWriter(memStream))
            {
                foreach (var edge in edges)
                {
                    edge.WriteTo(binWriter);
                }
                int length = (int)memStream.Position;
                writer.WriteAttributeString("Length", length.ToString());
                writer.WriteBase64(memStream.ToArray(), 0, length);
            }
        }
    }
}
