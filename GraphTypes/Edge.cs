using System;
using System.Collections.Generic;

namespace TrailEvolutionModelling.GraphTypes
{
    [Serializable]
    public class Edge : IEquatable<Edge>
    {
        public Node Node1 { get; internal set; }
        public Node Node2 { get; internal set; }
        public float Weight { get; set; }
        public bool IsTramplable { get; set; }
        public float Trampledness { get; set; }


        private Edge() { }

        internal Edge(Node node1, Node node2, float weight, bool isTramplable)
        {
            Node1 = node1;
            Node2 = node2;
            Weight = weight;
            Trampledness = 0;
            IsTramplable = isTramplable;
        }

        public Node GetOppositeNode(Node node)
        {
            if (node == Node1)
                return Node2;
            if (node == Node2)
                return Node1;

            throw new ArgumentException("Non-incident node");
        }

        public bool Equals(Edge other)
        {
            return (this.Node1 == other.Node1 && this.Node2 == other.Node2 ||
                    this.Node2 == other.Node1 && this.Node1 == other.Node2);
        }

        public override bool Equals(object obj) => obj is Edge other && this.Equals(other);

        public override int GetHashCode()
        {
            int hashCode = 1001471489;
            hashCode = hashCode * -1521134295 + Node1.GetHashCode();
            hashCode = hashCode * -1521134295 + Node2.GetHashCode();
            return hashCode;
        }
    }
}