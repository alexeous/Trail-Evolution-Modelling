using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GraphTypes
{
    [Serializable]
    public struct IncidentEdge : IEquatable<IncidentEdge>
    {
        public readonly Direction Direction;
        public readonly Edge Edge;
        public readonly Node OppositeNode;

        internal IncidentEdge(Direction direction, Edge edge, Node oppositeNode)
        {
            Direction = direction;
            Edge = edge;
            OppositeNode = oppositeNode;
        }

        public override bool Equals(object obj)
        {
            return obj is IncidentEdge edge && Equals(edge);
        }

        public override int GetHashCode()
        {
            int hashCode = 984489718;
            hashCode = hashCode * -1521134295 + Direction.GetHashCode();
            hashCode = hashCode * -1521134295 + Edge.GetHashCode();
            return hashCode;
        }

        public bool Equals(IncidentEdge edge)
        {
            return Direction == edge.Direction &&
                   Edge.Equals(edge.Edge);
        }
    }
}
