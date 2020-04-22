using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace TrailEvolutionModelling.GraphTypes
{
    [Serializable]
    public class Node
    {
        public int I { get; internal set; }
        public int J { get; internal set; }
        public IReadOnlyList<IncidentEdge> IncidentEdges => incidentEdges;
        public Node ComponentParent
        {
            get
            {
                if (componentParent != this)
                    componentParent = componentParent.ComponentParent;
                return componentParent;
            }
        }


        private List<IncidentEdge> incidentEdges;
        private int hasEdgesBits = 0;
        private Node componentParent;
        private int componentRank;


        private Node() { }

        internal Node(int i, int j)
        {
            I = i;
            J = j;

            incidentEdges = new List<IncidentEdge>();
            componentParent = this;
        }

        public bool HasIncidentEdge(Direction direction)
        {
            return ((hasEdgesBits >> (int)direction) & 1) != 0;
        }

        internal void AddIncidentEdge(Direction direction, Edge edge)
        {
            Node oppositeNode = edge.GetOppositeNode(this);
            incidentEdges.Add(new IncidentEdge(direction, edge, oppositeNode));
            hasEdgesBits |= 1 << (int)direction;
        }

        internal bool UnionComponents(Node other)
        {
            Node x = this.ComponentParent;
            Node y = other.ComponentParent;

            if (x == y)
                return false;

            if (x.componentRank < y.componentRank)
            {
                x.componentParent = y;
            }
            else
            {
                if (x.componentRank == y.componentRank)
                    x.componentRank++;
                y.componentParent = x;
            }

            return true;
        }
    }
}
