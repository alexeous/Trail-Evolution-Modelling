using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace TrailEvolutionModelling.GraphTypes
{
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
        private Node componentParent;
        private int componentRank;


        internal Node(int i, int j)
        {
            I = i;
            J = j;

            incidentEdges = new List<IncidentEdge>();
            componentParent = this;
        }

        internal void AddIncidentEdge(Direction direction, Edge edge)
        {
            Node oppositeNode = edge.GetOppositeNode(this);
            incidentEdges.Add(new IncidentEdge(direction, edge, oppositeNode));
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
