using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GraphTypes
{
    public class Graph
    {
        public int Width { get; private set; }
        public int Height { get; private set; }
        public float Step { get; set; }
        public IReadOnlyList<Edge> Edges => edges;


        private Node[,] nodes;
        private List<Edge> edges;


        public Graph(int width, int height, float step)
        {
            Width = width;
            Height = height;
            Step = step;

            nodes = new Node[width, height];
            edges = new List<Edge>();
        }

        public Node GetNodeAtOrNull(int i, int j)
        {
            if (i < 0 || j < 0 || i >= Width || j >= Height)
                return null;

            return nodes[i, j];
        }

        public Node GetNodeNeighbourOrNull(Node node, Direction direction)
        {
            if (node == null)
                throw new ArgumentNullException(nameof(node));
            
            var shift = direction.ToShift();
            return GetNodeAtOrNull(node.I + shift.di, node.J + shift.dj);
        }

        public Node AddNode(int i, int j)
        {
            var node = new Node(i, j);
            nodes[i, j] = node;
            return node;
        }

        public Edge AddEdge(Node node, Direction direction, float weight, bool isTramplable)
        {
            Node neighbour = GetNodeNeighbourOrNull(node, direction);
            if (neighbour == null)
            {
                throw new ArgumentException("Neighbour node doesn't exist");
            }

            var edge = new Edge(node, neighbour, weight, isTramplable);
            node.SetIncidentEdge(direction, edge);
            neighbour.SetIncidentEdge(direction.Opposite(), edge);
            node.UnionComponents(neighbour);

            edges.Add(edge);
            return edge;
        }

        public float Distance(Node a, Node b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            int di = a.I - b.I;
            int dj = a.J - b.J;

            return Step * (float)Math.Sqrt(di * di + dj * dj);
        }
    }
}
