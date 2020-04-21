using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GraphTypes
{
    public class Graph
    {
        public static readonly int MaximumWidth = 500;
        public static readonly int MaximumHeight = 500;

        public int Width { get; private set; }
        public int Height { get; private set; }
        public float OriginX { get; set; }
        public float OriginY { get; set; }
        public float Step { get; set; }
        public IReadOnlyList<Edge> Edges => edges;


        private Node[,] nodes;
        private List<Edge> edges;


        public Graph(int width, int height, float originX, float originY, float step)
        {
            if (width < 0 || height < 0)
            {
                throw new ArgumentOutOfRangeException();
            }
            if (width > MaximumWidth)
            {
                throw new ArgumentOutOfRangeException(nameof(width), width,
                    $"Width is too large. Must not be greater than {MaximumWidth}");
            }
            if (height > MaximumHeight)
            {
                throw new ArgumentOutOfRangeException(nameof(height), height,
                    $"Height is too large. Must not be greater than {MaximumHeight}");
            }

            Width = width;
            Height = height;

            OriginX = originX;
            OriginY = originY;
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
            var shift = direction.ToShift();
            return GetNodeAtOrNull(node.I + shift.di, node.J + shift.dj);
        }

        public (float x, float y) GetNodePosition(Node node)
        {
            return GetNodePosition(node.I, node.J);
        }

        public (float x, float y) GetNodePosition(int i, int j)
        {
            float x = OriginX + Step * i;
            float y = OriginY + Step * j;
            return (x, y);
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

            var edge = new Edge(node, neighbour, weight, isTramplable);
            node.AddIncidentEdge(direction, edge);
            neighbour.AddIncidentEdge(direction.Opposite(), edge);
            node.UnionComponents(neighbour);

            edges.Add(edge);
            return edge;
        }

        public float Distance(Node a, Node b)
        {
            int di = a.I - b.I;
            int dj = a.J - b.J;

            return Step * (float)Math.Sqrt(di * di + dj * dj);
        }
    }
}
