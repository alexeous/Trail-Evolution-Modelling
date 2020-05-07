using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.GraphTypes
{
    [Serializable]
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

        private Graph() { }

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

        public Node GetClosestNode(float x, float y)
        {
            int snappedI = (int)Math.Round((x - OriginX) / Step);
            int snappedJ = (int)Math.Round((y - OriginY) / Step);
            snappedI = Math.Max(0, Math.Min(snappedI, Width - 1));
            snappedJ = Math.Max(0, Math.Min(snappedJ, Height - 1));
            if (nodes[snappedI, snappedJ] != null)
            {
                return nodes[snappedI, snappedJ];
            }

            float minDistance = float.PositiveInfinity;
            Node closest = null;
            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    Node node = GetNodeAtOrNull(i, j);
                    float distance = Distance(snappedI, snappedJ, i, j);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closest = node;
                    }
                }
            }

            return closest;
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
            if (nodes[i, j] != null)
                return null;

            var node = new Node(i, j);
            nodes[i, j] = node;
            return node;
        }

        public Edge AddEdge(Node node, Direction direction, float weight, bool isTramplable)
        {
            if (node.HasIncidentEdge(direction))
                return null;

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
            return Distance(a.I, a.J, b.I, b.J);
        }

        public float Distance(int i1, int j1, int i2, int j2)
        {
            int di = i1 - i2;
            int dj = j1 - j2;

            return Step * (float)Math.Sqrt(di * di + dj * dj);
        }
    }
}
