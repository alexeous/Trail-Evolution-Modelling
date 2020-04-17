using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Priority_Queue;
using UnityEngine;
using UnityEngine.AI;

namespace TrailEvolutionModelling
{
    public sealed class Node : FastPriorityQueueNode
    {
        public Vector2 Position { get; set; }
        public List<Edge> IncidentEdges { get; }

        public bool IsClosed;
        public float G1;
        public float F1;
        public float G2;
        public float F2;
        public int H1;
        public int H2;
        public Node CameFrom1;
        public Node CameFrom2;

        public int ComputeIndex;
        public int ComputeIndexI;
        public int ComputeIndexJ;

        public Node(Vector2 position)
        {
            Position = position;
            IncidentEdges = new List<Edge>();

            CleanupAfterPathSearch();
        }

        public void AddIncidentEdge(Edge edge)
        {
            if (!IncidentEdges.Contains(edge))
            {
                IncidentEdges.Add(edge);
            }
        }

        public bool RemoveIncidentEdge(Edge edge)
        {
            if (IncidentEdges.Remove(edge))
            {
                return true;
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref float G(bool forward) => ref (forward ? ref G1 : ref G2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref float F(bool forward) => ref (forward ? ref F1 : ref F2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref int H(bool forward) => ref (forward ? ref H1 : ref H2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref Node CameFrom(bool forward) => ref (forward ? ref CameFrom1 : ref CameFrom2);



        public void CleanupAfterPathSearch()
        {
            IsClosed = false;
            G1 = float.PositiveInfinity;
            G2 = float.PositiveInfinity;
            F1 = float.PositiveInfinity;
            F2 = float.PositiveInfinity;
            H1 = -1;
            H2 = -1;
            CameFrom1 = null;
            CameFrom2 = null;
        }
    }

    public sealed class Edge : IEquatable<Edge>
    {
        public Node Node1 { get; set; }
        public Node Node2 { get; set; }
        public float Weight { get; set; }
        public float Trampledness { get; set; }
        public bool IsTramplable { get; set; }

        public Edge(Node node1, Node node2, float weight, bool isTramplable)
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

        public override int GetHashCode() => Node1.GetHashCode() + Node2.GetHashCode();
    }

    //[StructLayout(LayoutKind.Sequential, Pack = 1, Size = StructSize)]
    public struct ComputeNode
    {
        public const int StructSize = sizeof(float) + sizeof(int); //sizeof(float) + sizeof(int) + sizeof(int);

        //private int Data;

        public float G;
        public int DirectionNext_IsStart;
        //public int IsStartByte;

        public bool IsStart
        {
            get => (byte)((DirectionNext_IsStart >> 3) & 1) != 0;
            set
            {
                if (value)
                    DirectionNext_IsStart |= (1 << 3); 
                else DirectionNext_IsStart &= ~(1 << 3);
            }
        }

        public int DirectionNext
        {
            get => DirectionNext_IsStart & 7;
            set => DirectionNext_IsStart = ((DirectionNext_IsStart >> 3) << 3) | (value & 7);
        }

        public override bool Equals(object obj)
        {
            return obj is ComputeNode node &&
                   G == node.G &&
                   DirectionNext_IsStart == node.DirectionNext_IsStart;
        }

        public override int GetHashCode()
        {
            int hashCode = 335743698;
            hashCode = hashCode * -1521134295 + G.GetHashCode();
            hashCode = hashCode * -1521134295 + DirectionNext_IsStart.GetHashCode();
            return hashCode;
        }

        //public bool IsStart
        //{
        //    get => IsStartByte != 0;
        //    set => IsStartByte = (value ? 1 : 0);
        //}

        public override string ToString()
        {
            return $"G={G}, DirNext={DirectionNext}, IsStart={IsStart}";
        }
    }

    public sealed class Graph
    {
        public Node[][] Nodes { get; set; }
        public HashSet<Edge> Edges { get; } = new HashSet<Edge>();
        
        public ComputeNode[] ComputeNodes { get; set; }
        public float[] ComputeEdgesVert { get; set; }
        public float[] ComputeEdgesHoriz { get; set; }
        public float[] ComputeEdgesLeftDiag { get; set; }
        public float[] ComputeEdgesRightDiag { get; set; }

        public ref float GetComputeEdgeForNode(int nodeI, int nodeJ, int di, int dj)
        {
            int w = Nodes.Length;

            if (di == -1 && dj == -1) return ref ComputeEdgesLeftDiag[nodeI + nodeJ * (w + 1)];
            if (di == 0 && dj == -1) return ref ComputeEdgesVert[nodeI + 1 + nodeJ * (w + 1)];
            if (di == 1 && dj == -1) return ref ComputeEdgesRightDiag[nodeI + 1 + nodeJ * (w + 1)];

            if (di == -1 && dj == 0) return ref ComputeEdgesHoriz[nodeI + (nodeJ + 1) * (w + 1)];
            if (di == 1 && dj == 0) return ref ComputeEdgesHoriz[nodeI + 1 + (nodeJ + 1) * (w + 1)];

            if (di == -1 && dj == 1) return ref ComputeEdgesRightDiag[nodeI + (nodeJ + 1) * (w + 1)];
            if (di == 0 && dj == 1) return ref ComputeEdgesVert[nodeI + 1 + (nodeJ + 1) * (w + 1)];
            if (di == 1 && dj == 1) return ref ComputeEdgesLeftDiag[nodeI + 1 + (nodeJ + 1) * (w + 1)];

            throw new Exception("Invalid rectangular moore shift");
        }

        public Edge AddEdge(Node node1, Node node2, float weight, bool isTramplable)
        {
            var edge = new Edge(node1, node2, weight, isTramplable);
            if (Edges.Add(edge))
            {
                node1.AddIncidentEdge(edge);
                node2.AddIncidentEdge(edge);
                return edge;
            }
            return null;
        }

        public void RemoveEdge(Edge edge)
        {
            Edges.Remove(edge);
            edge.Node1.RemoveIncidentEdge(edge);
            edge.Node2.RemoveIncidentEdge(edge);
        }

        public static (int di, int dj) DirectionToShift(int direction)
        {
            switch (direction)
            {
                case 0: return (-1, -1);
                case 1: return (0, -1);
                case 2: return (1, -1);
                case 3: return (-1, 0);
                case 4: return (1, 0);
                case 5: return (-1, 1);
                case 6: return (0, 1);
                case 7: return (1, 1);
                default: throw new ArgumentOutOfRangeException(nameof(direction));
            };
        }
    }
}