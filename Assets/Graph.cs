using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class Node
{
    public Vector2 Position { get; set; }
    public List<Edge> IncidentEdges { get; }

    public Node(Vector2 position)
    {
        Position = position;
        IncidentEdges = new List<Edge>();
    }

    public void AddIncidentEdge(Edge edge) => IncidentEdges.Add(edge);

    public void RemoveIncidentEdge(Edge edge) => IncidentEdges.Remove(edge);
}

public class Edge : IEquatable<Edge>
{
    public Node Node1 { get; set; }
    public Node Node2 { get; set; }
    public float Weight { get; set; }
    public float Trampledness { get; set; }

    public Edge(Node node1, Node node2, float weight)
    {
        Node1 = node1;
        Node2 = node2;
        Weight = weight;
        Trampledness = 0;
    }

    public Node OtherNode(Node node)
    {
        if (node == Node1)
            return Node2;
        if (node == Node2)
            return Node1;

        throw new ArgumentException("Non-incident node");
    }

    public bool Equals(Edge other)
    {
        return this.Weight == other.Weight &&
            (this.Node1 == other.Node1 && this.Node2 == other.Node2 ||
            this.Node2 == other.Node1 && this.Node1 == other.Node2);
    }

    public override bool Equals(object obj) => obj is Edge other && this.Equals(other);

    public override int GetHashCode() => (Node1, Node2, Weight).GetHashCode();
}

public class Graph
{
    public List<Node> Nodes { get; } = new List<Node>();
    public List<Edge> Edges { get; } = new List<Edge>();

    public Node AddNode(Vector2 position)
    {
        var node = new Node(position);
        Nodes.Add(node);
        return node;
    }

    public void RemoveNode(Node node)
    {
        Nodes.Remove(node);
        Edges.RemoveAll(edge => node.IncidentEdges.Contains(edge));
    }

    public Edge AddEdge(Node node1, Node node2, float weight)
    {
        var edge = new Edge(node1, node2, weight);
        node1.AddIncidentEdge(edge);
        node2.AddIncidentEdge(edge);
        
        Edges.Add(edge);
        return edge;
    }

    public void RemoveEdge(Edge edge)
    {
        edge.Node1.RemoveIncidentEdge(edge);
        edge.Node2.RemoveIncidentEdge(edge);

        Edges.Remove(edge);
    }
}
