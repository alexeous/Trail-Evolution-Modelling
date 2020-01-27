using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Node
{
    public Vector2 position { get; set; }
    public List<Edge> incidentEdges { get; }

    public Node(Vector2 position)
    {
        this.position = position;
        this.incidentEdges = new List<Edge>();
    }

    public void AddIncidentEdge(Edge edge) => incidentEdges.Add(edge);

    public void RemoveIncidentEdge(Edge edge) => incidentEdges.Remove(edge);
}

public class Edge : IEquatable<Edge>
{
    public Node node1 { get; set; }
    public Node node2 { get; set; }
    public float weight { get; set;  }

    public Edge(Node node1, Node node2, float weight)
    {
        this.node1 = node1;
        this.node2 = node2;
        this.weight = weight;
    }

    public Node OtherNode(Node node)
    {
        if (node == node1)
            return node2;
        if (node == node2)
            return node1;

        throw new ArgumentException("Non-incident node");
    }

    public bool Equals(Edge other)
    {
        return this.weight == other.weight &&
            (this.node1 == other.node1 && this.node2 == other.node2 ||
            this.node2 == other.node1 && this.node1 == other.node2);
    }

    public override bool Equals(object obj) => obj is Edge other && this.Equals(other);

    public override int GetHashCode() => (node1, node2, weight).GetHashCode();
}

public class Graph
{
    public List<Node> nodes { get; } = new List<Node>();
    public List<Edge> edges { get; } = new List<Edge>();

    public Node AddNode(Vector2 position)
    {
        var node = new Node(position);
        nodes.Add(node);
        return node;
    }

    public void RemoveNode(Node node)
    {
        nodes.Remove(node);
        edges.RemoveAll(edge => node.incidentEdges.Contains(edge));
    }

    public Edge AddEdge(Node node1, Node node2, float weight)
    {
        var edge = new Edge(node1, node2, weight);
        node1.AddIncidentEdge(edge);
        node2.AddIncidentEdge(edge);
        
        edges.Add(edge);
        return edge;
    }

    public void RemoveEdge(Edge edge)
    {
        edge.node1.RemoveIncidentEdge(edge);
        edge.node2.RemoveIncidentEdge(edge);

        edges.Remove(edge);
    }
}
