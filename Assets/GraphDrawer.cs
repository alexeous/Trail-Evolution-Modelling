using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

[ExecuteInEditMode]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
public class GraphDrawer : MonoBehaviour
{
    [SerializeField] GraphHolder graphHolder = null;
    [SerializeField] float minWeight = 1;
    [SerializeField] Color minWeightColor = Color.blue;
    [SerializeField] Color maxWeightColor = Color.red;

    private Color oldMinColor;
    private Color oldMaxColor;
    private GraphHolder oldGraphHolder;

    private void Update()
    {
        if (graphHolder != oldGraphHolder ||
            minWeightColor != oldMinColor ||
            maxWeightColor != oldMaxColor)
        {
            oldGraphHolder = graphHolder;
            oldMinColor = minWeightColor;
            oldMaxColor = maxWeightColor;

            Redraw();
        }

        if (graphHolder != null)
        {
            graphHolder.GraphChanged -= Redraw;
            graphHolder.GraphChanged += Redraw;
        }

        if (graphHolder?.Graph == null)
        {
            if (GetComponent<MeshFilter>().sharedMesh != null)
                GetComponent<MeshFilter>().sharedMesh.Clear();
        }
    }

    private void Redraw()
    {
        Graph graph = graphHolder.Graph;
        
        float maxWeight = graph.Edges.Max(edge => edge.Weight);

        var vertices = new List<Vector3>();
        var colors = new List<Color>();
        foreach (var edge in graph.Edges)
        {
            Vector3 vertex1 = edge.Node1.Position;
            Vector3 vertex2 = edge.Node2.Position;
            Color color = GetEdgeColor(edge, maxWeight);
            
            vertices.Add(vertex1);
            colors.Add(color);
            vertices.Add(vertex2);
            colors.Add(color);
        }

        int[] indices = Enumerable.Range(0, vertices.Count).ToArray();

        var mesh = new Mesh();
        mesh.SetVertices(vertices);
        mesh.SetColors(colors);
        mesh.SetIndices(indices, MeshTopology.Lines, 0);

        GetComponent<MeshFilter>().sharedMesh = mesh;
    }

    private Color GetEdgeColor(Edge edge, float maxWeight)
    {
        float t = (edge.Weight - minWeight) / (maxWeight - minWeight);
        return Color.Lerp(minWeightColor, maxWeightColor, t);
    }
}
