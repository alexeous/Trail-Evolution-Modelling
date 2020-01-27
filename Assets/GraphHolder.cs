using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class GraphHolder : MonoBehaviour
{
    [SerializeField] Color minWeightColor = Color.blue;
    [SerializeField] Color maxWeightColor = Color.red;

    public Graph graph { get; set; }

    private void OnDrawGizmos()
    {
        if (graph == null) 
            return;

        float maxWeight = graph.edges.Max(edge => edge.weight);

        foreach (var edge in graph.edges)
        {
            Handles.color = GetEdgeColor(edge, maxWeight);
            
            Handles.DrawSolidDisc(edge.node1.position, Vector3.forward, 0.1f);
            Handles.DrawSolidDisc(edge.node2.position, Vector3.forward, 0.1f);
            Handles.DrawLine(edge.node1.position, edge.node2.position);
        }
    }

    private Color GetEdgeColor(Edge edge, float maxWeight)
    {
        return Color.Lerp(minWeightColor, maxWeightColor, edge.weight / maxWeight);
    }
}
