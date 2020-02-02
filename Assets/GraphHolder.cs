using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class GraphHolder : MonoBehaviour
{
    [SerializeField] Color minWeightColor = Color.blue;
    [SerializeField] Color maxWeightColor = Color.red;

    public Graph Graph { get; set; }

    private void OnDrawGizmos()
    {
        if (Graph == null) 
            return;

        float maxWeight = Graph.Edges.Max(edge => edge.Weight);

        foreach (var edge in Graph.Edges)
        {
            Handles.color = GetEdgeColor(edge, maxWeight);
            
            //Handles.DrawSolidDisc(edge.Node1.Position, Vector3.forward, 0.1f);
            //Handles.DrawSolidDisc(edge.Node2.Position, Vector3.forward, 0.1f);
            Handles.DrawLine(edge.Node1.Position, edge.Node2.Position);
        }
    }

    private Color GetEdgeColor(Edge edge, float maxWeight)
    {
        return Color.Lerp(minWeightColor, maxWeightColor, edge.Weight / maxWeight);
    }
}
