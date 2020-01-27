using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class GraphBuilder : MonoBehaviour
{
    [SerializeField] WorkingArea workingArea = null;
    [SerializeField] GraphHolder target = null;
    [SerializeField] bool buildOnUpdate = false;

    [ContextMenu("Build")]
    private void Build()
    {
        if (workingArea == null || target == null)
            return;

        var graph = new Graph();
        var node1 = graph.AddNode(Vector2.zero);
        var node2 = graph.AddNode(new Vector2(0, 1));
        var node3 = graph.AddNode(new Vector2(2, 5));
        graph.AddEdge(node1, node2, 4);
        graph.AddEdge(node1, node3, 7);
        graph.AddEdge(node2, node3, 2);

        target.graph = graph;
    }

    private void Update()
    {
        if (buildOnUpdate)
        {
            Build();
        }
    }
}
