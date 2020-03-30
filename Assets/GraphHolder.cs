using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class GraphHolder : MonoBehaviour
{
    private Graph graph;

    public Graph Graph
    {
        get => graph;
        set
        {
            graph = value;
            GraphChanged?.Invoke();
        }
    }

    public event Action GraphChanged;
}
