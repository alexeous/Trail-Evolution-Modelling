using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

[ExecuteInEditMode]
[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
public class LinesDrawer : MonoBehaviour
{
    [SerializeField] LinesProvider linesProvider = null;

    private LinesProvider oldLinesPrivder;

    private void Update()
    {
        if (linesProvider != oldLinesPrivder)
        {
            if (oldLinesPrivder is ILinesChangedNotifier oldNotifier)
            {
                oldNotifier.LinesChanged -= OnLinesChanged;
            }

            oldLinesPrivder = linesProvider;

            Redraw();
        }

        if (linesProvider is ILinesChangedNotifier notifier)
        {
            notifier.LinesChanged -= OnLinesChanged;
            notifier.LinesChanged += OnLinesChanged;
        }
    }

    private void OnLinesChanged(ILinesChangedNotifier notifier)
    {
        Redraw();
    }

    private void Redraw()
    {
        ClearMesh();

        if (linesProvider == null)
            return;

        var vertices = new List<Vector3>();
        var colors = new List<Color>();
        foreach (var line in linesProvider.GetLines())
        {   
            vertices.Add(line.start);
            colors.Add(line.color);
            vertices.Add(line.end);
            colors.Add(line.color);
        }

        int[] indices = Enumerable.Range(0, vertices.Count).ToArray();

        var mesh = new Mesh();
        mesh.SetVertices(vertices);
        mesh.SetColors(colors);
        mesh.SetIndices(indices, MeshTopology.Lines, 0);

        GetComponent<MeshFilter>().sharedMesh = mesh;
    }

    private void ClearMesh()
    {
        Mesh mesh = GetComponent<MeshFilter>().sharedMesh;
        if (mesh != null)
            mesh.Clear();
    }
}
