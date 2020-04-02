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

    private LinesProvider oldLinesProvider;
    private object firstUpdateDetector = null;

    private void Awake()
    {
        GetComponent<MeshRenderer>().material = new Material(Shader.Find("Sprites/Default"));
    }

    private void Update()
    {
        if (linesProvider != oldLinesProvider)
        {
            if (oldLinesProvider is ILinesChangedNotifier oldNotifier)
            {
                oldNotifier.LinesChanged -= OnLinesChanged;
            }

            oldLinesProvider = linesProvider;

            Redraw();
        }

        if (linesProvider is ILinesChangedNotifier notifier)
        {
            notifier.LinesChanged -= OnLinesChanged;
            notifier.LinesChanged += OnLinesChanged;
        }

        if (firstUpdateDetector == null)
        {
            firstUpdateDetector = new object();
            Redraw();
        }
    }

    private void OnLinesChanged(ILinesChangedNotifier notifier)
    {
        Redraw();
    }

    private void Redraw()
    {
        transform.position = new Vector3(0, 0, transform.position.z);

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
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
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
