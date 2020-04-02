using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class QuickAccessTools : EditorWindow
{
    [MenuItem("Window/QuickAccessTools")]
    static void Init()
    {
        GetWindow<QuickAccessTools>().Show();
    }

    private void OnEnable()
    {
        SceneView.duringSceneGui += OnScene;
    }

    private void OnDisable()
    {
        SceneView.duringSceneGui -= OnScene;
    }

    private static void OnScene(SceneView sceneview)
    {
        Handles.BeginGUI();

        GUI.color = new Color(1, 1, 1, 0.4f);

        if (GUI.Button(new Rect(2, 2, 40, 15), "Build"))
            BuildGraphs();

        if (GUI.Button(new Rect(2, 19, 40, 15), "Find"))
            FindPaths();

        Handles.EndGUI();
    }

    private static void BuildGraphs()
    {
        foreach (var builder in FindObjectsOfType<GraphBuilder>())
        {
            builder.Build();
        }
    }

    private static void FindPaths()
    {
        foreach (var invoker in FindObjectsOfType<PathFinderInvoker>())
        {
            invoker.FindPath();
        }
    }
}
