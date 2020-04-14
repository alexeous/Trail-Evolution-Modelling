using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace TrailEvolutionModelling.Editor
{
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

            int y = 2;

            if (GUI.Button(new Rect(2, y, 60, 15), "Build 8"))
                BuildGraphs(true);

            if (GUI.Button(new Rect(2, y += 17, 60, 15), "Build 6"))
                BuildGraphs(false);

            if (GUI.Button(new Rect(2, y += 20, 40, 15), "A*"))
                FindPaths(PathFindingAlgorithm.AStar);

            if (GUI.Button(new Rect(2, y += 17, 40, 15), "NBA"))
                FindPaths(PathFindingAlgorithm.NBA);

            if (GUI.Button(new Rect(2, y += 17, 40, 15), "Wave"))
                FindPaths(PathFindingAlgorithm.Wavefront);

            if (GUI.Button(new Rect(2, y += 17, 40, 15), "||Wav"))
                FindPaths(PathFindingAlgorithm.WavefrontParallel);

            Handles.EndGUI();
        }

        private static void BuildGraphs(bool moore)
        {
            foreach (var builder in FindObjectsOfType<GraphBuilder>())
            {
                builder.Build(moore);
            }
        }

        private static void FindPaths(PathFindingAlgorithm algorithm)
        {
            foreach (var invoker in FindObjectsOfType<PathFinderInvoker>())
            {
                invoker.FindPath(algorithm);
            }
        }
    }
}