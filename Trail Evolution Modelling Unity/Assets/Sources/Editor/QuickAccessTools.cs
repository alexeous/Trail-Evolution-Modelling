using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;
using System.Threading.Tasks;
using TrailEvolutionModelling.GPUProxyCommunicator;
//using TrailEvolutionModelling.GPUProxyCommunicator;
using TrailEvolutionModelling.GraphTypes;
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
                BuildGraphs();

            if (GUI.Button(new Rect(2, y += 17, 60, 15), "Compute"))
                InvokeComputations();
            //if (GUI.Button(new Rect(2, y += 20, 40, 15), "A*"))
            //    FindPaths(PathFindingAlgorithm.AStar);

            //if (GUI.Button(new Rect(2, y += 17, 40, 15), "NBA"))
            //    FindPaths(PathFindingAlgorithm.NBA);

            //if (GUI.Button(new Rect(2, y += 17, 40, 15), "Wave"))
            //    FindPaths(PathFindingAlgorithm.Wavefront);

            //if (GUI.Button(new Rect(2, y += 17, 40, 15), "||Wav"))
            //    FindPaths(PathFindingAlgorithm.WavefrontParallel);

            Handles.EndGUI();
        }

        private static void BuildGraphs()
        {
            foreach (var builder in FindObjectsOfType<GraphBuilder>())
            {
                builder.Build();
            }
        }

        private static void InvokeComputations()
        {
            //var input = new TrailsComputationsInput
            //{
            //    Attractors = new Attractor[1],
            //    Graph = FindObjectOfType<GraphHolder>().Graph
            //};
            //var config = new Ceras.SerializerConfig
            //{
            //    DefaultTargets = Ceras.TargetMember.All
            //};
            //Ceras.CerasSerializer s = new Ceras.CerasSerializer(config);
            //byte[] bytes = s.Serialize(input);
            //var i2 = s.Deserialize<TrailsComputationsInput>(bytes);
            //Debug.Log(bytes.Length);
            string basePath = Application.dataPath;
            string path = Path.Combine(basePath, @"ExternalTools\UnityToTrailsGPUProxyCommunicatorProcess\UnityToTrailsGPUProxyCommunicatorProcess.exe");
            using (var communicator = new TrailsGPUProxyCommunicator(path))
            {
                communicator.ProcessExited += (s, e) =>
                    Debug.LogError($"Process exited. Exit code: {e.ExitCode}, Message: {e.ErrorMessage}");

                communicator.Start();

                try
                {
                    var input = new TrailsComputationsInput
                    {
                        Attractors = new Attractor[0],
                        Graph = FindObjectOfType<GraphHolder>().Graph
                    };
                    var task = communicator.ComputeAsync(input);
                    if (!task.Wait(2000))
                    {
                        throw new Exception("Computation task was not completed in 2 sec");
                    }
                    TrailsComputationsOutput output = task.GetAwaiter().GetResult();
                    Debug.Log("Output: " + output.Graph.Width + output.Graph.Height);
                }
                catch
                {
                    Thread.Sleep(150); // wait for ProcessExited to raise
                    throw;
                }
            }
        }

        //private static void FindPaths(PathFindingAlgorithm algorithm)
        //{
        //    foreach (var invoker in FindObjectsOfType<PathFinderInvoker>())
        //    {
        //        invoker.FindPath(algorithm);
        //    }
        //}
    }
}