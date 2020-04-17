using System.Diagnostics;
using System.Runtime.CompilerServices;
using TrailEvolutionModelling;
using UnityEngine;

public class PlaySupport : MonoBehaviour
{
    [SerializeField] Transform start = null;
    [SerializeField] Transform end = null;

    string time = "";

    private void Awake()
    {
        BuildGraphs(true);
    }

    private void OnGUI()
    {
        GUI.color = new Color(1, 1, 1, 0.4f);

        int y = 2;

        if (GUI.Button(new Rect(2, y, 100, 15), "Build Graph"))
            BuildGraphs(true);

        if (GUI.Button(new Rect(2, y += 20, 100, 15), "A*"))
            FindPaths(PathFindingAlgorithm.AStar);

        if (GUI.Button(new Rect(2, y += 17, 100, 15), "NBA"))
            FindPaths(PathFindingAlgorithm.NBA);

        if (GUI.Button(new Rect(2, y += 17, 100, 15), "Wave ||"))
            FindPaths(PathFindingAlgorithm.WavefrontParallel);
        
        var style = GUI.skin.box;
        style.fontSize = 15;
        GUI.Box(new Rect(2, y += 27, 100, 30), time, style);

        GUI.Label(new Rect(Screen.width - 170, 2, 168, 50), 
            "Left mouse = set start\nRight mouse = set end\nSpace = invoke Wave ||");

    }

    private void Update()
    {
        if (Input.GetMouseButton(0) && !IsMouseOverGUI())
        {
            Vector3 position = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            position.z = -2;
            start.position = position;
        }
        else if (Input.GetMouseButton(1) && !IsMouseOverGUI())
        {
            Vector3 position = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            position.z = -2;
            end.position = position;
        }
        
        if (Input.GetKeyDown(KeyCode.Space))
        {
            FindPaths(PathFindingAlgorithm.WavefrontParallel);
        }
    }

    static bool IsMouseOverGUI()
    {
        return Input.mousePosition.x < 100 && (Screen.height - Input.mousePosition.y) < 80;
    }

    private static void BuildGraphs(bool moore)
    {
        foreach (var builder in FindObjectsOfType<GraphBuilder>())
        {
            builder.Build(moore);
        }
    }

    private void FindPaths(PathFindingAlgorithm algorithm)
    {
        var time = FindObjectOfType<PathFinderInvoker>().FindPath(algorithm);
        this.time = $"{time.TotalMilliseconds:0.00} ms";
    }
}
