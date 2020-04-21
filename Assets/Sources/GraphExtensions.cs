using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TrailEvolutionModelling
{
    public static class GraphExtensions
    {
        public static Vector2Int ToVector2Int(this (int x, int y) coords)
        {
            return new Vector2Int(coords.x, coords.y);
        }

        public static Vector2 ToVector2(this (float x, float y) coords)
        {
            return new Vector2(coords.x, coords.y);
        }
    }
}