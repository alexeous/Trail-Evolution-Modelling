using System;
using System.Runtime.InteropServices;

namespace TrailEvolutionModelling.GraphTypes
{
    public enum Direction
    {
        NW = 0, 
        N = 1, 
        NE = 2,
        W = 3, 
        E = 4,
        SW = 5, 
        S = 6, 
        SE = 7,
        First = NW,
        Last = SE
    }

    public static class DirectionUtil
    {
        private static readonly (int, int)[] dirsToShift =
        {
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        };

        private static readonly Direction[] oppositeDirs =
        {
            Direction.SE, Direction.S, Direction.SW,
            Direction.E, Direction.W,
            Direction.NE, Direction.W, Direction.NW
        };

        private static readonly float Sqrt2 = (float)Math.Sqrt(2);

        public static (int di, int dj) ToShift(this Direction dir)
        {
            return dirsToShift[(int)dir];
        }

        public static Direction Opposite(this Direction dir)
        {
            return oppositeDirs[(int)dir];
        }

        public static Direction ShiftToDirection(int di, int dj)
        {
            switch (di)
            {
                case -1:
                    switch (dj)
                    {
                        case -1: return Direction.NW;
                        case 0: return Direction.W;
                        case 1: return Direction.SW;
                    }
                    break;
                    
                case 0:
                    switch (dj)
                    {
                        case -1: return Direction.N;
                        case 1: return Direction.S;
                    }
                    break;

                case 1:
                    switch (dj)
                    {
                        case -1: return Direction.NE;
                        case 0: return Direction.E;
                        case 1: return Direction.SE;
                    }
                    break;
            }
            throw new ArgumentException("Invalid octile shift");
        }

        public static float WeightMultiplier(this Direction dir)
        {
            switch (dir)
            {
                case Direction.NW:
                case Direction.NE:
                case Direction.SW:
                case Direction.SE:
                    return Sqrt2;

                default:
                    return 1f;
            }
        }
    }
}
