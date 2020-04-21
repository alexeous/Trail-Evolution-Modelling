using System;
using System.Runtime.InteropServices;

namespace TrailEvolutionModelling.GraphTypes
{
    public enum Direction
    {
        NW, N, NE,
        W, E,
        SW, S, SE
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
    }
}
