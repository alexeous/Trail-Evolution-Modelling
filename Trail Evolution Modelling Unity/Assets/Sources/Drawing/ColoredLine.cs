using UnityEngine;

namespace TrailEvolutionModelling.Drawing
{
    public struct ColoredLine
    {
        public readonly Vector3 start;
        public readonly Vector3 end;
        public readonly Color color;

        public ColoredLine(Vector3 start, Vector3 end, Color color)
        {
            this.start = start;
            this.end = end;
            this.color = color;
        }
    }
}