using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TrailEvolutionModelling.Drawing
{
    public abstract class LinesProvider : MonoBehaviour
    {
        public abstract IEnumerable<ColoredLine> GetLines();
    }
}