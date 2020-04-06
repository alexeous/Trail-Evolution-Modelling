using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TrailEvolutionModelling.Drawing
{
    public interface ILinesChangedNotifier
    {
        event Action<ILinesChangedNotifier> LinesChanged;
    }
}