using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ILinesChangedNotifier
{
    event Action<ILinesChangedNotifier> LinesChanged;
}
