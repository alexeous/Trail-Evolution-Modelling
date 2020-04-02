using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class LinesProvider : MonoBehaviour
{
    public abstract IEnumerable<ColoredLine> GetLines();
}
