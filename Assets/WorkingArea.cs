using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorkingArea : MonoBehaviour
{
    private SpriteRenderer spriteRenderer => GetComponent<SpriteRenderer>();

    public Bounds bounds => spriteRenderer.bounds;
}
