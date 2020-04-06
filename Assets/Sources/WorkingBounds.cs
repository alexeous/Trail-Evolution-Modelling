using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorkingBounds : MonoBehaviour
{
    private SpriteRenderer spriteRenderer => GetComponent<SpriteRenderer>();

    public Vector2 Min => Bounds.min;
    public Vector2 Max => Bounds.max;
    public Vector2 Size => Bounds.size;
    public Vector2 Center => Bounds.center;

    private Bounds Bounds => spriteRenderer.bounds;
}
