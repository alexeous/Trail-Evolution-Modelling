using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorkingBounds : MonoBehaviour
{
    private SpriteRenderer spriteRenderer => GetComponent<SpriteRenderer>();

    public Bounds Bounds => spriteRenderer.bounds;
}
