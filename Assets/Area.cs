using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Area : MonoBehaviour
{
    [SerializeField] bool isWalkable = true;
    [SerializeField] float weight = 1;
    [SerializeField] bool isTramplable = false;

    public bool IsWalkable => isWalkable;
    public float Weight => weight;
    public bool IsTramplable => isTramplable;
}
