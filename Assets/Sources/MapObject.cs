using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class MapObject : MonoBehaviour
{
    [SerializeField] bool isWalkable = true;
    [SerializeField] float weight = 1;
    [SerializeField] bool isTramplable = false;

    public AreaAttributes AreaAttributes => new AreaAttributes
    {
        IsWalkable = isWalkable,
        Weight = weight,
        IsTramplable = isTramplable
    };

    private void OnValidate()
    {
        if (weight < 1)
        {
            weight = 1;
        }
    }


    private static readonly AreaAttributes DefaultAreaAttributes = new AreaAttributes
    {
        IsWalkable = true,
        IsTramplable = true,
        Weight = 2.7f
    };

    private static readonly RaycastHit2D[] RaycastResults = new RaycastHit2D[30];
    private static readonly Collider2D[] OverlapResults = new Collider2D[30];

    public static AreaAttributes GetAreaAttributes(Vector2 point1, Vector2 point2)
    {
        int n = Physics2D.LinecastNonAlloc(point1, point2, RaycastResults);
        if (n == 0)
        {
            return DefaultAreaAttributes;
        }

        var result = new AreaAttributes();
        for (int i = 0; i < n; i++)
        {
            MapObject mapObject = RaycastResults[i].collider.GetComponent<MapObject>();
            if (mapObject == null)
            {
                continue;
            }

            AreaAttributes areaAttributes = mapObject.AreaAttributes;
            if (!areaAttributes.IsWalkable)
            {
                return new AreaAttributes { IsWalkable = false };
            }
            if (areaAttributes.Weight > result.Weight)
            {
                result = areaAttributes;
            }
        }
        
        return result;
    }

    public static bool IsPointWalkable(Vector2 point)
    {
        int n = Physics2D.OverlapPointNonAlloc(point, OverlapResults);
        for (int i = 0; i < n; i++)
        {
            MapObject mapObject = OverlapResults[i].GetComponent<MapObject>();
            if (mapObject != null && !mapObject.AreaAttributes.IsWalkable)
            {
                return false;
            }
        }
        return true;
    }
}
