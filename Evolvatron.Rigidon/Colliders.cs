namespace Evolvatron.Core;

/// <summary>
/// Static circle collider defined by center and radius.
/// </summary>
public struct CircleCollider
{
    /// <summary>Center X coordinate (meters).</summary>
    public float CX;

    /// <summary>Center Y coordinate (meters).</summary>
    public float CY;

    /// <summary>Radius (meters).</summary>
    public float Radius;

    public CircleCollider(float cx, float cy, float radius)
    {
        CX = cx;
        CY = cy;
        Radius = radius;
    }
}

/// <summary>
/// Static capsule (pill) collider: line segment with rounded ends.
/// Defined by center, unit axis direction, half-length, and radius.
/// </summary>
public struct CapsuleCollider
{
    /// <summary>Center X coordinate (meters).</summary>
    public float CX;

    /// <summary>Center Y coordinate (meters).</summary>
    public float CY;

    /// <summary>Unit axis X component (normalized direction vector).</summary>
    public float UX;

    /// <summary>Unit axis Y component (normalized direction vector).</summary>
    public float UY;

    /// <summary>Half-length of the capsule's core segment (meters).</summary>
    public float HalfLength;

    /// <summary>Radius of the rounded ends (meters).</summary>
    public float Radius;

    public CapsuleCollider(float cx, float cy, float ux, float uy, float halfLength, float radius)
    {
        CX = cx;
        CY = cy;
        UX = ux;
        UY = uy;
        HalfLength = halfLength;
        Radius = radius;
    }

    /// <summary>
    /// Helper: creates a capsule from two endpoints and radius.
    /// </summary>
    public static CapsuleCollider FromEndpoints(float x1, float y1, float x2, float y2, float radius)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len = MathF.Sqrt(dx * dx + dy * dy);
        if (len < 1e-6f)
        {
            // Degenerate case: treat as circle
            return new CapsuleCollider(x1, y1, 1f, 0f, 0f, radius);
        }

        float cx = (x1 + x2) * 0.5f;
        float cy = (y1 + y2) * 0.5f;
        float ux = dx / len;
        float uy = dy / len;
        float halfLen = len * 0.5f;

        return new CapsuleCollider(cx, cy, ux, uy, halfLen, radius);
    }
}

/// <summary>
/// Static oriented bounding box (OBB): rotatable rectangle.
/// Defined by center, unit axis (X direction), and half-extents.
/// </summary>
public struct OBBCollider
{
    /// <summary>Center X coordinate (meters).</summary>
    public float CX;

    /// <summary>Center Y coordinate (meters).</summary>
    public float CY;

    /// <summary>Unit axis X component (local X direction).</summary>
    public float UX;

    /// <summary>Unit axis Y component (local X direction).</summary>
    public float UY;

    /// <summary>Half-extent along local X axis (meters).</summary>
    public float HalfExtentX;

    /// <summary>Half-extent along local Y axis (meters).</summary>
    public float HalfExtentY;

    public OBBCollider(float cx, float cy, float ux, float uy, float hx, float hy)
    {
        CX = cx;
        CY = cy;
        UX = ux;
        UY = uy;
        HalfExtentX = hx;
        HalfExtentY = hy;
    }

    /// <summary>
    /// Helper: creates an axis-aligned OBB (unit axis = (1,0)).
    /// </summary>
    public static OBBCollider AxisAligned(float cx, float cy, float hx, float hy)
    {
        return new OBBCollider(cx, cy, 1f, 0f, hx, hy);
    }

    /// <summary>
    /// Helper: creates a rotated OBB from center, half-extents, and angle in radians.
    /// </summary>
    public static OBBCollider FromAngle(float cx, float cy, float hx, float hy, float angleRad)
    {
        float ux = MathF.Cos(angleRad);
        float uy = MathF.Sin(angleRad);
        return new OBBCollider(cx, cy, ux, uy, hx, hy);
    }
}
