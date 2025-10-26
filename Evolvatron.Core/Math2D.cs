using System;

namespace Evolvatron.Core;

/// <summary>
/// 2D math utilities for vector operations and signed distance functions (SDFs).
/// All distances and lengths use meters.
/// </summary>
public static class Math2D
{
    private const float Epsilon = 1e-9f;

    #region Vector Operations

    /// <summary>
    /// Computes squared distance between two points.
    /// </summary>
    public static float DistanceSq(float x1, float y1, float x2, float y2)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return dx * dx + dy * dy;
    }

    /// <summary>
    /// Computes distance between two points.
    /// </summary>
    public static float Distance(float x1, float y1, float x2, float y2)
    {
        return MathF.Sqrt(DistanceSq(x1, y1, x2, y2));
    }

    /// <summary>
    /// Normalizes a 2D vector. Returns length.
    /// If length is near zero, returns (1, 0) and length 0.
    /// </summary>
    public static float Normalize(ref float x, ref float y)
    {
        float len = MathF.Sqrt(x * x + y * y);
        if (len < Epsilon)
        {
            x = 1f;
            y = 0f;
            return 0f;
        }
        float invLen = 1f / len;
        x *= invLen;
        y *= invLen;
        return len;
    }

    /// <summary>
    /// Computes 2D dot product.
    /// </summary>
    public static float Dot(float x1, float y1, float x2, float y2)
    {
        return x1 * x2 + y1 * y2;
    }

    /// <summary>
    /// Computes 2D cross product (scalar z-component).
    /// </summary>
    public static float Cross(float x1, float y1, float x2, float y2)
    {
        return x1 * y2 - y1 * x2;
    }

    /// <summary>
    /// Computes angle between two vectors in radians [-π, π].
    /// </summary>
    public static float AngleBetween(float x1, float y1, float x2, float y2)
    {
        return MathF.Atan2(Cross(x1, y1, x2, y2), Dot(x1, y1, x2, y2));
    }

    /// <summary>
    /// Wraps angle to [-π, π] range.
    /// </summary>
    public static float WrapAngle(float angle)
    {
        while (angle > MathF.PI) angle -= 2f * MathF.PI;
        while (angle < -MathF.PI) angle += 2f * MathF.PI;
        return angle;
    }

    /// <summary>
    /// Clamps a value between min and max.
    /// </summary>
    public static float Clamp(float value, float min, float max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    #endregion

    #region Circle SDF

    /// <summary>
    /// Signed distance and normal from point to circle.
    /// φ > 0: outside, φ < 0: inside, φ = 0: on boundary.
    /// Normal points outward from circle.
    /// </summary>
    public static void CircleSDF(
        float px, float py,
        in CircleCollider circle,
        out float phi, out float nx, out float ny)
    {
        float dx = px - circle.CX;
        float dy = py - circle.CY;
        float dist = MathF.Sqrt(dx * dx + dy * dy);

        if (dist < Epsilon)
        {
            // Point at center: arbitrary normal
            phi = -circle.Radius;
            nx = 1f;
            ny = 0f;
        }
        else
        {
            phi = dist - circle.Radius;
            nx = dx / dist;
            ny = dy / dist;
        }
    }

    #endregion

    #region Capsule SDF

    /// <summary>
    /// Signed distance and normal from point to capsule.
    /// Capsule is a line segment with rounded ends.
    /// </summary>
    public static void CapsuleSDF(
        float px, float py,
        in CapsuleCollider capsule,
        out float phi, out float nx, out float ny)
    {
        // Capsule endpoints
        float ax = capsule.CX - capsule.UX * capsule.HalfLength;
        float ay = capsule.CY - capsule.UY * capsule.HalfLength;
        float bx = capsule.CX + capsule.UX * capsule.HalfLength;
        float by = capsule.CY + capsule.UY * capsule.HalfLength;

        // Project point onto line segment
        float abx = bx - ax;
        float aby = by - ay;
        float apx = px - ax;
        float apy = py - ay;

        float t = Dot(apx, apy, abx, aby) / Dot(abx, aby, abx, aby);
        t = Clamp(t, 0f, 1f);

        // Closest point on segment
        float qx = ax + t * abx;
        float qy = ay + t * aby;

        // Distance to closest point
        float dx = px - qx;
        float dy = py - qy;
        float dist = MathF.Sqrt(dx * dx + dy * dy);

        if (dist < Epsilon)
        {
            // Point on capsule axis: normal perpendicular to axis
            phi = -capsule.Radius;
            nx = -capsule.UY;  // Perpendicular to axis
            ny = capsule.UX;
        }
        else
        {
            phi = dist - capsule.Radius;
            nx = dx / dist;
            ny = dy / dist;
        }
    }

    #endregion

    #region OBB SDF

    /// <summary>
    /// Signed distance and normal from point to oriented bounding box (OBB).
    /// </summary>
    public static void OBBSDF(
        float px, float py,
        in OBBCollider obb,
        out float phi, out float nx, out float ny)
    {
        // Transform point to OBB local space
        float dx = px - obb.CX;
        float dy = py - obb.CY;

        // Local coordinates: project onto OBB axes
        // UX, UY is the X axis; perpendicular is -UY, UX (rotated 90° CCW)
        float localX = Dot(dx, dy, obb.UX, obb.UY);
        float localY = Dot(dx, dy, -obb.UY, obb.UX);

        // Distance to box in local space
        float qx = Clamp(localX, -obb.HalfExtentX, obb.HalfExtentX);
        float qy = Clamp(localY, -obb.HalfExtentY, obb.HalfExtentY);

        float ex = localX - qx;
        float ey = localY - qy;
        float distSq = ex * ex + ey * ey;

        if (distSq < Epsilon)
        {
            // Point inside or on boundary: compute distance to nearest face
            float distX = obb.HalfExtentX - MathF.Abs(localX);
            float distY = obb.HalfExtentY - MathF.Abs(localY);

            if (distX < distY)
            {
                // Nearest face is perpendicular to X
                phi = -distX;
                float signX = localX >= 0f ? 1f : -1f;
                // Normal in local space: (±1, 0)
                // Transform back to world space
                nx = signX * obb.UX;
                ny = signX * obb.UY;
            }
            else
            {
                // Nearest face is perpendicular to Y
                phi = -distY;
                float signY = localY >= 0f ? 1f : -1f;
                // Normal in local space: (0, ±1)
                // Transform back to world space (Y axis = perpendicular to U)
                nx = -signY * obb.UY;
                ny = signY * obb.UX;
            }
        }
        else
        {
            // Point outside: distance to closest point on box
            phi = MathF.Sqrt(distSq);

            // Normal in local space
            float localNx = ex / phi;
            float localNy = ey / phi;

            // Transform normal back to world space
            nx = localNx * obb.UX - localNy * obb.UY;
            ny = localNx * obb.UY + localNy * obb.UX;
        }
    }

    #endregion
}
