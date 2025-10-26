namespace Evolvatron.Core.Physics;

/// <summary>
/// Simple and robust collision detection for circles against static colliders.
/// All rigid body geoms are circles, so we only need circle-vs-static functions.
/// </summary>
public static class CircleCollision
{
    private const float Epsilon = 1e-9f;

    /// <summary>
    /// Detect collision between a circle (at world position) and a static circle collider.
    /// </summary>
    public static bool CircleVsStaticCircle(
        float circleX, float circleY, float circleRadius,
        CircleCollider staticCircle,
        out ContactInfo contact)
    {
        float dx = circleX - staticCircle.CX;
        float dy = circleY - staticCircle.CY;
        float distSq = dx * dx + dy * dy;
        float totalRadius = circleRadius + staticCircle.Radius;
        float totalRadiusSq = totalRadius * totalRadius;

        if (distSq >= totalRadiusSq)
        {
            contact = default;
            return false;
        }

        float dist = MathF.Sqrt(distSq);
        float separation = dist - totalRadius;

        // Normal points from static circle to dynamic circle (pushes circle away)
        float nx, ny;
        if (dist > Epsilon)
        {
            nx = dx / dist;
            ny = dy / dist;
        }
        else
        {
            // Circles exactly overlapping - use arbitrary direction
            nx = 1f;
            ny = 0f;
        }

        // Contact point: midpoint between surface points
        float staticSurfaceX = staticCircle.CX + nx * staticCircle.Radius;
        float staticSurfaceY = staticCircle.CY + ny * staticCircle.Radius;
        float circleSurfaceX = circleX - nx * circleRadius;
        float circleSurfaceY = circleY - ny * circleRadius;

        float contactX = (staticSurfaceX + circleSurfaceX) * 0.5f;
        float contactY = (staticSurfaceY + circleSurfaceY) * 0.5f;

        contact = new ContactInfo(contactX, contactY, nx, ny, separation);
        return true;
    }

    /// <summary>
    /// Detect collision between a circle and a static capsule collider.
    /// </summary>
    public static bool CircleVsStaticCapsule(
        float circleX, float circleY, float circleRadius,
        CapsuleCollider capsule,
        out ContactInfo contact)
    {
        // Find closest point on capsule's line segment to circle center
        // Capsule endpoints: center ± unit_vector * half_length
        float ax = capsule.CX - capsule.UX * capsule.HalfLength;
        float ay = capsule.CY - capsule.UY * capsule.HalfLength;
        float bx = capsule.CX + capsule.UX * capsule.HalfLength;
        float by = capsule.CY + capsule.UY * capsule.HalfLength;

        float abx = bx - ax;
        float aby = by - ay;
        float apx = circleX - ax;
        float apy = circleY - ay;

        float abLenSq = abx * abx + aby * aby;
        float t = (apx * abx + apy * aby) / (abLenSq + Epsilon);
        t = Math.Clamp(t, 0f, 1f);

        float closestX = ax + t * abx;
        float closestY = ay + t * aby;

        // Distance from circle to closest point on capsule axis
        float dx = circleX - closestX;
        float dy = circleY - closestY;
        float dist = MathF.Sqrt(dx * dx + dy * dy);

        float totalRadius = circleRadius + capsule.Radius;
        float separation = dist - totalRadius;

        if (separation >= 0f)
        {
            contact = default;
            return false;
        }

        // Normal points from capsule to circle
        float nx, ny;
        if (dist > Epsilon)
        {
            nx = dx / dist;
            ny = dy / dist;
        }
        else
        {
            // Circle center on capsule axis - use perpendicular to axis
            float perpX = -aby;
            float perpY = abx;
            float perpLen = MathF.Sqrt(perpX * perpX + perpY * perpY);
            if (perpLen > Epsilon)
            {
                nx = perpX / perpLen;
                ny = perpY / perpLen;
            }
            else
            {
                nx = 1f;
                ny = 0f;
            }
        }

        // Contact point: midpoint between surface points
        float capsuleSurfaceX = closestX + nx * capsule.Radius;
        float capsuleSurfaceY = closestY + ny * capsule.Radius;
        float circleSurfaceX = circleX - nx * circleRadius;
        float circleSurfaceY = circleY - ny * circleRadius;

        float contactX = (capsuleSurfaceX + circleSurfaceX) * 0.5f;
        float contactY = (capsuleSurfaceY + circleSurfaceY) * 0.5f;

        contact = new ContactInfo(contactX, contactY, nx, ny, separation);
        return true;
    }

    /// <summary>
    /// Detect collision between a circle and a static OBB collider.
    /// </summary>
    public static bool CircleVsStaticOBB(
        float circleX, float circleY, float circleRadius,
        OBBCollider obb,
        out ContactInfo contact)
    {
        // Use SDF approach for OBB (it's well-tested for this case)
        Math2D.OBBSDF(circleX, circleY, obb, out float phi, out float nx, out float ny);

        float separation = phi - circleRadius;

        if (separation >= 0f)
        {
            contact = default;
            return false;
        }

        // Contact point: midpoint between surface points
        float obbSurfaceX = circleX - nx * phi;
        float obbSurfaceY = circleY - ny * phi;
        float circleSurfaceX = circleX - nx * circleRadius;
        float circleSurfaceY = circleY - ny * circleRadius;

        float contactX = (obbSurfaceX + circleSurfaceX) * 0.5f;
        float contactY = (obbSurfaceY + circleSurfaceY) * 0.5f;

        contact = new ContactInfo(contactX, contactY, nx, ny, separation);
        return true;
    }

    /// <summary>
    /// Transform a geom from rigid body's local space to world space.
    /// </summary>
    public static void TransformGeomToWorld(RigidBody rb, RigidBodyGeom geom,
        out float worldX, out float worldY)
    {
        // Rotate local offset by rigid body's angle
        float cos = MathF.Cos(rb.Angle);
        float sin = MathF.Sin(rb.Angle);

        float rotatedX = geom.LocalX * cos - geom.LocalY * sin;
        float rotatedY = geom.LocalX * sin + geom.LocalY * cos;

        // Translate by rigid body position
        worldX = rb.X + rotatedX;
        worldY = rb.Y + rotatedY;
    }
}
