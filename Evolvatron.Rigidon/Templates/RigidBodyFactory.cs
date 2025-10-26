namespace Evolvatron.Core.Templates;

/// <summary>
/// Factory methods for creating rigid bodies composed of multiple circle geoms.
/// Handles mass and inertia calculations automatically.
/// </summary>
public static class RigidBodyFactory
{
    /// <summary>
    /// Creates a circular rigid body (single circle geom).
    /// </summary>
    public static int CreateCircle(WorldState world, float x, float y, float radius, float mass, float angle = 0f)
    {
        // Single circle at center
        float inertia = 0.5f * mass * radius * radius; // I = 0.5 * m * r^2

        int geomStartIndex = world.RigidBodyGeoms.Count;
        world.RigidBodyGeoms.Add(new RigidBodyGeom(0f, 0f, radius));

        var rb = new RigidBody(x, y, angle, mass, inertia, geomStartIndex, geomCount: 1);
        world.RigidBodies.Add(rb);

        return world.RigidBodies.Count - 1;
    }

    /// <summary>
    /// Creates a box rigid body approximated by multiple circles.
    /// Uses 5 circles: one at center, four at corners.
    /// </summary>
    public static int CreateBox(WorldState world, float x, float y, float halfExtentX, float halfExtentY,
        float mass, float angle = 0f)
    {
        // Approximate box inertia: I = (1/12) * m * (w^2 + h^2)
        float width = halfExtentX * 2f;
        float height = halfExtentY * 2f;
        float inertia = (mass / 12f) * (width * width + height * height);

        // Circle radius: must fit within the box edges
        // At corners, circles must not extend beyond box edges
        // For a circle at corner (±halfExtentX, ±halfExtentY), the max radius that fits is:
        // r ≤ min(halfExtentX, halfExtentY) to not stick out
        // We use slightly smaller to ensure coverage: 0.35 instead of 0.5
        float circleRadius = MathF.Min(halfExtentX, halfExtentY) * 0.35f;

        int geomStartIndex = world.RigidBodyGeoms.Count;

        // Center circle
        world.RigidBodyGeoms.Add(new RigidBodyGeom(0f, 0f, circleRadius));

        // Four corner circles positioned to stay within box bounds
        // Move corners inward by circleRadius to ensure they don't stick out
        float cornerOffsetX = halfExtentX - circleRadius;
        float cornerOffsetY = halfExtentY - circleRadius;
        world.RigidBodyGeoms.Add(new RigidBodyGeom(-cornerOffsetX, -cornerOffsetY, circleRadius));
        world.RigidBodyGeoms.Add(new RigidBodyGeom(cornerOffsetX, -cornerOffsetY, circleRadius));
        world.RigidBodyGeoms.Add(new RigidBodyGeom(cornerOffsetX, cornerOffsetY, circleRadius));
        world.RigidBodyGeoms.Add(new RigidBodyGeom(-cornerOffsetX, cornerOffsetY, circleRadius));

        var rb = new RigidBody(x, y, angle, mass, inertia, geomStartIndex, geomCount: 5);
        world.RigidBodies.Add(rb);

        return world.RigidBodies.Count - 1;
    }

    /// <summary>
    /// Creates a capsule rigid body approximated by multiple circles.
    /// Uses 3-7 circles along the capsule's length depending on size.
    /// </summary>
    public static int CreateCapsule(WorldState world, float x, float y, float halfLength, float radius,
        float mass, float angle = 0f)
    {
        // Capsule inertia (approximated as cylinder + 2 hemispheres)
        // I ≈ m * (r^2 / 4 + L^2 / 12)
        float length = halfLength * 2f;
        float inertia = mass * (radius * radius * 0.25f + length * length / 12f);

        int geomStartIndex = world.RigidBodyGeoms.Count;

        // Determine number of circles based on length
        int numCircles = Math.Clamp((int)(halfLength / radius) + 2, 3, 7);

        // Place circles along the capsule's local X axis
        for (int i = 0; i < numCircles; i++)
        {
            float t = (float)i / (numCircles - 1); // 0 to 1
            float localX = -halfLength + t * (halfLength * 2f);
            world.RigidBodyGeoms.Add(new RigidBodyGeom(localX, 0f, radius));
        }

        var rb = new RigidBody(x, y, angle, mass, inertia, geomStartIndex, numCircles);
        world.RigidBodies.Add(rb);

        return world.RigidBodies.Count - 1;
    }

    /// <summary>
    /// Applies an impulse to a rigid body at a world-space point.
    /// </summary>
    public static void ApplyImpulse(ref RigidBody rb, float impulseX, float impulseY,
        float contactX, float contactY)
    {
        // Linear impulse
        rb.VelX += impulseX * rb.InvMass;
        rb.VelY += impulseY * rb.InvMass;

        // Angular impulse: torque = r × impulse
        float rx = contactX - rb.X;
        float ry = contactY - rb.Y;
        float torque = rx * impulseY - ry * impulseX;
        rb.AngularVel += torque * rb.InvInertia;
    }

    /// <summary>
    /// Sets the velocity of a rigid body.
    /// </summary>
    public static void SetVelocity(ref RigidBody rb, float vx, float vy, float angularVel = 0f)
    {
        rb.VelX = vx;
        rb.VelY = vy;
        rb.AngularVel = angularVel;
    }

    /// <summary>
    /// Gets the velocity at a world-space point on the rigid body.
    /// </summary>
    public static void GetVelocityAtPoint(RigidBody rb, float worldX, float worldY,
        out float vx, out float vy)
    {
        float rx = worldX - rb.X;
        float ry = worldY - rb.Y;

        // v = v_linear + omega × r
        vx = rb.VelX - rb.AngularVel * ry;
        vy = rb.VelY + rb.AngularVel * rx;
    }
}
