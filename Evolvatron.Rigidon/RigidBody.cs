namespace Evolvatron.Core;

/// <summary>
/// Rigid body with position, rotation, and velocity.
/// Collides with static colliders but not with other rigid bodies.
/// Rigid bodies can have multiple circle geoms attached for complex shapes.
/// </summary>
public struct RigidBody
{
    public float X, Y;              // Center of mass position
    public float Angle;             // Rotation in radians
    public float VelX, VelY;        // Linear velocity
    public float AngularVel;        // Angular velocity (rad/s)
    public float InvMass;           // 1/mass (0 = static)
    public float InvInertia;        // 1/inertia (0 = static)

    // Multi-geom support
    public int GeomStartIndex;      // Index into WorldState.RigidBodyGeoms
    public int GeomCount;           // Number of circle geoms this body has

    public RigidBody(float x, float y, float angle, float mass, float inertia, int geomStartIndex, int geomCount)
    {
        X = x;
        Y = y;
        Angle = angle;
        VelX = 0f;
        VelY = 0f;
        AngularVel = 0f;
        InvMass = mass > 0f ? 1f / mass : 0f;
        InvInertia = inertia > 0f ? 1f / inertia : 0f;
        GeomStartIndex = geomStartIndex;
        GeomCount = geomCount;
    }
}

/// <summary>
/// Circle collision geometry attached to a rigid body.
/// Position is relative to the rigid body's center of mass.
/// </summary>
public struct RigidBodyGeom
{
    public float LocalX;    // X offset from rigid body center (in body's local space)
    public float LocalY;    // Y offset from rigid body center (in body's local space)
    public float Radius;    // Circle radius

    public RigidBodyGeom(float localX, float localY, float radius)
    {
        LocalX = localX;
        LocalY = localY;
        Radius = radius;
    }
}

/// <summary>
/// Contact information for rigid body collision.
/// </summary>
public struct ContactInfo
{
    public float ContactX, ContactY;  // World-space contact point
    public float NormalX, NormalY;    // Contact normal (from static to dynamic)
    public float Penetration;         // Signed distance (negative = overlap)

    public ContactInfo(float contactX, float contactY, float nx, float ny, float penetration)
    {
        ContactX = contactX;
        ContactY = contactY;
        NormalX = nx;
        NormalY = ny;
        Penetration = penetration;
    }
}

/// <summary>
/// Helper methods for rigid body calculations.
/// </summary>
public static class RigidBodyHelpers
{
    /// <summary>
    /// Rotates a vector by an angle (in radians).
    /// </summary>
    public static void Rotate(float x, float y, float angle, out float rx, out float ry)
    {
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);
        rx = x * cos - y * sin;
        ry = x * sin + y * cos;
    }

    /// <summary>
    /// Inverse rotation (rotate by -angle).
    /// </summary>
    public static void InverseRotate(float x, float y, float angle, out float rx, out float ry)
    {
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);
        rx = x * cos + y * sin;
        ry = -x * sin + y * cos;
    }
}
