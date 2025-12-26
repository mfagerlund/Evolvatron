namespace Evolvatron.Core.Physics;

/// <summary>
/// Contact constraint point for impulse-based solving.
/// Based on Box2D's b2ContactConstraintPoint.
/// </summary>
public struct ContactConstraintPoint
{
    /// <summary>
    /// World position of the contact point.
    /// </summary>
    public float WorldX;
    public float WorldY;

    /// <summary>
    /// Vector from rigid body center of mass to contact point (in world space).
    /// Used for torque calculations: torque = r x impulse
    /// </summary>
    public float RA_X;
    public float RA_Y;

    /// <summary>
    /// Separation distance (negative = penetration).
    /// </summary>
    public float Separation;

    /// <summary>
    /// Inverse effective mass for normal direction.
    /// normalMass = 1 / (invMass + invInertia * (r x n)^2)
    /// Pre-computed for efficiency.
    /// </summary>
    public float NormalMass;

    /// <summary>
    /// Inverse effective mass for tangent direction.
    /// tangentMass = 1 / (invMass + invInertia * (r x t)^2)
    /// Pre-computed for efficiency.
    /// </summary>
    public float TangentMass;

    /// <summary>
    /// Accumulated normal impulse for warm-starting.
    /// Cached between frames for faster convergence.
    /// </summary>
    public float NormalImpulse;

    /// <summary>
    /// Accumulated tangent impulse for warm-starting.
    /// Cached between frames for faster convergence.
    /// </summary>
    public float TangentImpulse;

    /// <summary>
    /// Velocity bias for position correction (Baumgarte stabilization).
    /// Used to resolve penetration without jitter.
    /// </summary>
    public float VelocityBias;
}

/// <summary>
/// Contact constraint between a rigid body and a static collider.
/// Based on Box2D's b2ContactConstraint.
/// In 2D, we can have up to 2 contact points per pair (box-box edge contact).
/// For simplicity, we start with 1 contact point per constraint.
/// </summary>
public struct ContactConstraint
{
    /// <summary>
    /// Index of the rigid body in world.RigidBodies.
    /// Static collider has infinite mass (treated separately).
    /// </summary>
    public int RigidBodyIndex;

    /// <summary>
    /// Type of static collider this constraint is with.
    /// </summary>
    public StaticColliderType ColliderType;

    /// <summary>
    /// Index of the static collider in the appropriate world collection.
    /// </summary>
    public int ColliderIndex;

    /// <summary>
    /// Index of the geom within the rigid body that created this contact.
    /// Used for warm-starting cache key (body + collider type + collider index + geom index).
    /// </summary>
    public int GeomIndex;

    /// <summary>
    /// Contact normal pointing FROM static collider TO rigid body.
    /// This is the direction we apply normal impulses.
    /// </summary>
    public float NormalX;
    public float NormalY;

    /// <summary>
    /// Tangent direction (perpendicular to normal) for friction.
    /// tangent = (-normalY, normalX)
    /// </summary>
    public float TangentX;
    public float TangentY;

    /// <summary>
    /// Coefficient of friction (Coulomb).
    /// </summary>
    public float Friction;

    /// <summary>
    /// Coefficient of restitution (bounciness).
    /// 0 = perfectly inelastic, 1 = perfectly elastic.
    /// </summary>
    public float Restitution;

    /// <summary>
    /// Number of active contact points (1 or 2 in 2D).
    /// For now, we use 1 contact point per constraint.
    /// </summary>
    public int PointCount;

    /// <summary>
    /// Contact points (up to 2 in 2D).
    /// For circle/capsule collisions, we only have 1 point.
    /// For box-box or box-edge collisions, we can have 2 points.
    /// </summary>
    public ContactConstraintPoint Point1;
    public ContactConstraintPoint Point2;
}

/// <summary>
/// Type of static collider.
/// </summary>
public enum StaticColliderType
{
    Circle = 0,
    Capsule = 1,
    OBB = 2
}

/// <summary>
/// Unique identifier for a contact, used for warm-starting cache lookup.
/// Combines rigid body index, collider type, collider index, and geom index.
/// </summary>
public readonly struct ContactId : IEquatable<ContactId>
{
    public readonly int RigidBodyIndex;
    public readonly StaticColliderType ColliderType;
    public readonly int ColliderIndex;
    public readonly int GeomIndex;

    public ContactId(int rigidBodyIndex, StaticColliderType colliderType, int colliderIndex, int geomIndex)
    {
        RigidBodyIndex = rigidBodyIndex;
        ColliderType = colliderType;
        ColliderIndex = colliderIndex;
        GeomIndex = geomIndex;
    }

    public ContactId(ContactConstraint constraint)
    {
        RigidBodyIndex = constraint.RigidBodyIndex;
        ColliderType = constraint.ColliderType;
        ColliderIndex = constraint.ColliderIndex;
        GeomIndex = constraint.GeomIndex;
    }

    public bool Equals(ContactId other) =>
        RigidBodyIndex == other.RigidBodyIndex &&
        ColliderType == other.ColliderType &&
        ColliderIndex == other.ColliderIndex &&
        GeomIndex == other.GeomIndex;

    public override bool Equals(object? obj) => obj is ContactId other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(RigidBodyIndex, (int)ColliderType, ColliderIndex, GeomIndex);
}

/// <summary>
/// Cached impulse data for warm-starting contacts across frames.
/// </summary>
public struct CachedContactImpulse
{
    public float NormalImpulse;
    public float TangentImpulse;
}
