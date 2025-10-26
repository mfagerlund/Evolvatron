using System.Collections.Generic;

namespace Evolvatron.Core.Physics;

/// <summary>
/// Impulse-based contact solver for rigid bodies.
/// Based on Box2D's sequential impulse method.
/// This is a velocity-level solver that applies impulses to resolve contacts and friction.
/// Unlike XPBD's position corrections, this doesn't inject energy into rotation.
/// </summary>
public static class ImpulseContactSolver
{
    private const float Epsilon = 1e-9f;

    // Baumgarte stabilization parameter (0-1)
    // Higher values = faster penetration resolution but more jitter
    // Box2D uses 0.2 by default
    private const float Baumgarte = 0.2f;

    // Slop tolerance - don't correct small penetrations to avoid jitter
    // Box2D uses 0.005f (0.5cm)
    private const float LinearSlop = 0.01f;

    /// <summary>
    /// Initialize contact constraints from detected contacts.
    /// This detects all contacts, computes normals, and calculates effective masses.
    /// </summary>
    public static List<ContactConstraint> InitializeConstraints(WorldState world, float dt, float friction, float restitution)
    {
        var constraints = new List<ContactConstraint>();

        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            if (rb.InvMass == 0f) continue; // Skip static rigid bodies

            // Check each circle geom attached to this rigid body
            for (int g = 0; g < rb.GeomCount; g++)
            {
                var geom = world.RigidBodyGeoms[rb.GeomStartIndex + g];

                // Transform geom to world space
                CircleCollision.TransformGeomToWorld(rb, geom, out float geomWorldX, out float geomWorldY);

                // Check against all static circle colliders
                for (int j = 0; j < world.Circles.Count; j++)
                {
                    if (CircleCollision.CircleVsStaticCircle(geomWorldX, geomWorldY, geom.Radius, world.Circles[j], out var contactInfo))
                    {
                        var constraint = CreateConstraint(i, StaticColliderType.Circle, j, contactInfo, rb, friction, restitution, dt);
                        constraints.Add(constraint);
                    }
                }

                // Check against all static capsule colliders
                for (int j = 0; j < world.Capsules.Count; j++)
                {
                    if (CircleCollision.CircleVsStaticCapsule(geomWorldX, geomWorldY, geom.Radius, world.Capsules[j], out var contactInfo))
                    {
                        var constraint = CreateConstraint(i, StaticColliderType.Capsule, j, contactInfo, rb, friction, restitution, dt);
                        constraints.Add(constraint);
                    }
                }

                // Check against all static OBB colliders
                for (int j = 0; j < world.Obbs.Count; j++)
                {
                    if (CircleCollision.CircleVsStaticOBB(geomWorldX, geomWorldY, geom.Radius, world.Obbs[j], out var contactInfo))
                    {
                        var constraint = CreateConstraint(i, StaticColliderType.OBB, j, contactInfo, rb, friction, restitution, dt);
                        constraints.Add(constraint);
                    }
                }
            }
        }

        return constraints;
    }

    /// <summary>
    /// Creates a contact constraint from contact info.
    /// Computes effective mass and velocity bias for Baumgarte stabilization.
    /// </summary>
    private static ContactConstraint CreateConstraint(
        int rbIndex,
        StaticColliderType colliderType,
        int colliderIndex,
        ContactInfo contact,
        RigidBody rb,
        float friction,
        float restitution,
        float dt)
    {
        var constraint = new ContactConstraint
        {
            RigidBodyIndex = rbIndex,
            ColliderType = colliderType,
            ColliderIndex = colliderIndex,
            NormalX = contact.NormalX,
            NormalY = contact.NormalY,
            TangentX = -contact.NormalY,  // Perpendicular to normal
            TangentY = contact.NormalX,
            Friction = friction,
            Restitution = restitution,
            PointCount = 1
        };

        // Vector from rigid body center to contact point
        float rx = contact.ContactX - rb.X;
        float ry = contact.ContactY - rb.Y;

        // Compute effective mass for normal direction
        // Formula: normalMass = 1 / (invMass + invInertia * (r × n)²)
        // This accounts for both linear and rotational motion
        float rnCross = rx * contact.NormalY - ry * contact.NormalX;
        float normalMass = rb.InvMass + rb.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        // Compute effective mass for tangent direction
        // Formula: tangentMass = 1 / (invMass + invInertia * (r × t)²)
        float rtCross = rx * constraint.TangentY - ry * constraint.TangentX;
        float tangentMass = rb.InvMass + rb.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        // Velocity bias for Baumgarte stabilization
        // This gradually resolves penetration without position correction
        // When penetrating (separation < 0), we want positive bias to push apart
        // velocityBias = (Baumgarte / dt) * max(-penetration - slop, 0)
        float velocityBias = 0f;
        if (contact.Penetration < -LinearSlop)
        {
            // Penetration is negative, so negate it to get positive depth
            velocityBias = (Baumgarte / dt) * (-contact.Penetration - LinearSlop);
        }

        constraint.Point1 = new ContactConstraintPoint
        {
            WorldX = contact.ContactX,
            WorldY = contact.ContactY,
            RA_X = rx,
            RA_Y = ry,
            Separation = contact.Penetration,
            NormalMass = normalMass,
            TangentMass = tangentMass,
            NormalImpulse = 0f,  // No warm-starting yet
            TangentImpulse = 0f,
            VelocityBias = velocityBias
        };

        return constraint;
    }

    /// <summary>
    /// Warm-start the solver by applying cached impulses from previous frame.
    /// This dramatically improves convergence speed.
    /// </summary>
    public static void WarmStart(WorldState world, List<ContactConstraint> constraints)
    {
        for (int i = 0; i < constraints.Count; i++)
        {
            var constraint = constraints[i];
            var rb = world.RigidBodies[constraint.RigidBodyIndex];
            var point = constraint.Point1;

            // Apply cached normal impulse
            float px = constraint.NormalX * point.NormalImpulse;
            float py = constraint.NormalY * point.NormalImpulse;

            // Apply cached tangent impulse
            px += constraint.TangentX * point.TangentImpulse;
            py += constraint.TangentY * point.TangentImpulse;

            // Apply to linear velocity
            rb.VelX += rb.InvMass * px;
            rb.VelY += rb.InvMass * py;

            // Apply to angular velocity (torque = r × impulse)
            float torque = point.RA_X * py - point.RA_Y * px;
            rb.AngularVel += rb.InvInertia * torque;

            world.RigidBodies[constraint.RigidBodyIndex] = rb;
        }
    }

    /// <summary>
    /// Solve velocity constraints (normal and friction).
    /// This is called multiple times per substep (sequential impulse).
    /// </summary>
    public static void SolveVelocityConstraints(WorldState world, List<ContactConstraint> constraints)
    {
        for (int i = 0; i < constraints.Count; i++)
        {
            var constraint = constraints[i];
            var rb = world.RigidBodies[constraint.RigidBodyIndex];
            var point = constraint.Point1;

            // === FRICTION (solve tangent constraint first) ===

            // Relative velocity at contact point
            // v_contact = v_linear + omega × r
            float vx = rb.VelX - rb.AngularVel * point.RA_Y;
            float vy = rb.VelY + rb.AngularVel * point.RA_X;

            // Tangential velocity
            float vt = vx * constraint.TangentX + vy * constraint.TangentY;

            // Compute tangent impulse change
            float lambda = -point.TangentMass * vt;

            // Coulomb friction cone: clamp accumulated impulse
            float maxFriction = constraint.Friction * point.NormalImpulse;
            float newImpulse = MathF.Max(-maxFriction, MathF.Min(point.TangentImpulse + lambda, maxFriction));
            lambda = newImpulse - point.TangentImpulse;
            point.TangentImpulse = newImpulse;

            // Apply tangent impulse
            float px = constraint.TangentX * lambda;
            float py = constraint.TangentY * lambda;

            rb.VelX += rb.InvMass * px;
            rb.VelY += rb.InvMass * py;
            rb.AngularVel += rb.InvInertia * (point.RA_X * py - point.RA_Y * px);

            // === NORMAL (solve normal constraint) ===

            // Recompute relative velocity after friction
            vx = rb.VelX - rb.AngularVel * point.RA_Y;
            vy = rb.VelY + rb.AngularVel * point.RA_X;

            // Normal velocity
            float vn = vx * constraint.NormalX + vy * constraint.NormalY;

            // Compute normal impulse change (with velocity bias for penetration resolution)
            lambda = -point.NormalMass * (vn - point.VelocityBias);

            // Clamp accumulated impulse (normal impulse must be non-negative)
            newImpulse = MathF.Max(point.NormalImpulse + lambda, 0f);
            lambda = newImpulse - point.NormalImpulse;
            point.NormalImpulse = newImpulse;

            // Apply normal impulse
            px = constraint.NormalX * lambda;
            py = constraint.NormalY * lambda;

            rb.VelX += rb.InvMass * px;
            rb.VelY += rb.InvMass * py;
            rb.AngularVel += rb.InvInertia * (point.RA_X * py - point.RA_Y * px);

            // Write back
            constraint.Point1 = point;
            constraints[i] = constraint;
            world.RigidBodies[constraint.RigidBodyIndex] = rb;
        }
    }

    /// <summary>
    /// Store final impulses for warm-starting next frame.
    /// For now, we don't persist between frames, so this is a no-op.
    /// TODO: Store impulses in a persistent cache keyed by body+collider pair.
    /// </summary>
    public static void StoreImpulses(List<ContactConstraint> constraints)
    {
        // TODO: Implement impulse caching for persistent contacts
        // For now, we rebuild constraints from scratch each frame
    }
}
