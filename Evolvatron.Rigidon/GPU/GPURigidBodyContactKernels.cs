using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU;

// GPU-compatible rigid body structs are defined in GPURigidBodyStructs.cs

/// <summary>
/// GPU kernels for rigid body contact detection and solving.
/// Ports the CPU ImpulseContactSolver to ILGPU.
/// Based on Box2D's sequential impulse method.
/// </summary>
public static class GPURigidBodyContactKernels
{
    private const float Epsilon = 1e-9f;
    private const float Baumgarte = 0.2f;
    private const float LinearSlop = 0.01f;

    #region Rigid Body Integration Kernels

    /// <summary>
    /// Applies gravity to all rigid bodies.
    /// One thread per rigid body.
    /// </summary>
    public static void ApplyRigidBodyGravityKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        float gravityX, float gravityY, float dt)
    {
        var body = bodies[index];
        if (body.InvMass > 0f)
        {
            body.VelX += gravityX * dt;
            body.VelY += gravityY * dt;
            bodies[index] = body;
        }
    }

    /// <summary>
    /// Integrates rigid body velocities to update positions.
    /// One thread per rigid body.
    /// </summary>
    public static void IntegrateRigidBodiesKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        float dt)
    {
        var body = bodies[index];
        if (body.InvMass > 0f)
        {
            body.X += body.VelX * dt;
            body.Y += body.VelY * dt;
            body.Angle += body.AngularVel * dt;
            bodies[index] = body;
        }
    }

    /// <summary>
    /// Applies global damping to rigid body velocities.
    /// One thread per rigid body.
    /// </summary>
    public static void DampRigidBodiesKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        float dampingFactor)
    {
        var body = bodies[index];
        if (body.InvMass > 0f)
        {
            body.VelX *= dampingFactor;
            body.VelY *= dampingFactor;
            body.AngularVel *= dampingFactor;
            bodies[index] = body;
        }
    }

    #endregion

    #region Contact Detection Kernels

    /// <summary>
    /// Detects contacts between rigid body geoms and static circle colliders.
    /// One thread per (body, geom, collider) combination.
    /// </summary>
    public static void DetectCircleContactsKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPUCircleCollider> circles,
        ArrayView<GPUContactConstraint> contacts,
        int contactOffset,
        int bodyCount, int maxGeomsPerBody, int circleCount,
        float friction, float restitution, float dt)
    {
        // Decode indices: index = body * maxGeomsPerBody * circleCount + geom * circleCount + circle
        int totalPerBody = maxGeomsPerBody * circleCount;
        int bodyIdx = index / totalPerBody;
        int remainder = index % totalPerBody;
        int geomLocalIdx = remainder / circleCount;
        int circleIdx = remainder % circleCount;

        int contactIdx = contactOffset + index;
        var contact = new GPUContactConstraint { IsValid = 0 };

        if (bodyIdx >= bodyCount || circleIdx >= circleCount)
        {
            contacts[contactIdx] = contact;
            return;
        }

        var body = bodies[bodyIdx];
        if (body.InvMass == 0f || geomLocalIdx >= body.GeomCount)
        {
            contacts[contactIdx] = contact;
            return;
        }

        var geom = geoms[body.GeomStartIndex + geomLocalIdx];
        var circle = circles[circleIdx];

        // Transform geom to world space
        float cosA = XMath.Cos(body.Angle);
        float sinA = XMath.Sin(body.Angle);
        float geomWorldX = body.X + geom.LocalX * cosA - geom.LocalY * sinA;
        float geomWorldY = body.Y + geom.LocalX * sinA + geom.LocalY * cosA;

        // Circle-circle distance
        float dx = geomWorldX - circle.CX;
        float dy = geomWorldY - circle.CY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);
        float combinedRadius = geom.Radius + circle.Radius;

        if (dist >= combinedRadius)
        {
            contacts[contactIdx] = contact;
            return;
        }

        // Collision detected
        float penetration = dist - combinedRadius;
        float nx, ny;
        if (dist > Epsilon)
        {
            nx = dx / dist;
            ny = dy / dist;
        }
        else
        {
            nx = 1f;
            ny = 0f;
        }

        // Contact point (on geom surface toward collider)
        float contactX = geomWorldX - nx * geom.Radius;
        float contactY = geomWorldY - ny * geom.Radius;

        // Vector from body center to contact point
        float rx = contactX - body.X;
        float ry = contactY - body.Y;

        // Compute effective masses
        float rnCross = rx * ny - ry * nx;
        float normalMass = body.InvMass + body.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        float tx = -ny;
        float ty = nx;
        float rtCross = rx * ty - ry * tx;
        float tangentMass = body.InvMass + body.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        // Baumgarte velocity bias
        float velocityBias = 0f;
        if (penetration < -LinearSlop)
        {
            velocityBias = (Baumgarte / dt) * (-penetration - LinearSlop);
        }

        contact.RigidBodyIndex = bodyIdx;
        contact.NormalX = nx;
        contact.NormalY = ny;
        contact.TangentX = tx;
        contact.TangentY = ty;
        contact.ContactX = contactX;
        contact.ContactY = contactY;
        contact.RA_X = rx;
        contact.RA_Y = ry;
        contact.Separation = penetration;
        contact.NormalMass = normalMass;
        contact.TangentMass = tangentMass;
        contact.VelocityBias = velocityBias;
        contact.NormalImpulse = 0f;
        contact.TangentImpulse = 0f;
        contact.Friction = friction;
        contact.Restitution = restitution;
        contact.GeomIndex = geomLocalIdx;
        contact.ColliderType = 0; // Circle
        contact.ColliderIndex = circleIdx;
        contact.IsValid = 1;

        contacts[contactIdx] = contact;
    }

    /// <summary>
    /// Detects contacts between rigid body geoms and static OBB colliders.
    /// One thread per (body, geom, obb) combination.
    /// </summary>
    public static void DetectOBBContactsKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPUOBBCollider> obbs,
        ArrayView<GPUContactConstraint> contacts,
        int contactOffset,
        int bodyCount, int maxGeomsPerBody, int obbCount,
        float friction, float restitution, float dt)
    {
        // Decode indices
        int totalPerBody = maxGeomsPerBody * obbCount;
        int bodyIdx = index / totalPerBody;
        int remainder = index % totalPerBody;
        int geomLocalIdx = remainder / obbCount;
        int obbIdx = remainder % obbCount;

        int contactIdx = contactOffset + index;
        var contact = new GPUContactConstraint { IsValid = 0 };

        if (bodyIdx >= bodyCount || obbIdx >= obbCount)
        {
            contacts[contactIdx] = contact;
            return;
        }

        var body = bodies[bodyIdx];
        if (body.InvMass == 0f || geomLocalIdx >= body.GeomCount)
        {
            contacts[contactIdx] = contact;
            return;
        }

        var geom = geoms[body.GeomStartIndex + geomLocalIdx];
        var obb = obbs[obbIdx];

        // Transform geom to world space
        float cosA = XMath.Cos(body.Angle);
        float sinA = XMath.Sin(body.Angle);
        float geomWorldX = body.X + geom.LocalX * cosA - geom.LocalY * sinA;
        float geomWorldY = body.Y + geom.LocalX * sinA + geom.LocalY * cosA;

        // Circle vs OBB SDF
        float phi, nx, ny;
        CircleVsOBBSDF(geomWorldX, geomWorldY, geom.Radius, obb, out phi, out nx, out ny);

        if (phi >= 0f)
        {
            contacts[contactIdx] = contact;
            return;
        }

        // Contact point
        float contactX = geomWorldX - nx * geom.Radius;
        float contactY = geomWorldY - ny * geom.Radius;

        // Vector from body center to contact point
        float rx = contactX - body.X;
        float ry = contactY - body.Y;

        // Compute effective masses
        float rnCross = rx * ny - ry * nx;
        float normalMass = body.InvMass + body.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        float tx = -ny;
        float ty = nx;
        float rtCross = rx * ty - ry * tx;
        float tangentMass = body.InvMass + body.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        float velocityBias = 0f;
        if (phi < -LinearSlop)
        {
            velocityBias = (Baumgarte / dt) * (-phi - LinearSlop);
        }

        contact.RigidBodyIndex = bodyIdx;
        contact.NormalX = nx;
        contact.NormalY = ny;
        contact.TangentX = tx;
        contact.TangentY = ty;
        contact.ContactX = contactX;
        contact.ContactY = contactY;
        contact.RA_X = rx;
        contact.RA_Y = ry;
        contact.Separation = phi;
        contact.NormalMass = normalMass;
        contact.TangentMass = tangentMass;
        contact.VelocityBias = velocityBias;
        contact.NormalImpulse = 0f;
        contact.TangentImpulse = 0f;
        contact.Friction = friction;
        contact.Restitution = restitution;
        contact.GeomIndex = geomLocalIdx;
        contact.ColliderType = 2; // OBB
        contact.ColliderIndex = obbIdx;
        contact.IsValid = 1;

        contacts[contactIdx] = contact;
    }

    #endregion

    #region Contact Solving Kernels

    /// <summary>
    /// Solves velocity constraints for all valid contacts.
    /// Uses sequential impulse method with friction.
    /// One thread per contact.
    /// </summary>
    public static void SolveContactVelocitiesKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPUContactConstraint> contacts)
    {
        var contact = contacts[index];
        if (contact.IsValid == 0)
            return;

        var body = bodies[contact.RigidBodyIndex];

        // === FRICTION (solve tangent constraint first) ===

        // Relative velocity at contact point
        float vx = body.VelX - body.AngularVel * contact.RA_Y;
        float vy = body.VelY + body.AngularVel * contact.RA_X;

        // Tangential velocity
        float vt = vx * contact.TangentX + vy * contact.TangentY;

        // Compute tangent impulse
        float lambda = -contact.TangentMass * vt;

        // Coulomb friction cone clamp
        float maxFriction = contact.Friction * contact.NormalImpulse;
        float oldTangentImpulse = contact.TangentImpulse;
        float newTangentImpulse = oldTangentImpulse + lambda;
        newTangentImpulse = XMath.Max(-maxFriction, XMath.Min(newTangentImpulse, maxFriction));
        lambda = newTangentImpulse - oldTangentImpulse;
        contact.TangentImpulse = newTangentImpulse;

        // Apply tangent impulse
        float px = contact.TangentX * lambda;
        float py = contact.TangentY * lambda;

        Atomic.Add(ref bodies[contact.RigidBodyIndex].VelX, body.InvMass * px);
        Atomic.Add(ref bodies[contact.RigidBodyIndex].VelY, body.InvMass * py);
        Atomic.Add(ref bodies[contact.RigidBodyIndex].AngularVel, body.InvInertia * (contact.RA_X * py - contact.RA_Y * px));

        // Update local velocity for normal solve
        body.VelX += body.InvMass * px;
        body.VelY += body.InvMass * py;
        body.AngularVel += body.InvInertia * (contact.RA_X * py - contact.RA_Y * px);

        // === NORMAL (solve normal constraint) ===

        // Recompute relative velocity after friction
        vx = body.VelX - body.AngularVel * contact.RA_Y;
        vy = body.VelY + body.AngularVel * contact.RA_X;

        // Normal velocity
        float vn = vx * contact.NormalX + vy * contact.NormalY;

        // Compute normal impulse with velocity bias
        lambda = -contact.NormalMass * (vn - contact.VelocityBias);

        // Clamp (normal impulse must be non-negative)
        float oldNormalImpulse = contact.NormalImpulse;
        float newNormalImpulse = oldNormalImpulse + lambda;
        newNormalImpulse = XMath.Max(newNormalImpulse, 0f);
        lambda = newNormalImpulse - oldNormalImpulse;
        contact.NormalImpulse = newNormalImpulse;

        // Apply normal impulse
        px = contact.NormalX * lambda;
        py = contact.NormalY * lambda;

        Atomic.Add(ref bodies[contact.RigidBodyIndex].VelX, body.InvMass * px);
        Atomic.Add(ref bodies[contact.RigidBodyIndex].VelY, body.InvMass * py);
        Atomic.Add(ref bodies[contact.RigidBodyIndex].AngularVel, body.InvInertia * (contact.RA_X * py - contact.RA_Y * px));

        // Write back updated contact
        contacts[index] = contact;
    }

    /// <summary>
    /// Detects contacts between rigid body geoms and static capsule colliders.
    /// One thread per (body, geom, capsule) combination.
    /// </summary>
    public static void DetectCapsuleContactsKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPUCapsuleCollider> capsules,
        ArrayView<GPUContactConstraint> contacts,
        int contactOffset,
        int bodyCount, int maxGeomsPerBody, int capsuleCount,
        float friction, float restitution, float dt)
    {
        // Decode indices
        int totalPerBody = maxGeomsPerBody * capsuleCount;
        int bodyIdx = index / totalPerBody;
        int remainder = index % totalPerBody;
        int geomLocalIdx = remainder / capsuleCount;
        int capsuleIdx = remainder % capsuleCount;

        int contactIdx = contactOffset + index;
        var contact = new GPUContactConstraint { IsValid = 0 };

        if (bodyIdx >= bodyCount || capsuleIdx >= capsuleCount)
        {
            contacts[contactIdx] = contact;
            return;
        }

        var body = bodies[bodyIdx];
        if (body.InvMass == 0f || geomLocalIdx >= body.GeomCount)
        {
            contacts[contactIdx] = contact;
            return;
        }

        var geom = geoms[body.GeomStartIndex + geomLocalIdx];
        var capsule = capsules[capsuleIdx];

        // Transform geom to world space
        float cosA = XMath.Cos(body.Angle);
        float sinA = XMath.Sin(body.Angle);
        float geomWorldX = body.X + geom.LocalX * cosA - geom.LocalY * sinA;
        float geomWorldY = body.Y + geom.LocalX * sinA + geom.LocalY * cosA;

        // Circle vs Capsule SDF
        float phi, nx, ny;
        CircleVsCapsuleSDF(geomWorldX, geomWorldY, geom.Radius, capsule, out phi, out nx, out ny);

        if (phi >= 0f)
        {
            contacts[contactIdx] = contact;
            return;
        }

        // Contact point
        float contactX = geomWorldX - nx * geom.Radius;
        float contactY = geomWorldY - ny * geom.Radius;

        // Vector from body center to contact point
        float rx = contactX - body.X;
        float ry = contactY - body.Y;

        // Compute effective masses
        float rnCross = rx * ny - ry * nx;
        float normalMass = body.InvMass + body.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        float tx = -ny;
        float ty = nx;
        float rtCross = rx * ty - ry * tx;
        float tangentMass = body.InvMass + body.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        float velocityBias = 0f;
        if (phi < -LinearSlop)
        {
            velocityBias = (Baumgarte / dt) * (-phi - LinearSlop);
        }

        contact.RigidBodyIndex = bodyIdx;
        contact.NormalX = nx;
        contact.NormalY = ny;
        contact.TangentX = tx;
        contact.TangentY = ty;
        contact.ContactX = contactX;
        contact.ContactY = contactY;
        contact.RA_X = rx;
        contact.RA_Y = ry;
        contact.Separation = phi;
        contact.NormalMass = normalMass;
        contact.TangentMass = tangentMass;
        contact.VelocityBias = velocityBias;
        contact.NormalImpulse = 0f;
        contact.TangentImpulse = 0f;
        contact.Friction = friction;
        contact.Restitution = restitution;
        contact.GeomIndex = geomLocalIdx;
        contact.ColliderType = 1; // Capsule
        contact.ColliderIndex = capsuleIdx;
        contact.IsValid = 1;

        contacts[contactIdx] = contact;
    }

    #endregion

    #region SDF Helpers

    private static void CircleVsCapsuleSDF(
        float px, float py, float radius,
        GPUCapsuleCollider capsule,
        out float phi, out float nx, out float ny)
    {
        // Capsule endpoints
        float ax = capsule.CX - capsule.UX * capsule.HalfLength;
        float ay = capsule.CY - capsule.UY * capsule.HalfLength;
        float bx = capsule.CX + capsule.UX * capsule.HalfLength;
        float by = capsule.CY + capsule.UY * capsule.HalfLength;

        // Project point onto capsule axis
        float abx = bx - ax;
        float aby = by - ay;
        float apx = px - ax;
        float apy = py - ay;

        float t = (apx * abx + apy * aby) / (abx * abx + aby * aby + Epsilon);
        t = XMath.Clamp(t, 0f, 1f);

        // Closest point on capsule axis
        float qx = ax + t * abx;
        float qy = ay + t * aby;

        // Distance from circle center to closest point
        float dx = px - qx;
        float dy = py - qy;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        if (dist < Epsilon)
        {
            // Circle center is on capsule axis, use perpendicular normal
            phi = -capsule.Radius - radius;
            nx = -capsule.UY;
            ny = capsule.UX;
        }
        else
        {
            phi = dist - capsule.Radius - radius;
            nx = dx / dist;
            ny = dy / dist;
        }
    }

    private static void CircleVsOBBSDF(
        float px, float py, float radius,
        GPUOBBCollider obb,
        out float phi, out float nx, out float ny)
    {
        // Transform point to OBB local space
        float dx = px - obb.CX;
        float dy = py - obb.CY;

        // OBB axes (ux, uy) is the local X axis, (-uy, ux) is local Y
        float localX = dx * obb.UX + dy * obb.UY;
        float localY = -dx * obb.UY + dy * obb.UX;

        // Clamp to box
        float clampedX = XMath.Clamp(localX, -obb.HalfExtentX, obb.HalfExtentX);
        float clampedY = XMath.Clamp(localY, -obb.HalfExtentY, obb.HalfExtentY);

        // Closest point on box in local space
        float closestLocalX = clampedX;
        float closestLocalY = clampedY;

        // Distance from point to closest point
        float diffX = localX - closestLocalX;
        float diffY = localY - closestLocalY;
        float dist = XMath.Sqrt(diffX * diffX + diffY * diffY);

        if (dist < Epsilon)
        {
            // Point is inside box, find nearest edge
            float distToRight = obb.HalfExtentX - localX;
            float distToLeft = localX + obb.HalfExtentX;
            float distToTop = obb.HalfExtentY - localY;
            float distToBottom = localY + obb.HalfExtentY;

            float minDist = distToRight;
            float localNx = 1f;
            float localNy = 0f;

            if (distToLeft < minDist)
            {
                minDist = distToLeft;
                localNx = -1f;
                localNy = 0f;
            }
            if (distToTop < minDist)
            {
                minDist = distToTop;
                localNx = 0f;
                localNy = 1f;
            }
            if (distToBottom < minDist)
            {
                minDist = distToBottom;
                localNx = 0f;
                localNy = -1f;
            }

            phi = -minDist - radius;

            // Transform normal back to world space
            nx = localNx * obb.UX - localNy * obb.UY;
            ny = localNx * obb.UY + localNy * obb.UX;
        }
        else
        {
            phi = dist - radius;

            // Normal in local space
            float localNx = diffX / dist;
            float localNy = diffY / dist;

            // Transform to world space
            nx = localNx * obb.UX - localNy * obb.UY;
            ny = localNx * obb.UY + localNy * obb.UX;
        }
    }

    #endregion

    #region Warm-Starting Kernels

    /// <summary>
    /// Applies cached impulses from previous frame to new contacts.
    /// Matches contacts by (RigidBodyIndex, GeomIndex, ColliderType, ColliderIndex).
    /// One thread per contact.
    /// </summary>
    public static void ApplyWarmStartKernel(
        Index1D index,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<GPUCachedContactImpulse> cache,
        int cacheSize)
    {
        var contact = contacts[index];
        if (contact.IsValid == 0)
            return;

        // Linear search through cache for matching contact
        // TODO: Use hash table for O(1) lookup in production
        for (int i = 0; i < cacheSize; i++)
        {
            var cached = cache[i];
            if (cached.IsValid != 0 &&
                cached.RigidBodyIndex == contact.RigidBodyIndex &&
                cached.GeomIndex == contact.GeomIndex &&
                cached.ColliderType == contact.ColliderType &&
                cached.ColliderIndex == contact.ColliderIndex)
            {
                // Found matching contact, apply cached impulses
                contact.NormalImpulse = cached.NormalImpulse;
                contact.TangentImpulse = cached.TangentImpulse;

                // Apply impulse to body
                var body = bodies[contact.RigidBodyIndex];
                float px = contact.NormalX * contact.NormalImpulse + contact.TangentX * contact.TangentImpulse;
                float py = contact.NormalY * contact.NormalImpulse + contact.TangentY * contact.TangentImpulse;

                Atomic.Add(ref bodies[contact.RigidBodyIndex].VelX, body.InvMass * px);
                Atomic.Add(ref bodies[contact.RigidBodyIndex].VelY, body.InvMass * py);
                Atomic.Add(ref bodies[contact.RigidBodyIndex].AngularVel, 
                    body.InvInertia * (contact.RA_X * py - contact.RA_Y * px));

                contacts[index] = contact;
                return;
            }
        }
    }

    /// <summary>
    /// Stores current contact impulses to cache for next frame warm-starting.
    /// One thread per contact.
    /// </summary>
    public static void StoreToCacheKernel(
        Index1D index,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<GPUCachedContactImpulse> cache)
    {
        var contact = contacts[index];
        if (contact.IsValid == 0)
        {
            cache[index] = new GPUCachedContactImpulse { IsValid = 0 };
            return;
        }

        cache[index] = new GPUCachedContactImpulse
        {
            RigidBodyIndex = contact.RigidBodyIndex,
            ColliderType = contact.ColliderType,
            ColliderIndex = contact.ColliderIndex,
            GeomIndex = contact.GeomIndex,
            NormalImpulse = contact.NormalImpulse,
            TangentImpulse = contact.TangentImpulse,
            IsValid = 1
        };
    }

    #endregion
}
