using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// Batched contact detection and solving kernels for N parallel worlds.
/// Static colliders are shared across all worlds; each world gets its own contact list.
///
/// Key insight: Arena walls (OBB colliders) are identical for all worlds, so they
/// are stored once and accessed by all worlds during contact detection.
/// </summary>
public static class GPUBatchedContactKernels
{
    private const float Epsilon = 1e-9f;
    private const float Baumgarte = 0.2f;
    private const float LinearSlop = 0.01f;

    #region Contact Count Management

    /// <summary>
    /// Clear contact counts for all worlds.
    /// One thread per world.
    /// </summary>
    public static void BatchedClearContactCountsKernel(
        Index1D worldIdx,
        ArrayView<int> contactCounts)
    {
        contactCounts[worldIdx] = 0;
    }

    #endregion

    #region Contact Detection Kernels

    /// <summary>
    /// Detect contacts between geoms and shared OBB colliders across all worlds.
    /// Each world's geoms are tested against the same shared colliders.
    /// One thread per (geom, collider) combination across all worlds.
    ///
    /// Thread indexing: globalIdx = worldIdx * geomsPerWorld * colliderCount
    ///                            + geomLocalIdx * colliderCount + colliderIdx
    /// </summary>
    public static void BatchedDetectOBBContactsKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPUOBBCollider> sharedColliders,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<int> contactCounts,
        int worldCount,
        int bodiesPerWorld,
        int geomsPerWorld,
        int colliderCount,
        int maxContactsPerWorld,
        float friction,
        float restitution,
        float dt)
    {
        // Decode indices
        int totalPerWorld = geomsPerWorld * colliderCount;
        int worldIdx = globalIdx / totalPerWorld;
        int remainder = globalIdx % totalPerWorld;
        int geomLocalIdx = remainder / colliderCount;
        int colliderIdx = remainder % colliderCount;

        if (worldIdx >= worldCount || colliderIdx >= colliderCount)
            return;

        // Get global geom index
        int geomGlobalIdx = worldIdx * geomsPerWorld + geomLocalIdx;
        var geom = geoms[geomGlobalIdx];

        // Find which body this geom belongs to
        // Scan through bodies in this world to find the owner
        int bodyGlobalIdx = -1;
        int geomLocalIdxInBody = -1;

        for (int b = 0; b < bodiesPerWorld; b++)
        {
            int bIdx = worldIdx * bodiesPerWorld + b;
            var body = bodies[bIdx];
            if (body.InvMass <= 0f) continue; // Skip static bodies

            // Check if this geom belongs to this body
            // GeomStartIndex in batched mode is local to the world
            int geomStart = body.GeomStartIndex;
            int geomEnd = geomStart + body.GeomCount;

            if (geomLocalIdx >= geomStart && geomLocalIdx < geomEnd)
            {
                bodyGlobalIdx = bIdx;
                geomLocalIdxInBody = geomLocalIdx - geomStart;
                break;
            }
        }

        // If no body owns this geom (or body is static), skip
        if (bodyGlobalIdx < 0)
            return;

        var body2 = bodies[bodyGlobalIdx];
        var obb = sharedColliders[colliderIdx];

        // Transform geom to world space
        float cosA = XMath.Cos(body2.Angle);
        float sinA = XMath.Sin(body2.Angle);
        float geomWorldX = body2.X + geom.LocalX * cosA - geom.LocalY * sinA;
        float geomWorldY = body2.Y + geom.LocalX * sinA + geom.LocalY * cosA;

        // Circle vs OBB SDF
        float phi, nx, ny;
        CircleVsOBBSDF(geomWorldX, geomWorldY, geom.Radius, obb, out phi, out nx, out ny);

        if (phi >= 0f)
            return; // No collision

        // Contact detected - atomically allocate slot
        int localContactIdx = Atomic.Add(ref contactCounts[worldIdx], 1);

        if (localContactIdx >= maxContactsPerWorld)
            return; // Contact buffer full

        int contactGlobalIdx = worldIdx * maxContactsPerWorld + localContactIdx;

        // Contact point on geom surface
        float contactX = geomWorldX - nx * geom.Radius;
        float contactY = geomWorldY - ny * geom.Radius;

        // Vector from body center to contact point
        float rx = contactX - body2.X;
        float ry = contactY - body2.Y;

        // Compute effective masses
        float rnCross = rx * ny - ry * nx;
        float normalMass = body2.InvMass + body2.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        float tx = -ny;
        float ty = nx;
        float rtCross = rx * ty - ry * tx;
        float tangentMass = body2.InvMass + body2.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        // Baumgarte velocity bias for penetration correction
        float velocityBias = 0f;
        if (phi < -LinearSlop)
        {
            velocityBias = (Baumgarte / dt) * (-phi - LinearSlop);
        }

        // Build contact constraint
        var contact = new GPUContactConstraint
        {
            RigidBodyIndex = bodyGlobalIdx % bodiesPerWorld, // Local body index within world
            NormalX = nx,
            NormalY = ny,
            TangentX = tx,
            TangentY = ty,
            ContactX = contactX,
            ContactY = contactY,
            RA_X = rx,
            RA_Y = ry,
            Separation = phi,
            NormalMass = normalMass,
            TangentMass = tangentMass,
            VelocityBias = velocityBias,
            NormalImpulse = 0f,
            TangentImpulse = 0f,
            Friction = friction,
            Restitution = restitution,
            GeomIndex = geomLocalIdxInBody,
            ColliderType = 2, // OBB
            ColliderIndex = colliderIdx,
            IsValid = 1
        };

        contacts[contactGlobalIdx] = contact;
    }

    /// <summary>
    /// Detect contacts between geoms and shared circle colliders across all worlds.
    /// One thread per (geom, collider) combination across all worlds.
    /// </summary>
    public static void BatchedDetectCircleContactsKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPUCircleCollider> sharedColliders,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<int> contactCounts,
        int worldCount,
        int bodiesPerWorld,
        int geomsPerWorld,
        int colliderCount,
        int maxContactsPerWorld,
        float friction,
        float restitution,
        float dt)
    {
        // Decode indices
        int totalPerWorld = geomsPerWorld * colliderCount;
        int worldIdx = globalIdx / totalPerWorld;
        int remainder = globalIdx % totalPerWorld;
        int geomLocalIdx = remainder / colliderCount;
        int colliderIdx = remainder % colliderCount;

        if (worldIdx >= worldCount || colliderIdx >= colliderCount)
            return;

        // Get global geom index
        int geomGlobalIdx = worldIdx * geomsPerWorld + geomLocalIdx;
        var geom = geoms[geomGlobalIdx];

        // Find which body this geom belongs to
        int bodyGlobalIdx = -1;
        int geomLocalIdxInBody = -1;

        for (int b = 0; b < bodiesPerWorld; b++)
        {
            int bIdx = worldIdx * bodiesPerWorld + b;
            var body = bodies[bIdx];
            if (body.InvMass <= 0f) continue;

            int geomStart = body.GeomStartIndex;
            int geomEnd = geomStart + body.GeomCount;

            if (geomLocalIdx >= geomStart && geomLocalIdx < geomEnd)
            {
                bodyGlobalIdx = bIdx;
                geomLocalIdxInBody = geomLocalIdx - geomStart;
                break;
            }
        }

        if (bodyGlobalIdx < 0)
            return;

        var body2 = bodies[bodyGlobalIdx];
        var circle = sharedColliders[colliderIdx];

        // Transform geom to world space
        float cosA = XMath.Cos(body2.Angle);
        float sinA = XMath.Sin(body2.Angle);
        float geomWorldX = body2.X + geom.LocalX * cosA - geom.LocalY * sinA;
        float geomWorldY = body2.Y + geom.LocalX * sinA + geom.LocalY * cosA;

        // Circle-circle distance
        float dx = geomWorldX - circle.CX;
        float dy = geomWorldY - circle.CY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);
        float combinedRadius = geom.Radius + circle.Radius;

        if (dist >= combinedRadius)
            return; // No collision

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

        // Contact detected - atomically allocate slot
        int localContactIdx = Atomic.Add(ref contactCounts[worldIdx], 1);

        if (localContactIdx >= maxContactsPerWorld)
            return;

        int contactGlobalIdx = worldIdx * maxContactsPerWorld + localContactIdx;

        // Contact point on geom surface
        float contactX = geomWorldX - nx * geom.Radius;
        float contactY = geomWorldY - ny * geom.Radius;

        // Vector from body center to contact point
        float rx = contactX - body2.X;
        float ry = contactY - body2.Y;

        // Compute effective masses
        float rnCross = rx * ny - ry * nx;
        float normalMass = body2.InvMass + body2.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        float tx = -ny;
        float ty = nx;
        float rtCross = rx * ty - ry * tx;
        float tangentMass = body2.InvMass + body2.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        float velocityBias = 0f;
        if (penetration < -LinearSlop)
        {
            velocityBias = (Baumgarte / dt) * (-penetration - LinearSlop);
        }

        var contact = new GPUContactConstraint
        {
            RigidBodyIndex = bodyGlobalIdx % bodiesPerWorld,
            NormalX = nx,
            NormalY = ny,
            TangentX = tx,
            TangentY = ty,
            ContactX = contactX,
            ContactY = contactY,
            RA_X = rx,
            RA_Y = ry,
            Separation = penetration,
            NormalMass = normalMass,
            TangentMass = tangentMass,
            VelocityBias = velocityBias,
            NormalImpulse = 0f,
            TangentImpulse = 0f,
            Friction = friction,
            Restitution = restitution,
            GeomIndex = geomLocalIdxInBody,
            ColliderType = 0, // Circle
            ColliderIndex = colliderIdx,
            IsValid = 1
        };

        contacts[contactGlobalIdx] = contact;
    }

    /// <summary>
    /// Detect contacts between geoms and shared capsule colliders across all worlds.
    /// One thread per (geom, collider) combination across all worlds.
    /// </summary>
    public static void BatchedDetectCapsuleContactsKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPUCapsuleCollider> sharedColliders,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<int> contactCounts,
        int worldCount,
        int bodiesPerWorld,
        int geomsPerWorld,
        int colliderCount,
        int maxContactsPerWorld,
        float friction,
        float restitution,
        float dt)
    {
        // Decode indices
        int totalPerWorld = geomsPerWorld * colliderCount;
        int worldIdx = globalIdx / totalPerWorld;
        int remainder = globalIdx % totalPerWorld;
        int geomLocalIdx = remainder / colliderCount;
        int colliderIdx = remainder % colliderCount;

        if (worldIdx >= worldCount || colliderIdx >= colliderCount)
            return;

        int geomGlobalIdx = worldIdx * geomsPerWorld + geomLocalIdx;
        var geom = geoms[geomGlobalIdx];

        // Find which body this geom belongs to
        int bodyGlobalIdx = -1;
        int geomLocalIdxInBody = -1;

        for (int b = 0; b < bodiesPerWorld; b++)
        {
            int bIdx = worldIdx * bodiesPerWorld + b;
            var body = bodies[bIdx];
            if (body.InvMass <= 0f) continue;

            int geomStart = body.GeomStartIndex;
            int geomEnd = geomStart + body.GeomCount;

            if (geomLocalIdx >= geomStart && geomLocalIdx < geomEnd)
            {
                bodyGlobalIdx = bIdx;
                geomLocalIdxInBody = geomLocalIdx - geomStart;
                break;
            }
        }

        if (bodyGlobalIdx < 0)
            return;

        var body2 = bodies[bodyGlobalIdx];
        var capsule = sharedColliders[colliderIdx];

        // Transform geom to world space
        float cosA = XMath.Cos(body2.Angle);
        float sinA = XMath.Sin(body2.Angle);
        float geomWorldX = body2.X + geom.LocalX * cosA - geom.LocalY * sinA;
        float geomWorldY = body2.Y + geom.LocalX * sinA + geom.LocalY * cosA;

        // Circle vs Capsule SDF
        float phi, nx, ny;
        CircleVsCapsuleSDF(geomWorldX, geomWorldY, geom.Radius, capsule, out phi, out nx, out ny);

        if (phi >= 0f)
            return;

        // Contact detected
        int localContactIdx = Atomic.Add(ref contactCounts[worldIdx], 1);

        if (localContactIdx >= maxContactsPerWorld)
            return;

        int contactGlobalIdx = worldIdx * maxContactsPerWorld + localContactIdx;

        float contactX = geomWorldX - nx * geom.Radius;
        float contactY = geomWorldY - ny * geom.Radius;

        float rx = contactX - body2.X;
        float ry = contactY - body2.Y;

        float rnCross = rx * ny - ry * nx;
        float normalMass = body2.InvMass + body2.InvInertia * rnCross * rnCross;
        normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

        float tx = -ny;
        float ty = nx;
        float rtCross = rx * ty - ry * tx;
        float tangentMass = body2.InvMass + body2.InvInertia * rtCross * rtCross;
        tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

        float velocityBias = 0f;
        if (phi < -LinearSlop)
        {
            velocityBias = (Baumgarte / dt) * (-phi - LinearSlop);
        }

        var contact = new GPUContactConstraint
        {
            RigidBodyIndex = bodyGlobalIdx % bodiesPerWorld,
            NormalX = nx,
            NormalY = ny,
            TangentX = tx,
            TangentY = ty,
            ContactX = contactX,
            ContactY = contactY,
            RA_X = rx,
            RA_Y = ry,
            Separation = phi,
            NormalMass = normalMass,
            TangentMass = tangentMass,
            VelocityBias = velocityBias,
            NormalImpulse = 0f,
            TangentImpulse = 0f,
            Friction = friction,
            Restitution = restitution,
            GeomIndex = geomLocalIdxInBody,
            ColliderType = 1, // Capsule
            ColliderIndex = colliderIdx,
            IsValid = 1
        };

        contacts[contactGlobalIdx] = contact;
    }

    #endregion

    #region Contact Velocity Solving

    /// <summary>
    /// Solve contact velocity constraints for all worlds.
    /// Uses sequential impulse method with friction (Coulomb cone).
    /// One thread per contact slot across all worlds.
    /// </summary>
    public static void BatchedSolveContactVelocitiesKernel(
        Index1D globalContactIdx,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<GPURigidBody> bodies,
        ArrayView<int> contactCounts,
        int worldCount,
        int bodiesPerWorld,
        int maxContactsPerWorld)
    {
        // Compute world index and local contact index
        int worldIdx = globalContactIdx / maxContactsPerWorld;
        int localContactIdx = globalContactIdx % maxContactsPerWorld;

        if (worldIdx >= worldCount)
            return;

        // Check if this contact slot is valid
        int validContactCount = contactCounts[worldIdx];
        if (localContactIdx >= validContactCount)
            return;

        var contact = contacts[globalContactIdx];
        if (contact.IsValid == 0)
            return;

        // Get the body (RigidBodyIndex is local to the world)
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + contact.RigidBodyIndex;
        var body = bodies[bodyGlobalIdx];

        if (body.InvMass <= 0f)
            return;

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

        Atomic.Add(ref bodies[bodyGlobalIdx].VelX, body.InvMass * px);
        Atomic.Add(ref bodies[bodyGlobalIdx].VelY, body.InvMass * py);
        Atomic.Add(ref bodies[bodyGlobalIdx].AngularVel, body.InvInertia * (contact.RA_X * py - contact.RA_Y * px));

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

        Atomic.Add(ref bodies[bodyGlobalIdx].VelX, body.InvMass * px);
        Atomic.Add(ref bodies[bodyGlobalIdx].VelY, body.InvMass * py);
        Atomic.Add(ref bodies[bodyGlobalIdx].AngularVel, body.InvInertia * (contact.RA_X * py - contact.RA_Y * px));

        // Write back updated contact (accumulated impulses)
        contacts[globalContactIdx] = contact;
    }

    #endregion

    #region Warm-Starting

    /// <summary>
    /// Apply cached impulses from previous frame for warm-starting.
    /// One thread per contact slot.
    /// </summary>
    public static void BatchedApplyWarmStartKernel(
        Index1D globalContactIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<GPUCachedContactImpulse> cache,
        ArrayView<int> contactCounts,
        int worldCount,
        int bodiesPerWorld,
        int maxContactsPerWorld,
        int cacheSize)
    {
        int worldIdx = globalContactIdx / maxContactsPerWorld;
        int localContactIdx = globalContactIdx % maxContactsPerWorld;

        if (worldIdx >= worldCount)
            return;

        int validContactCount = contactCounts[worldIdx];
        if (localContactIdx >= validContactCount)
            return;

        var contact = contacts[globalContactIdx];
        if (contact.IsValid == 0)
            return;

        // Linear search through cache for matching contact
        // Cache is per-world, offset by worldIdx * cacheSize
        int cacheStart = worldIdx * cacheSize;
        int cacheEnd = cacheStart + cacheSize;

        for (int i = cacheStart; i < cacheEnd; i++)
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
                int bodyGlobalIdx = worldIdx * bodiesPerWorld + contact.RigidBodyIndex;
                var body = bodies[bodyGlobalIdx];

                float px = contact.NormalX * contact.NormalImpulse + contact.TangentX * contact.TangentImpulse;
                float py = contact.NormalY * contact.NormalImpulse + contact.TangentY * contact.TangentImpulse;

                Atomic.Add(ref bodies[bodyGlobalIdx].VelX, body.InvMass * px);
                Atomic.Add(ref bodies[bodyGlobalIdx].VelY, body.InvMass * py);
                Atomic.Add(ref bodies[bodyGlobalIdx].AngularVel,
                    body.InvInertia * (contact.RA_X * py - contact.RA_Y * px));

                contacts[globalContactIdx] = contact;
                return;
            }
        }
    }

    /// <summary>
    /// Store contact impulses to cache for next frame warm-starting.
    /// One thread per contact slot.
    /// </summary>
    public static void BatchedStoreToCacheKernel(
        Index1D globalContactIdx,
        ArrayView<GPUContactConstraint> contacts,
        ArrayView<GPUCachedContactImpulse> cache,
        ArrayView<int> contactCounts,
        int worldCount,
        int maxContactsPerWorld)
    {
        int worldIdx = globalContactIdx / maxContactsPerWorld;
        int localContactIdx = globalContactIdx % maxContactsPerWorld;

        if (worldIdx >= worldCount)
            return;

        int validContactCount = contactCounts[worldIdx];

        // Store to same position in cache
        if (localContactIdx >= validContactCount)
        {
            cache[globalContactIdx] = new GPUCachedContactImpulse { IsValid = 0 };
            return;
        }

        var contact = contacts[globalContactIdx];
        if (contact.IsValid == 0)
        {
            cache[globalContactIdx] = new GPUCachedContactImpulse { IsValid = 0 };
            return;
        }

        cache[globalContactIdx] = new GPUCachedContactImpulse
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

    #region SDF Helpers

    /// <summary>
    /// Compute signed distance and normal from circle to capsule collider.
    /// </summary>
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

    /// <summary>
    /// Compute signed distance and normal from circle to OBB collider.
    /// </summary>
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

        // Distance from point to closest point on box
        float diffX = localX - clampedX;
        float diffY = localY - clampedY;
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
}
