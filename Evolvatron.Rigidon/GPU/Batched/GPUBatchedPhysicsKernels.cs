using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// ILGPU kernels for batched physics simulation of N parallel worlds.
/// Each kernel processes all items across all worlds in parallel.
///
/// Key insight: globalIdx iterates over ALL items across ALL worlds.
/// For per-world indexing: worldIdx = globalIdx / itemsPerWorld, localIdx = globalIdx % itemsPerWorld.
/// </summary>
public static class GPUBatchedPhysicsKernels
{
    private const float Epsilon = 1e-9f;

    #region Gravity and Integration

    /// <summary>
    /// Apply gravity to all rigid bodies across all worlds.
    /// One thread per rigid body (total = worldCount * bodiesPerWorld).
    /// </summary>
    public static void BatchedApplyGravityKernel(
        Index1D globalIdx,                              // 0..TotalRigidBodies-1
        ArrayView<GPURigidBody> bodies,
        int bodiesPerWorld,
        float gravityX,
        float gravityY,
        float dt)
    {
        // globalIdx covers all bodies across all worlds
        // Each body gets gravity applied independently
        var body = bodies[globalIdx];

        // Skip static bodies (invMass == 0)
        if (body.InvMass <= 0f) return;

        body.VelX += gravityX * dt;
        body.VelY += gravityY * dt;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Integrate rigid body positions from velocities (symplectic Euler).
    /// One thread per rigid body.
    /// </summary>
    public static void BatchedIntegrateKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        float dt)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        // Symplectic Euler integration
        body.X += body.VelX * dt;
        body.Y += body.VelY * dt;
        body.Angle += body.AngularVel * dt;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Apply velocity damping to all rigid bodies.
    /// Reduces velocities by a damping factor each step.
    /// </summary>
    public static void BatchedDampVelocitiesKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        float linearDamping,
        float angularDamping,
        float dt)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        float linearFactor = 1f - linearDamping * dt;
        float angularFactor = 1f - angularDamping * dt;

        // Clamp to avoid negative factors (would reverse velocities)
        if (linearFactor < 0f) linearFactor = 0f;
        if (angularFactor < 0f) angularFactor = 0f;

        body.VelX *= linearFactor;
        body.VelY *= linearFactor;
        body.AngularVel *= angularFactor;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Apply global damping using a precomputed factor.
    /// dampingFactor = max(0, 1 - globalDamping * dt)
    /// </summary>
    public static void BatchedGlobalDampingKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        float dampingFactor)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        body.VelX *= dampingFactor;
        body.VelY *= dampingFactor;
        body.AngularVel *= dampingFactor;

        bodies[globalIdx] = body;
    }

    #endregion

    #region Velocity Stabilization

    /// <summary>
    /// Save previous positions for velocity stabilization.
    /// Must be called BEFORE integration to store pre-solve positions.
    /// </summary>
    public static void BatchedSavePreviousPositionsKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies)
    {
        var body = bodies[globalIdx];

        body.PrevX = body.X;
        body.PrevY = body.Y;
        body.PrevAngle = body.Angle;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Velocity stabilization: derive velocities from position changes.
    /// v = beta * (pos - prevPos) / dt + (1 - beta) * v
    /// Call AFTER constraint solving to correct velocities based on position corrections.
    /// </summary>
    public static void BatchedVelocityStabilizationKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        float invDt,
        float beta,
        float maxVelocity)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        // Compute corrected velocity from position change
        float correctedVelX = (body.X - body.PrevX) * invDt;
        float correctedVelY = (body.Y - body.PrevY) * invDt;
        float correctedAngVel = (body.Angle - body.PrevAngle) * invDt;

        // Blend with current velocity
        float oneMinusBeta = 1f - beta;
        float vx = correctedVelX * beta + body.VelX * oneMinusBeta;
        float vy = correctedVelY * beta + body.VelY * oneMinusBeta;
        float av = correctedAngVel * beta + body.AngularVel * oneMinusBeta;

        // Clamp velocity magnitude if maxVelocity > 0
        if (maxVelocity > 0f)
        {
            float velSq = vx * vx + vy * vy;
            float maxVelSq = maxVelocity * maxVelocity;
            if (velSq > maxVelSq)
            {
                // Dissipate energy: scale down to 50% of max velocity
                float scale = (maxVelocity * 0.5f) / XMath.Sqrt(velSq);
                vx *= scale;
                vy *= scale;
            }
        }

        body.VelX = vx;
        body.VelY = vy;
        body.AngularVel = av;

        bodies[globalIdx] = body;
    }

    #endregion

    #region Geom Transformation

    /// <summary>
    /// Transform geom positions from local to world space.
    /// Must be called after rigid body positions update (after integration).
    /// One thread per geom across all worlds.
    /// </summary>
    public static void BatchedUpdateGeomPositionsKernel(
        Index1D globalGeomIdx,                          // 0..TotalGeoms-1
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPURigidBody> bodies,
        int geomsPerWorld,
        int bodiesPerWorld)
    {
        var geom = geoms[globalGeomIdx];

        // Compute world index and find the parent body
        int worldIdx = globalGeomIdx / geomsPerWorld;
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + geom.BodyIndex;

        var body = bodies[bodyGlobalIdx];

        // Transform local position to world using body's rotation
        float cos = XMath.Cos(body.Angle);
        float sin = XMath.Sin(body.Angle);

        geom.WorldX = body.X + geom.LocalX * cos - geom.LocalY * sin;
        geom.WorldY = body.Y + geom.LocalX * sin + geom.LocalY * cos;

        geoms[globalGeomIdx] = geom;
    }

    #endregion

    #region Force Application

    /// <summary>
    /// Apply a force to rigid bodies at their center of mass.
    /// Forces are specified per-world in the forces array.
    /// forceX/forceY arrays: [world0_body0_fx, world0_body0_fy, world0_body1_fx, ...]
    /// </summary>
    public static void BatchedApplyForcesKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> forceX,
        ArrayView<float> forceY,
        float dt)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        // F = ma => a = F * invMass => dv = a * dt = F * invMass * dt
        body.VelX += forceX[globalIdx] * body.InvMass * dt;
        body.VelY += forceY[globalIdx] * body.InvMass * dt;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Apply torque to rigid bodies.
    /// Torque is specified per-body.
    /// </summary>
    public static void BatchedApplyTorqueKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> torque,
        float dt)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        // T = I*alpha => alpha = T * invInertia => dw = alpha * dt
        body.AngularVel += torque[globalIdx] * body.InvInertia * dt;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Apply thrust to rocket bodies at an offset from center of mass.
    /// This applies both linear force and torque from the offset.
    ///
    /// thrustData layout per body: [thrustMagnitude, thrustDirX, thrustDirY, offsetX, offsetY]
    /// </summary>
    public static void BatchedApplyThrustKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> thrustMagnitude,
        ArrayView<float> thrustDirX,
        ArrayView<float> thrustDirY,
        ArrayView<float> offsetX,
        ArrayView<float> offsetY,
        float dt)
    {
        var body = bodies[globalIdx];

        if (body.InvMass <= 0f) return;

        float mag = thrustMagnitude[globalIdx];
        if (mag < Epsilon) return;

        float fx = thrustDirX[globalIdx] * mag;
        float fy = thrustDirY[globalIdx] * mag;

        // Apply linear acceleration
        body.VelX += fx * body.InvMass * dt;
        body.VelY += fy * body.InvMass * dt;

        // Compute torque from offset: tau = r x F = rx*Fy - ry*Fx
        float rx = offsetX[globalIdx];
        float ry = offsetY[globalIdx];
        float torque = rx * fy - ry * fx;

        body.AngularVel += torque * body.InvInertia * dt;

        bodies[globalIdx] = body;
    }

    #endregion

    #region State Reset

    /// <summary>
    /// Reset rigid body velocities to zero.
    /// Useful for episode resets.
    /// </summary>
    public static void BatchedResetVelocitiesKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies)
    {
        var body = bodies[globalIdx];

        body.VelX = 0f;
        body.VelY = 0f;
        body.AngularVel = 0f;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Set rigid body position and angle directly.
    /// Used for teleporting/resetting bodies.
    /// </summary>
    public static void BatchedSetPositionKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> newX,
        ArrayView<float> newY,
        ArrayView<float> newAngle)
    {
        var body = bodies[globalIdx];

        body.X = newX[globalIdx];
        body.Y = newY[globalIdx];
        body.Angle = newAngle[globalIdx];
        body.PrevX = body.X;
        body.PrevY = body.Y;
        body.PrevAngle = body.Angle;

        bodies[globalIdx] = body;
    }

    /// <summary>
    /// Reset specific worlds to initial state.
    /// worldResetFlags[worldIdx] = 1 means reset that world.
    /// templateBodies contains the initial state for one world (bodiesPerWorld entries).
    /// </summary>
    public static void BatchedResetWorldsKernel(
        Index1D globalIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<GPURigidBody> templateBodies,
        ArrayView<int> worldResetFlags,
        int bodiesPerWorld)
    {
        int worldIdx = globalIdx / bodiesPerWorld;
        int localBodyIdx = globalIdx % bodiesPerWorld;

        // Only reset if this world is flagged
        if (worldResetFlags[worldIdx] == 0) return;

        // Copy from template
        bodies[globalIdx] = templateBodies[localBodyIdx];
    }

    /// <summary>
    /// Reset geom world positions for worlds that were reset.
    /// Call after BatchedResetWorldsKernel.
    /// </summary>
    public static void BatchedResetWorldGeomsKernel(
        Index1D globalGeomIdx,
        ArrayView<GPURigidBodyGeom> geoms,
        ArrayView<GPURigidBodyGeom> templateGeoms,
        ArrayView<int> worldResetFlags,
        int geomsPerWorld)
    {
        int worldIdx = globalGeomIdx / geomsPerWorld;
        int localGeomIdx = globalGeomIdx % geomsPerWorld;

        if (worldResetFlags[worldIdx] == 0) return;

        geoms[globalGeomIdx] = templateGeoms[localGeomIdx];
    }

    #endregion

    #region Utility Queries

    /// <summary>
    /// Compute center of mass position for each world.
    /// Writes to output arrays indexed by world.
    /// Note: This is a reduction operation - simplified version for single-body rockets.
    /// For multi-body, use atomic adds or separate reduction kernel.
    /// </summary>
    public static void BatchedGetPrimaryBodyPositionKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> outX,
        ArrayView<float> outY,
        ArrayView<float> outAngle,
        int bodiesPerWorld,
        int primaryBodyLocalIdx)
    {
        int globalBodyIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[globalBodyIdx];

        outX[worldIdx] = body.X;
        outY[worldIdx] = body.Y;
        outAngle[worldIdx] = body.Angle;
    }

    /// <summary>
    /// Get velocities of primary body for each world.
    /// </summary>
    public static void BatchedGetPrimaryBodyVelocityKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> outVelX,
        ArrayView<float> outVelY,
        ArrayView<float> outAngularVel,
        int bodiesPerWorld,
        int primaryBodyLocalIdx)
    {
        int globalBodyIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[globalBodyIdx];

        outVelX[worldIdx] = body.VelX;
        outVelY[worldIdx] = body.VelY;
        outAngularVel[worldIdx] = body.AngularVel;
    }

    #endregion
}
