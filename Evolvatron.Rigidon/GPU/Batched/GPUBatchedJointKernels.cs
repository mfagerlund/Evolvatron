using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// Batched joint constraint solver kernels for N parallel worlds.
/// Each world has its own set of joints connecting rigid bodies within that world.
///
/// Joint constraints enforce revolute (hinge) joints between rigid body pairs,
/// keeping anchor points coincident while allowing relative rotation.
/// Supports angle limits and motors.
///
/// Based on Box2D's sequential impulse joint solver, adapted for batched GPU execution.
/// </summary>
public static class GPUBatchedJointKernels
{
    private const float Epsilon = 1e-9f;
    private const float MaxLinearCorrection = 0.2f;
    private const float LinearSlop = 0.005f;
    private const float AngularSlop = 2f * XMath.PI / 180f;  // 2 degrees

    #region Initialize Joint Constraints

    /// <summary>
    /// Initialize joint constraints from joint definitions.
    /// Computes world-space anchors, effective mass matrices, and prepares solver data.
    /// One thread per joint across all worlds.
    ///
    /// globalJointIdx = worldIdx * jointsPerWorld + localJointIdx
    /// </summary>
    public static void BatchedInitializeJointConstraintsKernel(
        Index1D globalJointIdx,
        ArrayView<GPURevoluteJoint> joints,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<GPURigidBody> bodies,
        int jointsPerWorld,
        int bodiesPerWorld,
        float dt)
    {
        // Decode world and local indices
        int worldIdx = globalJointIdx / jointsPerWorld;
        int localJointIdx = globalJointIdx % jointsPerWorld;

        var joint = joints[globalJointIdx];

        // Map local body indices to global
        int bodyAGlobal = worldIdx * bodiesPerWorld + joint.BodyA;
        int bodyBGlobal = worldIdx * bodiesPerWorld + joint.BodyB;

        var bodyA = bodies[bodyAGlobal];
        var bodyB = bodies[bodyBGlobal];

        // Skip if both bodies are static
        if (bodyA.InvMass == 0f && bodyB.InvMass == 0f)
        {
            constraints[globalJointIdx] = new GPUJointConstraint
            {
                BodyAIndex = joint.BodyA,
                BodyBIndex = joint.BodyB,
                Mass11 = 0f,
                Mass12 = 0f,
                Mass21 = 0f,
                Mass22 = 0f,
                EnableLimits = 0,
                EnableMotor = 0
            };
            return;
        }

        // Initialize constraint data
        var constraint = new GPUJointConstraint
        {
            BodyAIndex = joint.BodyA,
            BodyBIndex = joint.BodyB,
            RA_X = joint.LocalAnchorAX,
            RA_Y = joint.LocalAnchorAY,
            RB_X = joint.LocalAnchorBX,
            RB_Y = joint.LocalAnchorBY,
            EnableLimits = joint.EnableLimits,
            LowerAngle = joint.LowerAngle,
            UpperAngle = joint.UpperAngle,
            ReferenceAngle = joint.ReferenceAngle,
            EnableMotor = joint.EnableMotor,
            MotorSpeed = joint.MotorSpeed,
            MaxMotorTorque = joint.MaxMotorTorque,
            ImpulseX = 0f,
            ImpulseY = 0f,
            AngleLimitImpulse = 0f,
            MotorImpulse = 0f
        };

        // Transform local anchors to world space for effective mass computation
        float cosA = XMath.Cos(bodyA.Angle);
        float sinA = XMath.Sin(bodyA.Angle);
        float rAX_world = constraint.RA_X * cosA - constraint.RA_Y * sinA;
        float rAY_world = constraint.RA_X * sinA + constraint.RA_Y * cosA;

        float cosB = XMath.Cos(bodyB.Angle);
        float sinB = XMath.Sin(bodyB.Angle);
        float rBX_world = constraint.RB_X * cosB - constraint.RB_Y * sinB;
        float rBY_world = constraint.RB_X * sinB + constraint.RB_Y * cosB;

        // Compute effective mass matrix K
        // K = invMassA + invMassB + invInertiaA * (rA x I)^2 + invInertiaB * (rB x I)^2
        float k11 = bodyA.InvMass + bodyB.InvMass;
        float k12 = 0f;
        float k21 = 0f;
        float k22 = bodyA.InvMass + bodyB.InvMass;

        // Add rotational contributions
        // For X constraint: add invInertia * rY^2
        k11 += bodyA.InvInertia * rAY_world * rAY_world + bodyB.InvInertia * rBY_world * rBY_world;
        k22 += bodyA.InvInertia * rAX_world * rAX_world + bodyB.InvInertia * rBX_world * rBX_world;

        // Off-diagonal terms: -invInertia * rX * rY
        float k12_a = -bodyA.InvInertia * rAX_world * rAY_world;
        float k12_b = -bodyB.InvInertia * rBX_world * rBY_world;
        k12 = k12_a + k12_b;
        k21 = k12; // Matrix is symmetric

        // Invert the K matrix to get effective mass matrix
        float det = k11 * k22 - k12 * k21;
        if (XMath.Abs(det) > Epsilon)
        {
            float invDet = 1f / det;
            constraint.Mass11 = k22 * invDet;
            constraint.Mass12 = -k12 * invDet;
            constraint.Mass21 = -k21 * invDet;
            constraint.Mass22 = k11 * invDet;
        }
        else
        {
            // Singular matrix - bodies are aligned or one is static
            constraint.Mass11 = 0f;
            constraint.Mass12 = 0f;
            constraint.Mass21 = 0f;
            constraint.Mass22 = 0f;
        }

        // Effective mass for angle limit constraint
        if (constraint.EnableLimits != 0)
        {
            float angularMass = bodyA.InvInertia + bodyB.InvInertia;
            constraint.AngleLimitMass = angularMass > Epsilon ? 1f / angularMass : 0f;
        }
        else
        {
            constraint.AngleLimitMass = 0f;
        }

        // Effective mass for motor
        if (constraint.EnableMotor != 0)
        {
            float motorMass = bodyA.InvInertia + bodyB.InvInertia;
            constraint.MotorMass = motorMass > Epsilon ? 1f / motorMass : 0f;
        }
        else
        {
            constraint.MotorMass = 0f;
        }

        constraints[globalJointIdx] = constraint;
    }

    #endregion

    #region Solve Joint Velocities

    /// <summary>
    /// Solve velocity constraints for all joints across all worlds.
    /// Enforces joint constraints by applying impulses to keep anchors together.
    /// Handles motors and angle limits.
    /// One thread per joint.
    /// </summary>
    public static void BatchedSolveJointVelocitiesKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<GPURigidBody> bodies,
        int jointsPerWorld,
        int bodiesPerWorld,
        float dt)
    {
        var constraint = constraints[globalJointIdx];

        // Skip invalid constraints (both bodies static)
        if (constraint.Mass11 == 0f && constraint.Mass22 == 0f &&
            constraint.MotorMass == 0f && constraint.AngleLimitMass == 0f)
        {
            return;
        }

        int worldIdx = globalJointIdx / jointsPerWorld;
        int bodyAGlobal = worldIdx * bodiesPerWorld + constraint.BodyAIndex;
        int bodyBGlobal = worldIdx * bodiesPerWorld + constraint.BodyBIndex;

        var bodyA = bodies[bodyAGlobal];
        var bodyB = bodies[bodyBGlobal];

        // Transform anchors to world space
        float cosA = XMath.Cos(bodyA.Angle);
        float sinA = XMath.Sin(bodyA.Angle);
        float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
        float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

        float cosB = XMath.Cos(bodyB.Angle);
        float sinB = XMath.Sin(bodyB.Angle);
        float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
        float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

        // === SOLVE MOTOR ===
        if (constraint.EnableMotor != 0 && constraint.MotorMass > 0f)
        {
            // Motor constraint: enforce relative angular velocity
            float angularVel = bodyB.AngularVel - bodyA.AngularVel;
            float motorError = angularVel - constraint.MotorSpeed;

            float motorImpulse = -constraint.MotorMass * motorError;

            // Clamp to max torque
            float oldMotorImpulse = constraint.MotorImpulse;
            float maxImpulse = constraint.MaxMotorTorque * dt;
            float newMotorImpulse = oldMotorImpulse + motorImpulse;
            newMotorImpulse = XMath.Max(-maxImpulse, XMath.Min(newMotorImpulse, maxImpulse));
            motorImpulse = newMotorImpulse - oldMotorImpulse;
            constraint.MotorImpulse = newMotorImpulse;

            // Apply motor impulse (using atomics for thread safety)
            Atomic.Add(ref bodies[bodyAGlobal].AngularVel, -bodyA.InvInertia * motorImpulse);
            Atomic.Add(ref bodies[bodyBGlobal].AngularVel, bodyB.InvInertia * motorImpulse);

            // Update local copies for subsequent calculations
            bodyA.AngularVel -= bodyA.InvInertia * motorImpulse;
            bodyB.AngularVel += bodyB.InvInertia * motorImpulse;
        }

        // === SOLVE ANGLE LIMITS ===
        if (constraint.EnableLimits != 0 && constraint.AngleLimitMass > 0f)
        {
            float angle = bodyB.Angle - bodyA.Angle - constraint.ReferenceAngle;

            // Normalize angle to [-pi, pi]
            while (angle > XMath.PI) angle -= 2f * XMath.PI;
            while (angle < -XMath.PI) angle += 2f * XMath.PI;

            float limitImpulse = 0f;

            if (angle < constraint.LowerAngle)
            {
                // Lower limit violated
                float angularError = angle - constraint.LowerAngle;
                limitImpulse = -constraint.AngleLimitMass * angularError;

                // Limit must push angle upward (positive impulse)
                float oldImpulse = constraint.AngleLimitImpulse;
                float newImpulse = oldImpulse + limitImpulse;
                newImpulse = XMath.Max(newImpulse, 0f);
                limitImpulse = newImpulse - oldImpulse;
                constraint.AngleLimitImpulse = newImpulse;
            }
            else if (angle > constraint.UpperAngle)
            {
                // Upper limit violated
                float angularError = angle - constraint.UpperAngle;
                limitImpulse = -constraint.AngleLimitMass * angularError;

                // Limit must push angle downward (negative impulse)
                float oldImpulse = constraint.AngleLimitImpulse;
                float newImpulse = oldImpulse + limitImpulse;
                newImpulse = XMath.Min(newImpulse, 0f);
                limitImpulse = newImpulse - oldImpulse;
                constraint.AngleLimitImpulse = newImpulse;
            }

            if (XMath.Abs(limitImpulse) > Epsilon)
            {
                // Apply limit impulse (using atomics for thread safety)
                Atomic.Add(ref bodies[bodyAGlobal].AngularVel, -bodyA.InvInertia * limitImpulse);
                Atomic.Add(ref bodies[bodyBGlobal].AngularVel, bodyB.InvInertia * limitImpulse);

                // Update local copies
                bodyA.AngularVel -= bodyA.InvInertia * limitImpulse;
                bodyB.AngularVel += bodyB.InvInertia * limitImpulse;
            }
        }

        // === SOLVE POSITION CONSTRAINT ===
        // Constraint: anchors must coincide in world space
        // C = (posB + rB) - (posA + rA) = 0

        // Compute relative velocity at anchor points
        // vA = velA + omegaA x rA
        // vB = velB + omegaB x rB
        float vAX = bodyA.VelX - bodyA.AngularVel * rAY;
        float vAY = bodyA.VelY + bodyA.AngularVel * rAX;
        float vBX = bodyB.VelX - bodyB.AngularVel * rBY;
        float vBY = bodyB.VelY + bodyB.AngularVel * rBX;

        // Relative velocity
        float relVelX = vBX - vAX;
        float relVelY = vBY - vAY;

        // Compute impulse to enforce constraint
        // lambda = -Mass * relVel
        float lambdaX = -(constraint.Mass11 * relVelX + constraint.Mass12 * relVelY);
        float lambdaY = -(constraint.Mass21 * relVelX + constraint.Mass22 * relVelY);

        // Accumulate impulse
        constraint.ImpulseX += lambdaX;
        constraint.ImpulseY += lambdaY;

        // Apply impulse to bodies (using atomics for thread safety)
        // Body A: subtract impulse
        Atomic.Add(ref bodies[bodyAGlobal].VelX, -bodyA.InvMass * lambdaX);
        Atomic.Add(ref bodies[bodyAGlobal].VelY, -bodyA.InvMass * lambdaY);
        Atomic.Add(ref bodies[bodyAGlobal].AngularVel, -bodyA.InvInertia * (rAX * lambdaY - rAY * lambdaX));

        // Body B: add impulse
        Atomic.Add(ref bodies[bodyBGlobal].VelX, bodyB.InvMass * lambdaX);
        Atomic.Add(ref bodies[bodyBGlobal].VelY, bodyB.InvMass * lambdaY);
        Atomic.Add(ref bodies[bodyBGlobal].AngularVel, bodyB.InvInertia * (rBX * lambdaY - rBY * lambdaX));

        // Write back updated constraint
        constraints[globalJointIdx] = constraint;
    }

    #endregion

    #region Solve Joint Positions

    /// <summary>
    /// Solve position constraints to correct drift.
    /// Directly adjusts positions/angles when anchors have separated.
    /// One thread per joint.
    /// </summary>
    public static void BatchedSolveJointPositionsKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<GPURigidBody> bodies,
        int jointsPerWorld,
        int bodiesPerWorld)
    {
        var constraint = constraints[globalJointIdx];

        // Skip invalid constraints
        if (constraint.Mass11 == 0f && constraint.Mass22 == 0f && constraint.AngleLimitMass == 0f)
        {
            return;
        }

        int worldIdx = globalJointIdx / jointsPerWorld;
        int bodyAGlobal = worldIdx * bodiesPerWorld + constraint.BodyAIndex;
        int bodyBGlobal = worldIdx * bodiesPerWorld + constraint.BodyBIndex;

        var bodyA = bodies[bodyAGlobal];
        var bodyB = bodies[bodyBGlobal];

        // Transform anchors to world space
        float cosA = XMath.Cos(bodyA.Angle);
        float sinA = XMath.Sin(bodyA.Angle);
        float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
        float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

        float cosB = XMath.Cos(bodyB.Angle);
        float sinB = XMath.Sin(bodyB.Angle);
        float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
        float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

        // Compute position error
        float worldAX = bodyA.X + rAX;
        float worldAY = bodyA.Y + rAY;
        float worldBX = bodyB.X + rBX;
        float worldBY = bodyB.Y + rBY;

        float errorX = worldBX - worldAX;
        float errorY = worldBY - worldAY;
        float errorLength = XMath.Sqrt(errorX * errorX + errorY * errorY);

        if (errorLength > LinearSlop)
        {
            // Clamp correction
            float correction = XMath.Min(errorLength, MaxLinearCorrection);
            float scale = correction / errorLength;
            errorX *= scale;
            errorY *= scale;

            // Compute position impulse
            float impulseX = -(constraint.Mass11 * errorX + constraint.Mass12 * errorY);
            float impulseY = -(constraint.Mass21 * errorX + constraint.Mass22 * errorY);

            // Apply position corrections (using atomics for thread safety)
            // Body A: subtract correction
            Atomic.Add(ref bodies[bodyAGlobal].X, -bodyA.InvMass * impulseX);
            Atomic.Add(ref bodies[bodyAGlobal].Y, -bodyA.InvMass * impulseY);
            Atomic.Add(ref bodies[bodyAGlobal].Angle, -bodyA.InvInertia * (rAX * impulseY - rAY * impulseX));

            // Body B: add correction
            Atomic.Add(ref bodies[bodyBGlobal].X, bodyB.InvMass * impulseX);
            Atomic.Add(ref bodies[bodyBGlobal].Y, bodyB.InvMass * impulseY);
            Atomic.Add(ref bodies[bodyBGlobal].Angle, bodyB.InvInertia * (rBX * impulseY - rBY * impulseX));
        }

        // Solve angle limit position errors
        if (constraint.EnableLimits != 0 && constraint.AngleLimitMass > 0f)
        {
            // Re-read body angles in case they were modified by position correction above
            float angleA = bodies[bodyAGlobal].Angle;
            float angleB = bodies[bodyBGlobal].Angle;

            float angle = angleB - angleA - constraint.ReferenceAngle;

            // Normalize angle to [-pi, pi]
            while (angle > XMath.PI) angle -= 2f * XMath.PI;
            while (angle < -XMath.PI) angle += 2f * XMath.PI;

            float angleError = 0f;
            if (angle < constraint.LowerAngle - AngularSlop)
            {
                angleError = angle - constraint.LowerAngle;
            }
            else if (angle > constraint.UpperAngle + AngularSlop)
            {
                angleError = angle - constraint.UpperAngle;
            }

            if (XMath.Abs(angleError) > Epsilon)
            {
                float angleImpulse = -constraint.AngleLimitMass * angleError;

                // Apply angle corrections (using atomics for thread safety)
                Atomic.Add(ref bodies[bodyAGlobal].Angle, -bodyA.InvInertia * angleImpulse);
                Atomic.Add(ref bodies[bodyBGlobal].Angle, bodyB.InvInertia * angleImpulse);
            }
        }
    }

    #endregion

    #region Warm Starting

    /// <summary>
    /// Apply cached impulses from previous frame for warm-starting joints.
    /// Accelerates convergence by starting from a good initial guess.
    /// One thread per joint.
    /// </summary>
    public static void BatchedWarmStartJointsKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<GPURigidBody> bodies,
        int jointsPerWorld,
        int bodiesPerWorld)
    {
        var constraint = constraints[globalJointIdx];

        // Skip invalid constraints
        if (constraint.Mass11 == 0f && constraint.Mass22 == 0f)
        {
            return;
        }

        // Skip if no accumulated impulse to apply
        if (XMath.Abs(constraint.ImpulseX) < Epsilon &&
            XMath.Abs(constraint.ImpulseY) < Epsilon &&
            XMath.Abs(constraint.MotorImpulse) < Epsilon &&
            XMath.Abs(constraint.AngleLimitImpulse) < Epsilon)
        {
            return;
        }

        int worldIdx = globalJointIdx / jointsPerWorld;
        int bodyAGlobal = worldIdx * bodiesPerWorld + constraint.BodyAIndex;
        int bodyBGlobal = worldIdx * bodiesPerWorld + constraint.BodyBIndex;

        var bodyA = bodies[bodyAGlobal];
        var bodyB = bodies[bodyBGlobal];

        // Transform anchors to world space
        float cosA = XMath.Cos(bodyA.Angle);
        float sinA = XMath.Sin(bodyA.Angle);
        float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
        float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

        float cosB = XMath.Cos(bodyB.Angle);
        float sinB = XMath.Sin(bodyB.Angle);
        float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
        float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

        // Apply position constraint impulse
        float lambdaX = constraint.ImpulseX;
        float lambdaY = constraint.ImpulseY;

        Atomic.Add(ref bodies[bodyAGlobal].VelX, -bodyA.InvMass * lambdaX);
        Atomic.Add(ref bodies[bodyAGlobal].VelY, -bodyA.InvMass * lambdaY);
        Atomic.Add(ref bodies[bodyAGlobal].AngularVel, -bodyA.InvInertia * (rAX * lambdaY - rAY * lambdaX));

        Atomic.Add(ref bodies[bodyBGlobal].VelX, bodyB.InvMass * lambdaX);
        Atomic.Add(ref bodies[bodyBGlobal].VelY, bodyB.InvMass * lambdaY);
        Atomic.Add(ref bodies[bodyBGlobal].AngularVel, bodyB.InvInertia * (rBX * lambdaY - rBY * lambdaX));

        // Apply motor impulse
        if (constraint.EnableMotor != 0)
        {
            float motorImpulse = constraint.MotorImpulse;
            Atomic.Add(ref bodies[bodyAGlobal].AngularVel, -bodyA.InvInertia * motorImpulse);
            Atomic.Add(ref bodies[bodyBGlobal].AngularVel, bodyB.InvInertia * motorImpulse);
        }

        // Apply angle limit impulse
        if (constraint.EnableLimits != 0)
        {
            float limitImpulse = constraint.AngleLimitImpulse;
            Atomic.Add(ref bodies[bodyAGlobal].AngularVel, -bodyA.InvInertia * limitImpulse);
            Atomic.Add(ref bodies[bodyBGlobal].AngularVel, bodyB.InvInertia * limitImpulse);
        }
    }

    /// <summary>
    /// Reset accumulated impulses to zero for a fresh start.
    /// Use this when resetting worlds or when warm-starting would be detrimental.
    /// One thread per joint.
    /// </summary>
    public static void BatchedResetJointImpulsesKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints)
    {
        var constraint = constraints[globalJointIdx];

        constraint.ImpulseX = 0f;
        constraint.ImpulseY = 0f;
        constraint.MotorImpulse = 0f;
        constraint.AngleLimitImpulse = 0f;

        constraints[globalJointIdx] = constraint;
    }

    /// <summary>
    /// Reset joint impulses only for specific worlds that are being reset.
    /// worldResetFlags[worldIdx] = 1 means reset that world's joint impulses.
    /// One thread per joint.
    /// </summary>
    public static void BatchedConditionalResetJointImpulsesKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<int> worldResetFlags,
        int jointsPerWorld)
    {
        int worldIdx = globalJointIdx / jointsPerWorld;

        // Only reset if this world is flagged
        if (worldResetFlags[worldIdx] == 0) return;

        var constraint = constraints[globalJointIdx];

        constraint.ImpulseX = 0f;
        constraint.ImpulseY = 0f;
        constraint.MotorImpulse = 0f;
        constraint.AngleLimitImpulse = 0f;

        constraints[globalJointIdx] = constraint;
    }

    #endregion

    #region Motor Control

    /// <summary>
    /// Set motor speed for joints across worlds.
    /// Used to control gimbal angle or other actuated joints.
    /// motorSpeeds array: one value per joint globally.
    /// One thread per joint.
    /// </summary>
    public static void BatchedSetMotorSpeedKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<float> motorSpeeds)
    {
        var constraint = constraints[globalJointIdx];
        constraint.MotorSpeed = motorSpeeds[globalJointIdx];
        constraints[globalJointIdx] = constraint;
    }

    /// <summary>
    /// Set motor speed uniformly per world.
    /// motorSpeedsPerWorld: one value per world, applied to all joints in that world.
    /// Useful when each world has one motorized joint (e.g., gimbal).
    /// </summary>
    public static void BatchedSetMotorSpeedPerWorldKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<float> motorSpeedsPerWorld,
        int jointsPerWorld)
    {
        int worldIdx = globalJointIdx / jointsPerWorld;

        var constraint = constraints[globalJointIdx];
        constraint.MotorSpeed = motorSpeedsPerWorld[worldIdx];
        constraints[globalJointIdx] = constraint;
    }

    /// <summary>
    /// Enable or disable motor for specific joints.
    /// enableFlags: 0 = disable, nonzero = enable.
    /// One thread per joint.
    /// </summary>
    public static void BatchedSetMotorEnabledKernel(
        Index1D globalJointIdx,
        ArrayView<GPUJointConstraint> constraints,
        ArrayView<GPURevoluteJoint> joints,
        ArrayView<byte> enableFlags)
    {
        var constraint = constraints[globalJointIdx];
        var joint = joints[globalJointIdx];

        byte enabled = enableFlags[globalJointIdx];
        constraint.EnableMotor = enabled;
        joint.EnableMotor = enabled;

        // If enabling, ensure motor mass is computed
        // (This is a simplified approach; normally would need body data)
        if (enabled != 0 && constraint.MotorMass == 0f)
        {
            // Motor mass will be recomputed on next Initialize call
            // For now, we just toggle the flag
        }

        constraints[globalJointIdx] = constraint;
        joints[globalJointIdx] = joint;
    }

    #endregion
}
