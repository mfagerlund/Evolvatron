using System.Collections.Generic;

namespace Evolvatron.Core.Physics;

/// <summary>
/// Solves rigid body joint constraints using sequential impulse method.
/// Based on Box2D's joint solver implementation.
/// </summary>
public static class RigidBodyJointSolver
{
    private const float Epsilon = 1e-9f;

    /// <summary>
    /// Initialize joint constraints from joint definitions.
    /// Computes effective masses and prepares solver data.
    /// </summary>
    public static List<RevoluteJointConstraint> InitializeConstraints(WorldState world, float dt)
    {
        var constraints = new List<RevoluteJointConstraint>();

        foreach (var joint in world.RevoluteJoints)
        {
            var bodyA = world.RigidBodies[joint.BodyA];
            var bodyB = world.RigidBodies[joint.BodyB];

            // Skip if either body is static
            if (bodyA.InvMass == 0f && bodyB.InvMass == 0f)
                continue;

            var constraint = new RevoluteJointConstraint
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

            // Compute effective mass for position constraint
            // This is a 2x2 matrix that accounts for both linear and angular motion

            // Transform anchors to world space
            float cosA = MathF.Cos(bodyA.Angle);
            float sinA = MathF.Sin(bodyA.Angle);
            float rAX_world = constraint.RA_X * cosA - constraint.RA_Y * sinA;
            float rAY_world = constraint.RA_X * sinA + constraint.RA_Y * cosA;

            float cosB = MathF.Cos(bodyB.Angle);
            float sinB = MathF.Sin(bodyB.Angle);
            float rBX_world = constraint.RB_X * cosB - constraint.RB_Y * sinB;
            float rBY_world = constraint.RB_X * sinB + constraint.RB_Y * cosB;

            // K = invMassA + invMassB + invInertiaA * (rA × I)² + invInertiaB * (rB × I)²
            // Where I is the identity matrix, and × is cross product (scalar in 2D)

            float k11 = bodyA.InvMass + bodyB.InvMass;
            float k12 = 0f;
            float k21 = 0f;
            float k22 = bodyA.InvMass + bodyB.InvMass;

            // Add rotational contributions
            // For X constraint: add invInertia * rY²
            k11 += bodyA.InvInertia * rAY_world * rAY_world + bodyB.InvInertia * rBY_world * rBY_world;
            k22 += bodyA.InvInertia * rAX_world * rAX_world + bodyB.InvInertia * rBX_world * rBX_world;

            // Off-diagonal terms: -invInertia * rX * rY
            float k12_a = -bodyA.InvInertia * rAX_world * rAY_world;
            float k12_b = -bodyB.InvInertia * rBX_world * rBY_world;
            k12 = k12_a + k12_b;
            k21 = k12; // Matrix is symmetric

            constraint.K11 = k11;
            constraint.K12 = k12;
            constraint.K21 = k21;
            constraint.K22 = k22;

            // Invert the K matrix to get effective mass matrix
            float det = k11 * k22 - k12 * k21;
            if (MathF.Abs(det) > Epsilon)
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
            if (constraint.EnableLimits)
            {
                float angularMass = bodyA.InvInertia + bodyB.InvInertia;
                constraint.AngleLimitMass = angularMass > Epsilon ? 1f / angularMass : 0f;
            }

            // Effective mass for motor
            if (constraint.EnableMotor)
            {
                float motorMass = bodyA.InvInertia + bodyB.InvInertia;
                constraint.MotorMass = motorMass > Epsilon ? 1f / motorMass : 0f;
            }

            constraints.Add(constraint);
        }

        return constraints;
    }

    /// <summary>
    /// Solve velocity constraints for all joints.
    /// This enforces the joint constraints by applying impulses.
    /// </summary>
    public static void SolveVelocityConstraints(WorldState world, List<RevoluteJointConstraint> constraints)
    {
        for (int i = 0; i < constraints.Count; i++)
        {
            var constraint = constraints[i];
            var bodyA = world.RigidBodies[constraint.BodyAIndex];
            var bodyB = world.RigidBodies[constraint.BodyBIndex];

            // Transform anchors to world space
            float cosA = MathF.Cos(bodyA.Angle);
            float sinA = MathF.Sin(bodyA.Angle);
            float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
            float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

            float cosB = MathF.Cos(bodyB.Angle);
            float sinB = MathF.Sin(bodyB.Angle);
            float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
            float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

            // === SOLVE MOTOR ===
            if (constraint.EnableMotor)
            {
                // Motor constraint: enforce relative angular velocity
                float angularVel = bodyB.AngularVel - bodyA.AngularVel;
                float motorError = angularVel - constraint.MotorSpeed;

                float motorImpulse = -constraint.MotorMass * motorError;

                // Clamp to max torque
                float oldMotorImpulse = constraint.MotorImpulse;
                float maxImpulse = constraint.MaxMotorTorque * (1f / 240f); // Assuming dt = 1/240
                constraint.MotorImpulse = MathF.Max(-maxImpulse, MathF.Min(oldMotorImpulse + motorImpulse, maxImpulse));
                motorImpulse = constraint.MotorImpulse - oldMotorImpulse;

                bodyA.AngularVel -= bodyA.InvInertia * motorImpulse;
                bodyB.AngularVel += bodyB.InvInertia * motorImpulse;
            }

            // === SOLVE ANGLE LIMITS ===
            if (constraint.EnableLimits)
            {
                float angle = bodyB.Angle - bodyA.Angle - constraint.ReferenceAngle;

                // Normalize angle to [-π, π]
                while (angle > MathF.PI) angle -= 2f * MathF.PI;
                while (angle < -MathF.PI) angle += 2f * MathF.PI;

                float limitImpulse = 0f;

                if (angle < constraint.LowerAngle)
                {
                    // Lower limit violated
                    float angularError = angle - constraint.LowerAngle;
                    limitImpulse = -constraint.AngleLimitMass * angularError;

                    // Limit must push angle upward (positive impulse)
                    float oldImpulse = constraint.AngleLimitImpulse;
                    constraint.AngleLimitImpulse = MathF.Max(oldImpulse + limitImpulse, 0f);
                    limitImpulse = constraint.AngleLimitImpulse - oldImpulse;
                }
                else if (angle > constraint.UpperAngle)
                {
                    // Upper limit violated
                    float angularError = angle - constraint.UpperAngle;
                    limitImpulse = -constraint.AngleLimitMass * angularError;

                    // Limit must push angle downward (negative impulse)
                    float oldImpulse = constraint.AngleLimitImpulse;
                    constraint.AngleLimitImpulse = MathF.Min(oldImpulse + limitImpulse, 0f);
                    limitImpulse = constraint.AngleLimitImpulse - oldImpulse;
                }

                bodyA.AngularVel -= bodyA.InvInertia * limitImpulse;
                bodyB.AngularVel += bodyB.InvInertia * limitImpulse;
            }

            // === SOLVE POSITION CONSTRAINT ===
            // Constraint: anchors must coincide in world space
            // C = (posB + rB) - (posA + rA) = 0

            // Compute relative velocity at anchor points
            // vA = velA + omegaA × rA
            // vB = velB + omegaB × rB
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

            // Apply impulse to bodies
            bodyA.VelX -= bodyA.InvMass * lambdaX;
            bodyA.VelY -= bodyA.InvMass * lambdaY;
            bodyA.AngularVel -= bodyA.InvInertia * (rAX * lambdaY - rAY * lambdaX);

            bodyB.VelX += bodyB.InvMass * lambdaX;
            bodyB.VelY += bodyB.InvMass * lambdaY;
            bodyB.AngularVel += bodyB.InvInertia * (rBX * lambdaY - rBY * lambdaX);

            // Write back
            constraints[i] = constraint;
            world.RigidBodies[constraint.BodyAIndex] = bodyA;
            world.RigidBodies[constraint.BodyBIndex] = bodyB;
        }
    }

    /// <summary>
    /// Solve position constraints (optional, for additional stability).
    /// This directly corrects position/angle errors that remain after velocity solving.
    /// </summary>
    public static void SolvePositionConstraints(WorldState world, List<RevoluteJointConstraint> constraints)
    {
        const float MaxLinearCorrection = 0.2f;
        const float LinearSlop = 0.005f;
        const float AngularSlop = 2f * MathF.PI / 180f; // 2 degrees

        for (int i = 0; i < constraints.Count; i++)
        {
            var constraint = constraints[i];
            var bodyA = world.RigidBodies[constraint.BodyAIndex];
            var bodyB = world.RigidBodies[constraint.BodyBIndex];

            // Transform anchors to world space
            float cosA = MathF.Cos(bodyA.Angle);
            float sinA = MathF.Sin(bodyA.Angle);
            float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
            float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

            float cosB = MathF.Cos(bodyB.Angle);
            float sinB = MathF.Sin(bodyB.Angle);
            float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
            float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

            // Compute position error
            float worldAX = bodyA.X + rAX;
            float worldAY = bodyA.Y + rAY;
            float worldBX = bodyB.X + rBX;
            float worldBY = bodyB.Y + rBY;

            float errorX = worldBX - worldAX;
            float errorY = worldBY - worldAY;
            float errorLength = MathF.Sqrt(errorX * errorX + errorY * errorY);

            if (errorLength > LinearSlop)
            {
                // Clamp correction
                float correction = MathF.Min(errorLength, MaxLinearCorrection);
                float scale = correction / errorLength;
                errorX *= scale;
                errorY *= scale;

                // Compute position impulse
                float impulseX = -(constraint.Mass11 * errorX + constraint.Mass12 * errorY);
                float impulseY = -(constraint.Mass21 * errorX + constraint.Mass22 * errorY);

                // Apply position corrections
                bodyA.X -= bodyA.InvMass * impulseX;
                bodyA.Y -= bodyA.InvMass * impulseY;
                bodyA.Angle -= bodyA.InvInertia * (rAX * impulseY - rAY * impulseX);

                bodyB.X += bodyB.InvMass * impulseX;
                bodyB.Y += bodyB.InvMass * impulseY;
                bodyB.Angle += bodyB.InvInertia * (rBX * impulseY - rBY * impulseX);
            }

            // Solve angle limit position errors
            if (constraint.EnableLimits)
            {
                float angle = bodyB.Angle - bodyA.Angle - constraint.ReferenceAngle;

                // Normalize angle
                while (angle > MathF.PI) angle -= 2f * MathF.PI;
                while (angle < -MathF.PI) angle += 2f * MathF.PI;

                float angleError = 0f;
                if (angle < constraint.LowerAngle - AngularSlop)
                {
                    angleError = angle - constraint.LowerAngle;
                }
                else if (angle > constraint.UpperAngle + AngularSlop)
                {
                    angleError = angle - constraint.UpperAngle;
                }

                if (MathF.Abs(angleError) > Epsilon)
                {
                    float angleImpulse = -constraint.AngleLimitMass * angleError;
                    bodyA.Angle -= bodyA.InvInertia * angleImpulse;
                    bodyB.Angle += bodyB.InvInertia * angleImpulse;
                }
            }

            // Write back
            world.RigidBodies[constraint.BodyAIndex] = bodyA;
            world.RigidBodies[constraint.BodyBIndex] = bodyB;
        }
    }
}
