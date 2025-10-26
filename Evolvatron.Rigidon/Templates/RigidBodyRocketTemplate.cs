using System;

namespace Evolvatron.Core.Templates;

/// <summary>
/// Template for creating a rocket from rigid bodies connected by revolute joints.
/// Structure: Main body capsule with two leg capsules attached via hinges.
/// </summary>
public static class RigidBodyRocketTemplate
{
    /// <summary>
    /// Creates a rocket using rigid bodies and joints.
    /// Returns the rigid body indices: [body, leftLeg, rightLeg]
    /// </summary>
    /// <param name="world">World to add rocket to</param>
    /// <param name="centerX">Center X position</param>
    /// <param name="centerY">Center Y position (base of rocket body)</param>
    /// <param name="bodyHeight">Height of main rocket body capsule</param>
    /// <param name="bodyRadius">Radius of main body capsule</param>
    /// <param name="legLength">Length of each leg capsule</param>
    /// <param name="legRadius">Radius of each leg capsule</param>
    /// <param name="legSpread">Horizontal distance between leg attachment points</param>
    /// <param name="bodyMass">Mass of main body</param>
    /// <param name="legMass">Mass of each leg</param>
    /// <returns>Rigid body indices: [body, leftLeg, rightLeg]</returns>
    public static int[] CreateRocket(
        WorldState world,
        float centerX = 0f,
        float centerY = 0f,
        float bodyHeight = 2f,
        float bodyRadius = 0.3f,
        float legLength = 1.5f,
        float legRadius = 0.15f,
        float legSpread = 0.8f,
        float bodyMass = 10f,
        float legMass = 2f)
    {
        // Create main body capsule (vertical)
        // Positioned so bottom is at centerY
        // NOTE: Capsules are oriented along local X-axis, so rotate 90° for vertical
        float bodyY = centerY + bodyHeight * 0.5f;
        int body = RigidBodyFactory.CreateCapsule(
            world,
            x: centerX,
            y: bodyY,
            halfLength: bodyHeight * 0.5f,
            radius: bodyRadius,
            mass: bodyMass,
            angle: MathF.PI / 2f  // 90° to make vertical (capsules default to horizontal)
        );

        // Create left leg capsule (angled outward)
        // Capsules extend from -halfLength to +halfLength along local X-axis
        //
        // For left leg pointing down-left:
        //   - We want the leg to extend from attachment (top) down-left to foot (bottom)
        //   - Capsule local coords: -X end should be at attachment, +X end at foot
        //   - So the +X direction should point down-left
        //   - Down-left is 225° from horizontal (or -135°)

        float leftLegAngle = 225f * MathF.PI / 180f; // Down-left direction for +X axis
        float attachPointY = centerY; // Bottom of body

        // Attachment point in world coords
        float leftAttachWorldX = centerX;
        float leftAttachWorldY = centerY;

        // Leg center is halfLength away from attachment in the +X direction
        float leftLegOffsetX = MathF.Cos(leftLegAngle) * legLength * 0.5f;
        float leftLegOffsetY = MathF.Sin(leftLegAngle) * legLength * 0.5f;

        int leftLeg = RigidBodyFactory.CreateCapsule(
            world,
            x: leftAttachWorldX + leftLegOffsetX,
            y: leftAttachWorldY + leftLegOffsetY,
            halfLength: legLength * 0.5f,
            radius: legRadius,
            mass: legMass,
            angle: leftLegAngle
        );

        // Right leg points down-right: 315° from horizontal (or -45°)
        float rightLegAngle = 315f * MathF.PI / 180f; // Down-right direction for +X axis
        float rightAttachWorldX = centerX;
        float rightAttachWorldY = centerY;

        float rightLegOffsetX = MathF.Cos(rightLegAngle) * legLength * 0.5f;
        float rightLegOffsetY = MathF.Sin(rightLegAngle) * legLength * 0.5f;

        int rightLeg = RigidBodyFactory.CreateCapsule(
            world,
            x: rightAttachWorldX + rightLegOffsetX,
            y: rightAttachWorldY + rightLegOffsetY,
            halfLength: legLength * 0.5f,
            radius: legRadius,
            mass: legMass,
            angle: rightLegAngle
        );

        // Add revolute joints connecting legs to body
        // Body capsule extends along local X-axis, rotated 90° makes it vertical
        // When angle = π/2:
        //   - Local X-axis points UP (world +Y direction)
        //   - Local Y-axis points LEFT (world -X direction)
        //
        // For a vertical capsule at angle π/2:
        //   - Top end: local X = +bodyHeight/2 (local coords)
        //   - Bottom end: local X = -bodyHeight/2 (local coords)
        //
        // So bottom of body in local coords is: localX = -bodyHeight/2, localY = 0

        // Calculate actual body and leg angles
        var bodyRB = world.RigidBodies[body];
        var leftLegRB = world.RigidBodies[leftLeg];
        var rightLegRB = world.RigidBodies[rightLeg];

        // Left leg joint
        var leftJoint = new RevoluteJoint(
            bodyA: body,
            bodyB: leftLeg,
            anchorAX: -bodyHeight * 0.5f,    // Body local: bottom end (capsule extends along X)
            anchorAY: 0f,                    // Body local: center (both legs attach at center bottom)
            anchorBX: -legLength * 0.5f,     // Leg local: -X end (top of leg, at attachment)
            anchorBY: 0f                     // Leg local: center Y
        );

        // Set reference angle to current relative angle
        // Use a motor with zero speed and high max torque to lock the joint
        float leftRefAngle = leftLegRB.Angle - bodyRB.Angle;
        leftJoint.ReferenceAngle = leftRefAngle;
        leftJoint.EnableLimits = false;  // Disable limits, use motor instead
        leftJoint.EnableMotor = true;
        leftJoint.MotorSpeed = 0f;  // Zero speed = hold position
        leftJoint.MaxMotorTorque = 1000f;  // High torque to resist collapse
        world.RevoluteJoints.Add(leftJoint);

        // Right leg joint
        var rightJoint = new RevoluteJoint(
            bodyA: body,
            bodyB: rightLeg,
            anchorAX: -bodyHeight * 0.5f,    // Body local: bottom end
            anchorAY: 0f,                    // Body local: center
            anchorBX: -legLength * 0.5f,     // Leg local: -X end (top of leg, at attachment)
            anchorBY: 0f                     // Leg local: center Y
        );

        float rightRefAngle = rightLegRB.Angle - bodyRB.Angle;
        rightJoint.ReferenceAngle = rightRefAngle;
        rightJoint.EnableLimits = false;  // Disable limits, use motor instead
        rightJoint.EnableMotor = true;
        rightJoint.MotorSpeed = 0f;  // Zero speed = hold position
        rightJoint.MaxMotorTorque = 1000f;  // High torque to resist collapse
        world.RevoluteJoints.Add(rightJoint);

        return new int[] { body, leftLeg, rightLeg };
    }

    /// <summary>
    /// Applies thrust force to the rocket body in the upward direction (relative to body orientation).
    /// </summary>
    public static void ApplyThrust(
        WorldState world,
        int[] rocketIndices,
        float throttle,
        float maxThrust = 100f)
    {
        int bodyIndex = rocketIndices[0];
        var body = world.RigidBodies[bodyIndex];

        // Thrust direction is along body's Y axis (upward in local space)
        float thrustDirX = -MathF.Sin(body.Angle);
        float thrustDirY = MathF.Cos(body.Angle);

        // Apply force at center of mass (no torque from thrust)
        float thrust = throttle * maxThrust;
        body.VelX += thrustDirX * thrust * body.InvMass * (1f / 240f); // Assuming dt = 1/240
        body.VelY += thrustDirY * thrust * body.InvMass * (1f / 240f);

        world.RigidBodies[bodyIndex] = body;
    }

    /// <summary>
    /// Applies gimbal torque to control rocket orientation.
    /// </summary>
    public static void ApplyGimbal(
        WorldState world,
        int[] rocketIndices,
        float gimbalTorque)
    {
        int bodyIndex = rocketIndices[0];
        var body = world.RigidBodies[bodyIndex];

        // Apply torque directly to body
        body.AngularVel += gimbalTorque * body.InvInertia * (1f / 240f);

        world.RigidBodies[bodyIndex] = body;
    }

    /// <summary>
    /// Gets the center of mass of the rocket (weighted average of all parts).
    /// </summary>
    public static void GetCenterOfMass(
        WorldState world,
        int[] rocketIndices,
        out float comX, out float comY)
    {
        float totalMass = 0f;
        comX = 0f;
        comY = 0f;

        foreach (int idx in rocketIndices)
        {
            var rb = world.RigidBodies[idx];
            float mass = rb.InvMass > 0f ? 1f / rb.InvMass : 0f;
            totalMass += mass;
            comX += rb.X * mass;
            comY += rb.Y * mass;
        }

        if (totalMass > 0f)
        {
            comX /= totalMass;
            comY /= totalMass;
        }
    }

    /// <summary>
    /// Gets the velocity of the rocket's center of mass.
    /// </summary>
    public static void GetVelocity(
        WorldState world,
        int[] rocketIndices,
        out float velX, out float velY)
    {
        float totalMass = 0f;
        velX = 0f;
        velY = 0f;

        foreach (int idx in rocketIndices)
        {
            var rb = world.RigidBodies[idx];
            float mass = rb.InvMass > 0f ? 1f / rb.InvMass : 0f;
            totalMass += mass;
            velX += rb.VelX * mass;
            velY += rb.VelY * mass;
        }

        if (totalMass > 0f)
        {
            velX /= totalMass;
            velY /= totalMass;
        }
    }

    /// <summary>
    /// Gets the rocket body's upright vector (normalized orientation).
    /// </summary>
    public static void GetUpVector(
        WorldState world,
        int[] rocketIndices,
        out float ux, out float uy)
    {
        int bodyIndex = rocketIndices[0];
        var body = world.RigidBodies[bodyIndex];

        // Body's local +Y axis in world space
        ux = -MathF.Sin(body.Angle);
        uy = MathF.Cos(body.Angle);
    }
}
