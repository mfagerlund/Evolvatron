using System;

namespace Evolvatron.Core.Templates;

/// <summary>
/// Template for creating a 5-particle rocket contraption with legs and gimbaled engine.
/// Layout: 0=top, 1=bottom, 2=leftFoot, 3=rightFoot, 4=engine
/// </summary>
public static class RocketTemplate
{
    /// <summary>
    /// Creates a 5-particle rocket and returns the particle indices.
    /// </summary>
    /// <param name="world">World to add rocket to</param>
    /// <param name="centerX">Center X position</param>
    /// <param name="centerY">Center Y position (base of rocket body)</param>
    /// <param name="bodyHeight">Height of rocket body</param>
    /// <param name="legSpread">Horizontal distance between feet</param>
    /// <param name="bodyMass">Mass of core particles (top, bottom)</param>
    /// <param name="footMass">Mass of feet particles</param>
    /// <param name="engineMass">Mass of engine particle</param>
    /// <param name="particleRadius">Radius of particles</param>
    /// <returns>Particle indices: [top, bottom, leftFoot, rightFoot, engine]</returns>
    public static int[] CreateRocket(
        WorldState world,
        float centerX = 0f,
        float centerY = 0f,
        float bodyHeight = 2f,
        float legSpread = 1.5f,
        float bodyMass = 5f,
        float footMass = 1f,
        float engineMass = 2f,
        float particleRadius = 0.15f)
    {
        // Create particles
        int top = world.AddParticle(
            x: centerX,
            y: centerY + bodyHeight,
            vx: 0f, vy: 0f,
            mass: bodyMass,
            radius: particleRadius
        );

        int bottom = world.AddParticle(
            x: centerX,
            y: centerY,
            vx: 0f, vy: 0f,
            mass: bodyMass,
            radius: particleRadius
        );

        float legY = centerY - bodyHeight * 0.3f;
        float legHalfSpread = legSpread * 0.5f;

        int leftFoot = world.AddParticle(
            x: centerX - legHalfSpread,
            y: legY,
            vx: 0f, vy: 0f,
            mass: footMass,
            radius: particleRadius * 1.2f // Slightly larger feet
        );

        int rightFoot = world.AddParticle(
            x: centerX + legHalfSpread,
            y: legY,
            vx: 0f, vy: 0f,
            mass: footMass,
            radius: particleRadius * 1.2f
        );

        // Engine particle below bottom
        int engine = world.AddParticle(
            x: centerX,
            y: centerY - bodyHeight * 0.2f,
            vx: 0f, vy: 0f,
            mass: engineMass,
            radius: particleRadius * 0.8f
        );

        // Add rods to form structure (minimal set to avoid over-constraint)
        // Main body
        world.Rods.Add(CreateRod(world, top, bottom));

        // Left leg (only connect to bottom to avoid over-constraint)
        world.Rods.Add(CreateRod(world, bottom, leftFoot));

        // Right leg (only connect to bottom to avoid over-constraint)
        world.Rods.Add(CreateRod(world, bottom, rightFoot));

        // Engine mounting (only to bottom to avoid over-constraint)
        world.Rods.Add(CreateRod(world, bottom, engine));

        // DISABLED: Angle constraints cause instability - the rocket explodes
        // The 4-rod structure should be sufficient to maintain shape
        // float leftLegAngle = ComputeAngle(world, leftFoot, bottom, top);
        // world.Angles.Add(new Angle(leftFoot, bottom, top, leftLegAngle, compliance: 1e-7f));

        // float rightLegAngle = ComputeAngle(world, rightFoot, bottom, top);
        // world.Angles.Add(new Angle(rightFoot, bottom, top, rightLegAngle, compliance: 1e-7f));

        // Add motor for gimbal control (engine angle relative to body)
        // Motor controls angle: engine-top-bottom
        float gimbalAngle = ComputeAngle(world, engine, top, bottom);
        world.Motors.Add(new MotorAngle(engine, top, bottom, target: gimbalAngle, compliance: 1e-6f));

        return new int[] { top, bottom, leftFoot, rightFoot, engine };
    }

    /// <summary>
    /// Creates a rocket with initial velocity (e.g., for testing landing from horizontal motion).
    /// </summary>
    public static int[] CreateRocketWithVelocity(
        WorldState world,
        float centerX, float centerY,
        float velX, float velY,
        float bodyHeight = 2f,
        float legSpread = 1.5f)
    {
        var indices = CreateRocket(world, centerX, centerY, bodyHeight, legSpread);

        // Apply velocity to all particles
        foreach (int idx in indices)
        {
            world.VelX[idx] = velX;
            world.VelY[idx] = velY;
        }

        return indices;
    }

    /// <summary>
    /// Helper: creates a rod constraint between two particles with current distance as rest length.
    /// </summary>
    private static Rod CreateRod(WorldState world, int i, int j)
    {
        float dx = world.PosX[i] - world.PosX[j];
        float dy = world.PosY[i] - world.PosY[j];
        float restLength = MathF.Sqrt(dx * dx + dy * dy);
        // Reduced number of rods, so can use stiffer constraints
        return new Rod(i, j, restLength, compliance: 1e-8f);
    }

    /// <summary>
    /// Helper: computes current angle i-j-k at vertex j.
    /// </summary>
    private static float ComputeAngle(WorldState world, int i, int j, int k)
    {
        float e1x = world.PosX[i] - world.PosX[j];
        float e1y = world.PosY[i] - world.PosY[j];
        float e2x = world.PosX[k] - world.PosX[j];
        float e2y = world.PosY[k] - world.PosY[j];

        return Math2D.AngleBetween(e1x, e1y, e2x, e2y);
    }

    /// <summary>
    /// Applies thrust force to the engine particle in the upward direction
    /// (relative to rocket body orientation).
    /// </summary>
    /// <param name="world">World state</param>
    /// <param name="rocketIndices">Indices from CreateRocket</param>
    /// <param name="throttle">Throttle value (0-1)</param>
    /// <param name="maxThrust">Maximum thrust force in Newtons</param>
    public static void ApplyThrust(
        WorldState world,
        int[] rocketIndices,
        float throttle,
        float maxThrust = 100f)
    {
        int top = rocketIndices[0];
        int bottom = rocketIndices[1];
        int engine = rocketIndices[4];

        // Compute rocket body direction (top - bottom)
        float dx = world.PosX[top] - world.PosX[bottom];
        float dy = world.PosY[top] - world.PosY[bottom];
        float len = MathF.Sqrt(dx * dx + dy * dy);

        if (len < 1e-6f)
            return;

        // Normalize to get thrust direction
        float thrustDirX = dx / len;
        float thrustDirY = dy / len;

        // Apply force to engine particle
        float thrust = throttle * maxThrust;
        world.ForceX[engine] += thrust * thrustDirX;
        world.ForceY[engine] += thrust * thrustDirY;
    }

    /// <summary>
    /// Sets the gimbal angle for the rocket motor.
    /// </summary>
    /// <param name="world">World state</param>
    /// <param name="rocketIndices">Indices from CreateRocket</param>
    /// <param name="gimbalAngleDeg">Gimbal angle in degrees (e.g., -15 to +15)</param>
    public static void SetGimbal(WorldState world, int[] rocketIndices, float gimbalAngleDeg)
    {
        // Motor is the last motor in the list (assuming we just created this rocket)
        // In a real system, you'd track which motor belongs to which rocket
        int motorIdx = world.Motors.Count - 1;
        if (motorIdx < 0)
            return;

        var motor = world.Motors[motorIdx];

        // Compute current body angle
        int top = rocketIndices[0];
        int bottom = rocketIndices[1];
        int engine = rocketIndices[4];

        float bodyAngle = ComputeAngle(world, top, bottom, engine);

        // Set target angle relative to body
        float gimbalRad = gimbalAngleDeg * MathF.PI / 180f;
        motor.Target = bodyAngle + gimbalRad;

        world.Motors[motorIdx] = motor;
    }

    /// <summary>
    /// Computes the center of mass of the rocket.
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
            float mass = world.InvMass[idx] > 0f ? 1f / world.InvMass[idx] : 0f;
            totalMass += mass;
            comX += world.PosX[idx] * mass;
            comY += world.PosY[idx] * mass;
        }

        if (totalMass > 0f)
        {
            comX /= totalMass;
            comY /= totalMass;
        }
    }

    /// <summary>
    /// Computes the average velocity of the rocket.
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
            float mass = world.InvMass[idx] > 0f ? 1f / world.InvMass[idx] : 0f;
            totalMass += mass;
            velX += world.VelX[idx] * mass;
            velY += world.VelY[idx] * mass;
        }

        if (totalMass > 0f)
        {
            velX /= totalMass;
            velY /= totalMass;
        }
    }

    /// <summary>
    /// Computes the rocket's upright vector (normalized top-bottom direction).
    /// </summary>
    public static void GetUpVector(
        WorldState world,
        int[] rocketIndices,
        out float ux, out float uy)
    {
        int top = rocketIndices[0];
        int bottom = rocketIndices[1];

        ux = world.PosX[top] - world.PosX[bottom];
        uy = world.PosY[top] - world.PosY[bottom];

        Math2D.Normalize(ref ux, ref uy);
    }
}
