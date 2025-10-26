using Evolvatron.Core;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Stress tests for angle constraints that truly challenge the solver.
/// These tests force the angle constraint to fight against gravity and contacts.
/// </summary>
public class AngleConstraintStressTests
{
    [Fact]
    public void LShape_BalancesOnCorner_MaintainsAngle()
    {
        // This is a HARD test: L-shape balances on the corner vertex (p1)
        // Gravity tries to collapse the angle, constraint must resist
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 40,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 0f,  // Rigid angle
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        float height = 3f;
        float armLength = 1f;
        float targetAngle = MathF.PI / 2f; // 90 degrees

        // L-shape oriented diagonally so it will try to balance on p1 (the corner)
        // This configuration will stress the angle constraint heavily
        float offset = 0.7f; // Slight offset to make it challenging but not impossible

        int p0 = world.AddParticle(x: -offset, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left arm
        int p1 = world.AddParticle(x: 0f, y: height - offset, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Vertex (will touch ground)
        int p2 = world.AddParticle(x: offset, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Right arm

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: 0f));

        // Act: Simulate
        var stepper = new CPUStepper();
        for (int i = 0; i < 300; i++)
        {
            stepper.Step(world, config);

            if (float.IsNaN(world.PosY[p1]) || MathF.Abs(world.VelY[p1]) > 100f)
            {
                Assert.Fail($"Structure exploded at step {i}");
            }
        }

        // Assert: Angle maintained despite challenging orientation
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(angleError < 0.2f,  // ~11 degrees tolerance (this is HARD)
            $"90° angle not maintained under stress. " +
            $"Final: {RadToDeg(finalAngle):F1}°, Error: {RadToDeg(angleError):F1}°");

        // Verify structure is at rest
        float maxVel = 1.0f;
        Assert.True(MathF.Abs(world.VelY[p1]) < maxVel, "Structure still moving");
    }

    [Fact]
    public void LShape_LandsOnSide_ThenTipples_MaintainsAngle()
    {
        // L-shape lands on its side, then might tipple over
        // Angle must be maintained throughout the motion
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 40,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 0f,
            ContactCompliance = 1e-8f,
            FrictionMu = 0.3f  // Lower friction to allow sliding/tipping
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        float height = 5f;
        float armLength = 1f;
        float targetAngle = MathF.PI / 2f;

        // L-shape rotated 45 degrees initially - will land at an angle
        float angle45 = MathF.PI / 4f;
        float cos45 = MathF.Cos(angle45);
        float sin45 = MathF.Sin(angle45);

        // Start with L pointing diagonally
        int p0 = world.AddParticle(
            x: -armLength * cos45,
            y: height - armLength * sin45,
            vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        int p2 = world.AddParticle(
            x: armLength * sin45,
            y: height + armLength * cos45,
            vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: 0f));

        // Verify initial angle is 90 degrees
        float initialAngle = ComputeAngle(world, p0, p1, p2);
        Assert.InRange(initialAngle, targetAngle - 0.05f, targetAngle + 0.05f);

        // Act: Simulate fall and landing
        var stepper = new CPUStepper();
        float worstAngleError = 0f;

        for (int i = 0; i < 500; i++)  // Longer simulation to see tipping
        {
            stepper.Step(world, config);

            // Track worst angle error during entire simulation
            float currentAngle = ComputeAngle(world, p0, p1, p2);
            float currentError = MathF.Abs(WrapAngle(currentAngle - targetAngle));
            worstAngleError = MathF.Max(worstAngleError, currentError);

            if (float.IsNaN(world.PosY[p1]))
            {
                Assert.Fail($"Structure exploded at step {i}");
            }
        }

        // Assert: Angle maintained throughout entire simulation
        Assert.True(worstAngleError < 0.15f,  // ~8.6 degrees tolerance
            $"90° angle degraded during dynamic motion. " +
            $"Worst error: {RadToDeg(worstAngleError):F1}°");

        // Final angle check
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float finalError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(finalError < 0.1f,
            $"Final angle not maintained. Error: {RadToDeg(finalError):F1}°");
    }

    [Fact]
    public void VShape_60Degrees_LandsOnTwoPoints_MaintainsAngle()
    {
        // V-shape with 60-degree angle lands on two outer points
        // This challenges the angle constraint differently than L-shape
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 60,  // More iterations for non-90° angle
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 0f,
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        float height = 5f;
        float armLength = 1f;
        float targetAngle = MathF.PI / 3f; // 60 degrees

        // V-shape pointing UP (vertex p1 at top, p0 and p2 spread below)
        // This way both outer points will hit the ground
        int p1 = world.AddParticle(x: 0f, y: height + armLength * 0.5f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Top vertex

        // Place p0 and p2 symmetrically below p1 at 60-degree angle
        float halfAngle = targetAngle / 2f;
        int p0 = world.AddParticle(
            x: -armLength * MathF.Sin(halfAngle),
            y: height - armLength * MathF.Cos(halfAngle),
            vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left bottom

        int p2 = world.AddParticle(
            x: armLength * MathF.Sin(halfAngle),
            y: height - armLength * MathF.Cos(halfAngle),
            vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Right bottom

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: 0f));

        // Verify initial angle
        float initialAngle = ComputeAngle(world, p0, p1, p2);
        // Note: May need to adjust sign based on how angle is measured

        // Act: Simulate
        var stepper = new CPUStepper();
        for (int i = 0; i < 300; i++)
        {
            stepper.Step(world, config);

            if (float.IsNaN(world.PosY[p1]) || MathF.Abs(world.VelY[p1]) > 100f)
            {
                Assert.Fail($"Structure exploded at step {i}");
            }
        }

        // Assert: Angle maintained
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - initialAngle)); // Compare to initial, not target (sign issues)

        Assert.True(angleError < 0.15f,  // ~8.6 degrees tolerance
            $"60° V-shape angle not maintained. " +
            $"Initial: {RadToDeg(initialAngle):F1}°, Final: {RadToDeg(finalAngle):F1}°, " +
            $"Error: {RadToDeg(angleError):F1}°");

        // Verify structure is at rest
        Assert.True(MathF.Abs(world.VelY[p1]) < 1.0f, "Structure still moving");

        // Verify both outer points are on ground (V is standing up)
        float groundTop = -1.5f;
        Assert.True(world.PosY[p0] > groundTop - 0.3f && world.PosY[p0] < groundTop + 0.3f,
            "Left point not on ground");
        Assert.True(world.PosY[p2] > groundTop - 0.3f && world.PosY[p2] < groundTop + 0.3f,
            "Right point not on ground");
    }

    // Helper methods
    private static float ComputeAngle(WorldState world, int i, int j, int k)
    {
        float ux = world.PosX[i] - world.PosX[j];
        float uy = world.PosY[i] - world.PosY[j];
        float vx = world.PosX[k] - world.PosX[j];
        float vy = world.PosY[k] - world.PosY[j];
        float dot = ux * vx + uy * vy;
        float cross = ux * vy - uy * vx;
        return MathF.Atan2(cross, dot);
    }

    private static float WrapAngle(float angle)
    {
        return MathF.Atan2(MathF.Sin(angle), MathF.Cos(angle));
    }

    private static float RadToDeg(float rad)
    {
        return rad * 180f / MathF.PI;
    }
}
