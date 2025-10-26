using Evolvatron.Core;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Minimal test to verify the corrected angle gradient formula works.
/// Tests a simple static configuration without gravity or contacts.
/// </summary>
public class AngleGradientVerificationTest
{
    [Fact]
    public void AngleConstraint_ConvergesToTarget_WithoutContacts()
    {
        // Arrange: 3 particles, 2 rods, 1 angle constraint
        // Start at 90 degrees, target 60 degrees
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 20,
            Substeps = 1,
            GravityX = 0f,
            GravityY = 0f,  // No gravity for this test
            RodCompliance = 0f,
            AngleCompliance = 0f,  // Rigid angle
            ContactCompliance = 1e-8f
        };

        float armLength = 1f;
        float targetAngle = MathF.PI / 3f; // 60 degrees

        // Start at 90 degrees (L-shape)
        int p1 = world.AddParticle(x: 0f, y: 0f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Center (vertex)
        int p0 = world.AddParticle(x: -armLength, y: 0f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left
        int p2 = world.AddParticle(x: 0f, y: armLength, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Up

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint: target 60 degrees
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: 0f));

        // Measure initial angle (should be -90 degrees = -π/2, since we go clockwise from p0 to p2)
        float initialAngle = ComputeAngle(world, p0, p1, p2);
        Assert.InRange(initialAngle, -MathF.PI / 2f - 0.01f, -MathF.PI / 2f + 0.01f);

        // Act: Run simulation (no gravity, no contacts)
        var stepper = new CPUStepper();
        for (int i = 0; i < 100; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Angle should converge to target (60 degrees)
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(angleError < 0.05f,
            $"Angle did not converge. Target: {RadToDeg(targetAngle):F1}°, " +
            $"Final: {RadToDeg(finalAngle):F1}°, Error: {RadToDeg(angleError):F1}°");

        // Also verify rod lengths preserved
        float len01 = Distance(world, p0, p1);
        float len12 = Distance(world, p1, p2);
        Assert.InRange(len01, armLength - 0.01f, armLength + 0.01f);
        Assert.InRange(len12, armLength - 0.01f, armLength + 0.01f);
    }

    [Fact]
    public void AngleConstraint_90Degrees_StableWithGravity()
    {
        // Arrange: L-shape that should maintain 90 degrees while falling
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 20,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 0f,  // Rigid angle
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        float armLength = 1f;
        float targetAngle = MathF.PI / 2f; // 90 degrees
        float height = 5f;

        // L-shape
        int p0 = world.AddParticle(x: -armLength, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left
        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);          // Center
        int p2 = world.AddParticle(x: 0f, y: height + armLength, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Up

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: 0f));

        // Act: Simulate falling and landing
        var stepper = new CPUStepper();
        for (int i = 0; i < 300; i++)
        {
            stepper.Step(world, config);

            // Safety check
            if (float.IsNaN(world.PosY[p1]) || MathF.Abs(world.VelY[p1]) > 100f)
            {
                Assert.Fail($"Structure exploded at step {i}");
            }
        }

        // Assert: Angle maintained
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(angleError < 0.1f,  // ~6 degrees tolerance
            $"90° angle not maintained. Final: {RadToDeg(finalAngle):F1}°, Error: {RadToDeg(angleError):F1}°");

        // Verify at rest
        Assert.True(MathF.Abs(world.VelY[p1]) < 1.0f, "Structure still moving");

        // Verify didn't fall through floor
        float groundTop = -1.5f;
        Assert.True(world.PosY[p0] > groundTop - 0.3f);
        Assert.True(world.PosY[p1] > groundTop - 0.3f);
        Assert.True(world.PosY[p2] > groundTop - 0.3f);
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

    private static float Distance(WorldState world, int i, int j)
    {
        float dx = world.PosX[i] - world.PosX[j];
        float dy = world.PosY[i] - world.PosY[j];
        return MathF.Sqrt(dx * dx + dy * dy);
    }
}
