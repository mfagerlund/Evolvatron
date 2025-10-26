using Evolvatron.Core;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Tests for angle constraint behavior when structures fall and rest on the ground.
/// These tests verify that angle constraints maintain the target angle during physics simulation.
/// </summary>
public class AngleConstraintDropTests
{
    private const float AngleTolerance = 0.05f; // ~3 degrees in radians

    [Fact]
    public void TwoSticks_ConnectedByAngle_MaintainAngleWhenRestingOnFloor()
    {
        // Arrange: Create two sticks (3 particles) connected by an angle constraint
        // This mimics the JavaScript demo with two rods and an angle constraint
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,  // Match the JS demo timestep
            XpbdIterations = 40,  // Higher iterations for stiff angle constraints
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,  // Rigid rods
            AngleCompliance = 2e-6f,  // Very small compliance (nearly rigid angle)
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f
        };

        // Ground (higher than usual to catch the structure)
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        // Three particles forming a V-shape (like two sticks meeting at a point)
        // p1 is the hinge/joint in the middle
        float height = 5f;
        float armLength = 1f;
        float targetAngle = MathF.PI / 3f; // 60 degrees

        // Calculate initial positions for the target angle
        // p0 is to the left of p1
        // p2 is to the right and up from p1
        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Middle (vertex)
        int p0 = world.AddParticle(x: -armLength, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left
        int p2 = world.AddParticle(
            x: armLength * MathF.Cos(targetAngle),
            y: height + armLength * MathF.Sin(targetAngle),
            vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Upper right

        // Connect with rigid rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Add angle constraint to maintain the angle at p1
        world.Angles.Add(new Angle(
            i: p0,
            j: p1,  // vertex
            k: p2,
            theta0: targetAngle,
            compliance: config.AngleCompliance));

        // Measure initial angle
        float initialAngle = ComputeAngle(world, p0, p1, p2);
        Assert.InRange(initialAngle, targetAngle - 0.01f, targetAngle + 0.01f); // Should start at target

        // Act: Simulate for 5 seconds (300 steps at 60 Hz)
        var stepper = new CPUStepper();
        for (int i = 0; i < 300; i++)
        {
            stepper.Step(world, config);

            // Early termination if structure explodes (safety check)
            if (float.IsNaN(world.PosY[p1]) || MathF.Abs(world.VelY[p1]) > 100f)
            {
                Assert.Fail("Structure became unstable or exploded during simulation");
            }
        }

        // Assert: Structure should have settled on ground
        // 1. All particles should be above ground level
        float groundTop = -1.5f; // -2.0 (ground center) + 0.5 (half height)
        float minY = groundTop - 0.05f - 0.2f; // ground top - radius - tolerance
        Assert.True(world.PosY[p0] > minY, $"Particle 0 fell through floor: y={world.PosY[p0]}");
        Assert.True(world.PosY[p1] > minY, $"Particle 1 fell through floor: y={world.PosY[p1]}");
        Assert.True(world.PosY[p2] > minY, $"Particle 2 fell through floor: y={world.PosY[p2]}");

        // 2. Structure should be at rest (low velocities)
        float maxVel = 1.0f; // Should be nearly stationary
        Assert.True(MathF.Abs(world.VelX[p0]) < maxVel, $"Particle 0 still moving: vx={world.VelX[p0]}");
        Assert.True(MathF.Abs(world.VelY[p0]) < maxVel, $"Particle 0 still moving: vy={world.VelY[p0]}");
        Assert.True(MathF.Abs(world.VelX[p1]) < maxVel, $"Particle 1 still moving: vx={world.VelX[p1]}");
        Assert.True(MathF.Abs(world.VelY[p1]) < maxVel, $"Particle 1 still moving: vy={world.VelY[p1]}");
        Assert.True(MathF.Abs(world.VelX[p2]) < maxVel, $"Particle 2 still moving: vx={world.VelX[p2]}");
        Assert.True(MathF.Abs(world.VelY[p2]) < maxVel, $"Particle 2 still moving: vy={world.VelY[p2]}");

        // 3. Rod lengths should be preserved (didn't collapse or stretch)
        float finalLen01 = Distance(world, p0, p1);
        float finalLen12 = Distance(world, p1, p2);
        float lengthTolerance = 0.05f; // 5%
        Assert.InRange(finalLen01, armLength - lengthTolerance, armLength + lengthTolerance);
        Assert.InRange(finalLen12, armLength - lengthTolerance, armLength + lengthTolerance);

        // 4. CRITICAL: Angle should be maintained
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(angleError < AngleTolerance,
            $"Angle constraint failed to maintain target angle. " +
            $"Target: {targetAngle:F4} rad ({RadToDeg(targetAngle):F1}°), " +
            $"Final: {finalAngle:F4} rad ({RadToDeg(finalAngle):F1}°), " +
            $"Error: {angleError:F4} rad ({RadToDeg(angleError):F1}°)");

        // 5. No NaN positions
        Assert.False(float.IsNaN(world.PosX[p0]) || float.IsNaN(world.PosY[p0]), "Particle 0 has NaN position");
        Assert.False(float.IsNaN(world.PosX[p1]) || float.IsNaN(world.PosY[p1]), "Particle 1 has NaN position");
        Assert.False(float.IsNaN(world.PosX[p2]) || float.IsNaN(world.PosY[p2]), "Particle 2 has NaN position");
    }

    [Fact]
    public void RightAngleSticks_MaintainNinetyDegrees_WhenResting()
    {
        // Arrange: Test with a simple 90-degree angle (L-shape)
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 40,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 2e-6f,
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        // L-shape with 90-degree angle
        float height = 5f;
        float armLength = 1f;
        float targetAngle = MathF.PI / 2f; // 90 degrees

        int p0 = world.AddParticle(x: -armLength, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left
        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);          // Center
        int p2 = world.AddParticle(x: 0f, y: height + armLength, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Top

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: config.AngleCompliance));

        // Act: Simulate
        var stepper = new CPUStepper();
        for (int i = 0; i < 300; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Angle maintained
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(angleError < AngleTolerance,
            $"90-degree angle not maintained. " +
            $"Target: 90°, Final: {RadToDeg(finalAngle):F1}°, Error: {RadToDeg(angleError):F1}°");
    }

    [Fact]
    public void ObtuseAngleSticks_Maintain120Degrees_WhenResting()
    {
        // Arrange: Test with an obtuse angle (120 degrees)
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 40,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 2e-6f,
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -2f, hx: 20f, hy: 0.5f));

        // V-shape with 120-degree angle
        float height = 5f;
        float armLength = 1f;
        float targetAngle = 2f * MathF.PI / 3f; // 120 degrees

        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Center
        int p0 = world.AddParticle(x: -armLength, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left
        int p2 = world.AddParticle(
            x: armLength * MathF.Cos(targetAngle),
            y: height + armLength * MathF.Sin(targetAngle),
            vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        // Rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Angle constraint
        world.Angles.Add(new Angle(p0, p1, p2, theta0: targetAngle, compliance: config.AngleCompliance));

        // Act: Simulate
        var stepper = new CPUStepper();
        for (int i = 0; i < 300; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Angle maintained
        float finalAngle = ComputeAngle(world, p0, p1, p2);
        float angleError = MathF.Abs(WrapAngle(finalAngle - targetAngle));

        Assert.True(angleError < AngleTolerance,
            $"120-degree angle not maintained. " +
            $"Target: 120°, Final: {RadToDeg(finalAngle):F1}°, Error: {RadToDeg(angleError):F1}°");
    }

    // Helper methods

    /// <summary>
    /// Computes the signed angle at vertex j between edges (i-j) and (k-j).
    /// Returns angle in radians in range [-π, π].
    /// </summary>
    private static float ComputeAngle(WorldState world, int i, int j, int k)
    {
        // Vector from j to i
        float ux = world.PosX[i] - world.PosX[j];
        float uy = world.PosY[i] - world.PosY[j];

        // Vector from j to k
        float vx = world.PosX[k] - world.PosX[j];
        float vy = world.PosY[k] - world.PosY[j];

        // Compute angle using atan2(cross, dot)
        float dot = ux * vx + uy * vy;
        float cross = ux * vy - uy * vx;

        return MathF.Atan2(cross, dot);
    }

    /// <summary>
    /// Wraps angle to range [-π, π].
    /// </summary>
    private static float WrapAngle(float angle)
    {
        return MathF.Atan2(MathF.Sin(angle), MathF.Cos(angle));
    }

    /// <summary>
    /// Converts radians to degrees.
    /// </summary>
    private static float RadToDeg(float rad)
    {
        return rad * 180f / MathF.PI;
    }

    /// <summary>
    /// Computes distance between two particles.
    /// </summary>
    private static float Distance(WorldState world, int i, int j)
    {
        float dx = world.PosX[i] - world.PosX[j];
        float dy = world.PosY[i] - world.PosY[j];
        return MathF.Sqrt(dx * dx + dy * dy);
    }
}
