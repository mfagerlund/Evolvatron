using Evolvatron.Core;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Tests for XPBD particle constraint stability.
/// Verifies that particle structures don't explode, collapse, or fall through floors.
/// </summary>
public class XPBDStabilityTests
{
    private const float Tolerance = 1e-3f;

    [Fact]
    public void TriangleParticles_FallsToGround_DoesNotExplodeOrCollapse()
    {
        // Arrange: Create a triangle of particles
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f, // Rigid rods
            ContactCompliance = 1e-8f
        };

        // Ground (OBB at y=-1)
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -1f, hx: 20f, hy: 0.5f));

        // Triangle: 3 particles at height y=2, forming equilateral triangle
        float height = 2f;
        float side = 1f;
        float h = side * MathF.Sqrt(3f) / 2f; // Height of equilateral triangle

        int p0 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);
        int p1 = world.AddParticle(x: -side / 2f, y: height - h, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);
        int p2 = world.AddParticle(x: side / 2f, y: height - h, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        // Connect with rods
        world.Rods.Add(new Rod(p0, p1, restLength: side, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: side, compliance: 0f));
        world.Rods.Add(new Rod(p2, p0, restLength: side, compliance: 0f));

        // Record initial edge lengths
        float initialEdge01 = Distance(world, p0, p1);
        float initialEdge12 = Distance(world, p1, p2);
        float initialEdge20 = Distance(world, p2, p0);

        // Act: Simulate for 2 seconds (480 steps)
        var stepper = new CPUStepper();
        for (int i = 0; i < 480; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Triangle should have settled on ground
        // 1. All particles should be above ground level (y > -0.5 - radius)
        float minY = -0.5f - 0.05f - 0.1f; // ground top - radius - tolerance
        Assert.True(world.PosY[p0] > minY, $"Particle 0 fell through floor: y={world.PosY[p0]}");
        Assert.True(world.PosY[p1] > minY, $"Particle 1 fell through floor: y={world.PosY[p1]}");
        Assert.True(world.PosY[p2] > minY, $"Particle 2 fell through floor: y={world.PosY[p2]}");

        // 2. Triangle should be at rest (low velocities)
        float maxVel = 0.5f; // Should be nearly stationary
        Assert.True(MathF.Abs(world.VelX[p0]) < maxVel, $"Particle 0 still moving: vx={world.VelX[p0]}");
        Assert.True(MathF.Abs(world.VelY[p0]) < maxVel, $"Particle 0 still moving: vy={world.VelY[p0]}");
        Assert.True(MathF.Abs(world.VelX[p1]) < maxVel, $"Particle 1 still moving: vx={world.VelX[p1]}");
        Assert.True(MathF.Abs(world.VelY[p1]) < maxVel, $"Particle 1 still moving: vy={world.VelY[p1]}");
        Assert.True(MathF.Abs(world.VelX[p2]) < maxVel, $"Particle 2 still moving: vx={world.VelX[p2]}");
        Assert.True(MathF.Abs(world.VelY[p2]) < maxVel, $"Particle 2 still moving: vy={world.VelY[p2]}");

        // 3. Edge lengths should be preserved (didn't collapse or explode)
        float finalEdge01 = Distance(world, p0, p1);
        float finalEdge12 = Distance(world, p1, p2);
        float finalEdge20 = Distance(world, p2, p0);

        float edgeTolerance = 0.05f; // 5% variation allowed
        Assert.InRange(finalEdge01, initialEdge01 - edgeTolerance, initialEdge01 + edgeTolerance);
        Assert.InRange(finalEdge12, initialEdge12 - edgeTolerance, initialEdge12 + edgeTolerance);
        Assert.InRange(finalEdge20, initialEdge20 - edgeTolerance, initialEdge20 + edgeTolerance);

        // 4. No NaN positions
        Assert.False(float.IsNaN(world.PosX[p0]) || float.IsNaN(world.PosY[p0]), "Particle 0 has NaN position");
        Assert.False(float.IsNaN(world.PosX[p1]) || float.IsNaN(world.PosY[p1]), "Particle 1 has NaN position");
        Assert.False(float.IsNaN(world.PosX[p2]) || float.IsNaN(world.PosY[p2]), "Particle 2 has NaN position");
    }

    [Fact]
    public void SquareWithDiagonal_FallsToGround_RemainsStable()
    {
        // Arrange: Create a square with diagonal for rigidity
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            ContactCompliance = 1e-8f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -1f, hx: 20f, hy: 0.5f));

        // Square: 4 particles at height y=2
        float height = 2f;
        float size = 1f;

        int p0 = world.AddParticle(x: -size / 2f, y: height + size / 2f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Top-left
        int p1 = world.AddParticle(x: size / 2f, y: height + size / 2f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);  // Top-right
        int p2 = world.AddParticle(x: size / 2f, y: height - size / 2f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);  // Bottom-right
        int p3 = world.AddParticle(x: -size / 2f, y: height - size / 2f, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Bottom-left

        // Connect perimeter
        world.Rods.Add(new Rod(p0, p1, restLength: size, compliance: 0f)); // Top
        world.Rods.Add(new Rod(p1, p2, restLength: size, compliance: 0f)); // Right
        world.Rods.Add(new Rod(p2, p3, restLength: size, compliance: 0f)); // Bottom
        world.Rods.Add(new Rod(p3, p0, restLength: size, compliance: 0f)); // Left

        // Diagonal for rigidity
        float diagonal = size * MathF.Sqrt(2f);
        world.Rods.Add(new Rod(p0, p2, restLength: diagonal, compliance: 0f));

        // Record initial edge lengths
        float initialSide = size;
        float initialDiagonal = diagonal;

        // Act: Simulate for 2 seconds (480 steps)
        var stepper = new CPUStepper();
        for (int i = 0; i < 480; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Square should be stable
        // 1. No particles fell through floor
        float minY = -0.5f - 0.05f - 0.1f;
        Assert.True(world.PosY[p0] > minY, $"Particle 0 fell through floor: y={world.PosY[p0]}");
        Assert.True(world.PosY[p1] > minY, $"Particle 1 fell through floor: y={world.PosY[p1]}");
        Assert.True(world.PosY[p2] > minY, $"Particle 2 fell through floor: y={world.PosY[p2]}");
        Assert.True(world.PosY[p3] > minY, $"Particle 3 fell through floor: y={world.PosY[p3]}");

        // 2. Should be at rest
        float maxVel = 0.5f;
        for (int i = 0; i < 4; i++)
        {
            Assert.True(MathF.Abs(world.VelX[i]) < maxVel, $"Particle {i} still moving: vx={world.VelX[i]}");
            Assert.True(MathF.Abs(world.VelY[i]) < maxVel, $"Particle {i} still moving: vy={world.VelY[i]}");
        }

        // 3. Edge lengths preserved (verify a few key edges)
        float finalSideTop = Distance(world, p0, p1);
        float finalSideBottom = Distance(world, p2, p3);
        float finalDiagonal = Distance(world, p0, p2);

        float edgeTolerance = 0.05f;
        Assert.InRange(finalSideTop, initialSide - edgeTolerance, initialSide + edgeTolerance);
        Assert.InRange(finalSideBottom, initialSide - edgeTolerance, initialSide + edgeTolerance);
        Assert.InRange(finalDiagonal, initialDiagonal - edgeTolerance, initialDiagonal + edgeTolerance);

        // 4. No NaN positions
        for (int i = 0; i < 4; i++)
        {
            Assert.False(float.IsNaN(world.PosX[i]) || float.IsNaN(world.PosY[i]), $"Particle {i} has NaN position");
        }

        // 5. Square didn't collapse (check area is roughly preserved)
        float initialArea = size * size;
        float finalArea = ComputeQuadArea(world, p0, p1, p2, p3);
        Assert.InRange(finalArea, initialArea * 0.8f, initialArea * 1.2f); // Allow 20% variation
    }

    [Fact]
    public void ThreeParticleLShape_FallsToGround_RemainsStable()
    {
        // Arrange: Create 3 particles forming an "L" shape
        // NOTE: We use a diagonal rod instead of an angle constraint because
        // angle constraints cause instability in XPBDwhen combined with contact constraints.
        // This is the approach used by Ten Minute Physics for bending constraints.
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            ContactCompliance = 1e-8f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -1f, hx: 20f, hy: 0.5f));

        // Three particles forming an "L" shape (90-degree angle)
        float height = 2f;
        float armLength = 0.5f;

        int p0 = world.AddParticle(x: -armLength, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Left
        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);          // Center (hinge)
        int p2 = world.AddParticle(x: 0f, y: height + armLength, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f); // Top

        // Connect with rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Add diagonal rod to maintain the "L" shape (bending constraint)
        float diagonal = MathF.Sqrt(armLength * armLength + armLength * armLength); // Pythagorean theorem
        world.Rods.Add(new Rod(p0, p2, restLength: diagonal, compliance: 0f));

        // Act: Simulate for 2 seconds (480 steps)
        var stepper = new CPUStepper();
        for (int i = 0; i < 480; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Hinge should maintain angle and settle
        // 1. No particles fell through floor
        float minY = -0.5f - 0.05f - 0.1f;
        Assert.True(world.PosY[p0] > minY, $"Particle 0 fell through floor: y={world.PosY[p0]}");
        Assert.True(world.PosY[p1] > minY, $"Particle 1 fell through floor: y={world.PosY[p1]}");
        Assert.True(world.PosY[p2] > minY, $"Particle 2 fell through floor: y={world.PosY[p2]}");

        // 2. Should be at rest
        float maxVel = 0.5f;
        for (int i = 0; i < 3; i++)
        {
            Assert.True(MathF.Abs(world.VelX[i]) < maxVel, $"Particle {i} still moving: vx={world.VelX[i]}");
            Assert.True(MathF.Abs(world.VelY[i]) < maxVel, $"Particle {i} still moving: vy={world.VelY[i]}");
        }

        // 3. Edge lengths preserved
        float finalEdge01 = Distance(world, p0, p1);
        float finalEdge12 = Distance(world, p1, p2);

        float edgeTolerance = 0.05f;
        Assert.InRange(finalEdge01, armLength - edgeTolerance, armLength + edgeTolerance);
        Assert.InRange(finalEdge12, armLength - edgeTolerance, armLength + edgeTolerance);

        // 4. Diagonal length preserved (maintains the "L" shape)
        float finalDiagonal = Distance(world, p0, p2);
        Assert.InRange(finalDiagonal, diagonal - edgeTolerance, diagonal + edgeTolerance);

        // 5. No NaN positions
        for (int i = 0; i < 3; i++)
        {
            Assert.False(float.IsNaN(world.PosX[i]) || float.IsNaN(world.PosY[i]), $"Particle {i} has NaN position");
        }

        // 6. Structure didn't explode (all particles within reasonable bounds)
        for (int i = 0; i < 3; i++)
        {
            Assert.True(MathF.Abs(world.PosX[i]) < 10f, $"Particle {i} exploded in X: x={world.PosX[i]}");
            Assert.True(world.PosY[i] < 10f, $"Particle {i} exploded in Y: y={world.PosY[i]}");
        }
    }

    [Fact]
    public void AngleConstraintAsRod_API_WorksCorrectly()
    {
        // Arrange: Test the convenient angle constraint API
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            ContactCompliance = 1e-8f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -1f, hx: 20f, hy: 0.5f));

        // Create L-shape using the new API
        float height = 2f;
        float armLength = 0.5f;

        int p0 = world.AddParticle(x: -armLength, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);
        int p1 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);
        int p2 = world.AddParticle(x: 0f, y: height + armLength, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        // Connect with rods
        world.Rods.Add(new Rod(p0, p1, restLength: armLength, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: armLength, compliance: 0f));

        // Use the convenient API to add angle constraint (as a diagonal rod)
        int angleRodIdx = world.AddAngleConstraintAsRod(
            i: p0,
            j: p1,  // vertex
            k: p2,
            targetAngle: MathF.PI / 2f,  // 90 degrees
            len1: armLength,
            len2: armLength,
            compliance: 0f);

        // Verify the rod was created correctly
        Assert.True(angleRodIdx >= 0);
        Assert.Equal(3, world.Rods.Count); // 2 edge rods + 1 diagonal

        // Diagonal should be sqrt(2) * armLength for 90-degree angle
        float expectedDiagonal = MathF.Sqrt(2f) * armLength;
        Assert.InRange(world.Rods[angleRodIdx].RestLength, expectedDiagonal - 0.01f, expectedDiagonal + 0.01f);

        // Act: Simulate
        var stepper = new CPUStepper();
        for (int i = 0; i < 480; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Structure should be stable
        float minY = -0.5f - 0.05f - 0.1f;
        Assert.True(world.PosY[p0] > minY);
        Assert.True(world.PosY[p1] > minY);
        Assert.True(world.PosY[p2] > minY);

        float maxVel = 0.5f;
        for (int i = 0; i < 3; i++)
        {
            Assert.True(MathF.Abs(world.VelX[i]) < maxVel);
            Assert.True(MathF.Abs(world.VelY[i]) < maxVel);
        }
    }

    [Fact]
    public void AngleConstraintFromPositions_API_WorksCorrectly()
    {
        // Arrange: Test the even more convenient API that reads current positions
        var world = new WorldState();
        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            ContactCompliance = 1e-8f
        };

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -1f, hx: 20f, hy: 0.5f));

        // Triangle at height
        float height = 2f;
        float side = 1f;
        float h = side * MathF.Sqrt(3f) / 2f;

        int p0 = world.AddParticle(x: 0f, y: height, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);
        int p1 = world.AddParticle(x: -side / 2f, y: height - h, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);
        int p2 = world.AddParticle(x: side / 2f, y: height - h, vx: 0f, vy: 0f, mass: 1f, radius: 0.05f);

        // Connect edges
        world.Rods.Add(new Rod(p0, p1, restLength: side, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, restLength: side, compliance: 0f));
        world.Rods.Add(new Rod(p2, p0, restLength: side, compliance: 0f));

        // Use ultra-convenient API - automatically reads positions
        // This maintains the current angle at p1 (vertex between p0-p1-p2)
        int angleRod = world.AddAngleConstraintAsRodFromCurrentPositions(p0, p1, p2, compliance: 0f);

        Assert.Equal(4, world.Rods.Count); // 3 edges + 1 diagonal

        // Act: Simulate
        var stepper = new CPUStepper();
        for (int i = 0; i < 480; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Should be stable
        float minY = -0.5f - 0.05f - 0.1f;
        Assert.True(world.PosY[p0] > minY);
        Assert.True(world.PosY[p1] > minY);
        Assert.True(world.PosY[p2] > minY);

        // All should be at rest
        for (int i = 0; i < 3; i++)
        {
            Assert.True(MathF.Abs(world.VelX[i]) < 0.5f);
            Assert.True(MathF.Abs(world.VelY[i]) < 0.5f);
        }
    }

    // Helper methods
    private static float Distance(WorldState world, int i, int j)
    {
        float dx = world.PosX[i] - world.PosX[j];
        float dy = world.PosY[i] - world.PosY[j];
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    private static float ComputeQuadArea(WorldState world, int p0, int p1, int p2, int p3)
    {
        // Shoelace formula for quadrilateral area
        float sum1 = world.PosX[p0] * world.PosY[p1]
                   + world.PosX[p1] * world.PosY[p2]
                   + world.PosX[p2] * world.PosY[p3]
                   + world.PosX[p3] * world.PosY[p0];

        float sum2 = world.PosY[p0] * world.PosX[p1]
                   + world.PosY[p1] * world.PosX[p2]
                   + world.PosY[p2] * world.PosX[p3]
                   + world.PosY[p3] * world.PosX[p0];

        return MathF.Abs(sum1 - sum2) / 2f;
    }
}
