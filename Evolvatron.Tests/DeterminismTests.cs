using Evolvatron.Core;
using Evolvatron.Core.Templates;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Tests for simulation determinism.
/// Verifies that identical initial conditions produce identical results.
/// </summary>
public class DeterminismTests
{
    private const float Tolerance = 1e-6f;

    [Fact]
    public void TwoIdenticalSimulations_ProduceIdenticalResults()
    {
        // Arrange: Create two identical simulations
        var world1 = CreateTestWorld();
        var world2 = CreateTestWorld();

        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1
        };

        var stepper = new CPUStepper();

        // Act: Run both simulations for 1000 steps
        for (int i = 0; i < 1000; i++)
        {
            stepper.Step(world1, config);
            stepper.Step(world2, config);
        }

        // Assert: All particle states should be identical
        Assert.Equal(world1.ParticleCount, world2.ParticleCount);

        for (int i = 0; i < world1.ParticleCount; i++)
        {
            Assert.InRange(world1.PosX[i], world2.PosX[i] - Tolerance, world2.PosX[i] + Tolerance);
            Assert.InRange(world1.PosY[i], world2.PosY[i] - Tolerance, world2.PosY[i] + Tolerance);
            Assert.InRange(world1.VelX[i], world2.VelX[i] - Tolerance, world2.VelX[i] + Tolerance);
            Assert.InRange(world1.VelY[i], world2.VelY[i] - Tolerance, world2.VelY[i] + Tolerance);
        }
    }

    [Fact]
    public void RocketSimulation_IsDeterministic()
    {
        // Arrange: Create two identical rocket scenarios
        var (world1, rocket1) = CreateRocketWorld();
        var (world2, rocket2) = CreateRocketWorld();

        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1
        };

        var stepper = new CPUStepper();

        // Act: Run both with identical control inputs
        for (int i = 0; i < 500; i++)
        {
            // Apply identical thrust
            RocketTemplate.ApplyThrust(world1, rocket1, throttle: 0.5f, maxThrust: 100f);
            RocketTemplate.ApplyThrust(world2, rocket2, throttle: 0.5f, maxThrust: 100f);

            stepper.Step(world1, config);
            stepper.Step(world2, config);
        }

        // Assert: Rocket states should be identical
        RocketTemplate.GetCenterOfMass(world1, rocket1, out float com1X, out float com1Y);
        RocketTemplate.GetCenterOfMass(world2, rocket2, out float com2X, out float com2Y);

        Assert.InRange(com1X, com2X - Tolerance, com2X + Tolerance);
        Assert.InRange(com1Y, com2Y - Tolerance, com2Y + Tolerance);

        RocketTemplate.GetVelocity(world1, rocket1, out float vel1X, out float vel1Y);
        RocketTemplate.GetVelocity(world2, rocket2, out float vel2X, out float vel2Y);

        Assert.InRange(vel1X, vel2X - Tolerance, vel2X + Tolerance);
        Assert.InRange(vel1Y, vel2Y - Tolerance, vel2Y + Tolerance);
    }

    [Fact]
    public void SimulationWithCollisions_IsDeterministic()
    {
        // Arrange: Create two worlds with falling particles and ground
        var world1 = CreateCollisionWorld();
        var world2 = CreateCollisionWorld();

        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            FrictionMu = 0.5f
        };

        var stepper = new CPUStepper();

        // Act: Run both for 2 seconds (480 steps)
        for (int i = 0; i < 480; i++)
        {
            stepper.Step(world1, config);
            stepper.Step(world2, config);
        }

        // Assert: States should match
        for (int i = 0; i < world1.ParticleCount; i++)
        {
            Assert.InRange(world1.PosX[i], world2.PosX[i] - Tolerance, world2.PosX[i] + Tolerance);
            Assert.InRange(world1.PosY[i], world2.PosY[i] - Tolerance, world2.PosY[i] + Tolerance);
            Assert.InRange(world1.VelX[i], world2.VelX[i] - Tolerance, world2.VelX[i] + Tolerance);
            Assert.InRange(world1.VelY[i], world2.VelY[i] - Tolerance, world2.VelY[i] + Tolerance);
        }
    }

    private static WorldState CreateTestWorld()
    {
        var world = new WorldState();

        // Simple scene with a few particles and constraints
        int p0 = world.AddParticle(0f, 5f, 0f, 0f, 1f, 0.1f);
        int p1 = world.AddParticle(1f, 5f, 0f, 0f, 1f, 0.1f);
        int p2 = world.AddParticle(0.5f, 6f, 0f, 0f, 1f, 0.1f);

        world.Rods.Add(new Rod(p0, p1, 1f, 0f));
        world.Rods.Add(new Rod(p1, p2, MathF.Sqrt(0.5f * 0.5f + 1f), 0f));
        world.Rods.Add(new Rod(p2, p0, MathF.Sqrt(0.5f * 0.5f + 1f), 0f));

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -1f, 10f, 0.5f));

        return world;
    }

    private static (WorldState, int[]) CreateRocketWorld()
    {
        var world = new WorldState();

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -5f, 20f, 0.5f));

        // Rocket
        var rocket = RocketTemplate.CreateRocket(world, centerX: 0f, centerY: 5f);

        return (world, rocket);
    }

    private static WorldState CreateCollisionWorld()
    {
        var world = new WorldState();

        // Ground and walls
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -5f, 20f, 0.5f));
        world.Obbs.Add(OBBCollider.AxisAligned(-10f, 0f, 0.5f, 10f));
        world.Obbs.Add(OBBCollider.AxisAligned(10f, 0f, 0.5f, 10f));

        // Some obstacles
        world.Circles.Add(new CircleCollider(-3f, 2f, 0.5f));
        world.Circles.Add(new CircleCollider(3f, 2f, 0.5f));

        // Falling particles
        for (int i = 0; i < 5; i++)
        {
            world.AddParticle(
                x: -4f + i * 2f,
                y: 8f + i * 0.5f,
                vx: 0f,
                vy: 0f,
                mass: 1f,
                radius: 0.1f
            );
        }

        // Connect some with rods
        world.Rods.Add(new Rod(0, 1, 2f, 0f));
        world.Rods.Add(new Rod(2, 3, 2f, 0f));

        return world;
    }
}
