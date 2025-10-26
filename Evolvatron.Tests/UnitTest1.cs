using Evolvatron.Core;
using Evolvatron.Core.Physics;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Unit tests for XPBD constraint solvers and collision detection.
/// </summary>
public class PhysicsTests
{
    private const float Tolerance = 1e-4f;

    [Fact]
    public void RodConstraint_MaintainsDistance()
    {
        // Arrange: Two particles with a rod
        var world = new WorldState();
        int p0 = world.AddParticle(0f, 0f, 0f, 0f, 1f, 0.1f);
        int p1 = world.AddParticle(2f, 0f, 0f, 0f, 1f, 0.1f);
        world.Rods.Add(new Rod(p0, p1, restLength: 1f, compliance: 0f));

        var config = new SimulationConfig { XpbdIterations = 20 };

        // Act: Solve constraints (particles should move to satisfy rod length)
        XPBDSolver.ResetLambdas(world);
        for (int i = 0; i < config.XpbdIterations; i++)
        {
            XPBDSolver.SolveRods(world, config.Dt, config.RodCompliance);
        }

        // Assert: Distance should be 1.0
        float dx = world.PosX[p1] - world.PosX[p0];
        float dy = world.PosY[p1] - world.PosY[p0];
        float dist = MathF.Sqrt(dx * dx + dy * dy);

        Assert.InRange(dist, 1f - Tolerance, 1f + Tolerance);
    }

    [Fact]
    public void AngleConstraint_MaintainsAngle()
    {
        // Arrange: Three particles forming a right angle at p1, with rods for stability
        var world = new WorldState();
        int p0 = world.AddParticle(1f, 0f, 0f, 0f, 1f, 0.1f);
        int p1 = world.AddParticle(0f, 0f, 0f, 0f, 1f, 0.1f);
        int p2 = world.AddParticle(0f, 1f, 0f, 0f, 1f, 0.1f);

        // Add rods to maintain distances
        world.Rods.Add(new Rod(p0, p1, 1f, compliance: 0f));
        world.Rods.Add(new Rod(p1, p2, 1f, compliance: 0f));

        float targetAngle = MathF.PI / 2f; // 90 degrees
        world.Angles.Add(new Angle(p0, p1, p2, targetAngle, compliance: 0f));

        // Perturb position slightly
        world.PosX[p2] = 0.3f;
        world.PosY[p2] = 0.9f;

        var config = new SimulationConfig { XpbdIterations = 40 };

        // Act: Solve both rods and angles
        XPBDSolver.ResetLambdas(world);
        for (int i = 0; i < config.XpbdIterations; i++)
        {
            XPBDSolver.SolveRods(world, config.Dt, config.RodCompliance);
            XPBDSolver.SolveAngles(world, config.Dt, config.AngleCompliance);
        }

        // Assert: Check angle (handle wrapping)
        float e1x = world.PosX[p0] - world.PosX[p1];
        float e1y = world.PosY[p0] - world.PosY[p1];
        float e2x = world.PosX[p2] - world.PosX[p1];
        float e2y = world.PosY[p2] - world.PosY[p1];

        float angle = Math2D.AngleBetween(e1x, e1y, e2x, e2y);
        float angleDiff = Math2D.WrapAngle(angle - targetAngle);

        Assert.InRange(MathF.Abs(angleDiff), 0f, 0.05f); // Slightly relaxed tolerance
    }

    [Fact]
    public void CircleCollision_PushesOut()
    {
        // Arrange: Particle penetrating a circle
        var world = new WorldState();
        int p = world.AddParticle(0.5f, 0f, 0f, 0f, 1f, 0.1f);
        world.Circles.Add(new CircleCollider(0f, 0f, 1f));

        var config = new SimulationConfig { XpbdIterations = 20 };

        // Act: Solve contacts
        for (int i = 0; i < config.XpbdIterations; i++)
        {
            XPBDSolver.SolveContacts(world, config.Dt, config.ContactCompliance);
        }

        // Assert: Particle should be pushed outside circle
        float dx = world.PosX[p];
        float dy = world.PosY[p];
        float dist = MathF.Sqrt(dx * dx + dy * dy);

        // Distance should be >= circle radius + particle radius
        Assert.True(dist >= 1.0f + 0.1f - Tolerance);
    }

    [Fact]
    public void OBBCollision_PushesOut()
    {
        // Arrange: Particle inside an OBB
        var world = new WorldState();
        int p = world.AddParticle(0f, 0f, 0f, 0f, 1f, 0.1f);
        world.Obbs.Add(OBBCollider.AxisAligned(0f, 0f, 2f, 2f));

        var config = new SimulationConfig { XpbdIterations = 20 };

        // Act: Solve contacts
        for (int i = 0; i < config.XpbdIterations; i++)
        {
            XPBDSolver.SolveContacts(world, config.Dt, config.ContactCompliance);
        }

        // Assert: Particle should be pushed to edge
        // Since it starts at center, it should be pushed to nearest face
        float distToEdge = MathF.Min(
            MathF.Abs(world.PosX[p] - 2f),
            MathF.Abs(world.PosX[p] + 2f)
        );

        // Should be near the edge (within tolerance)
        Assert.True(distToEdge < 0.2f || MathF.Abs(world.PosY[p]) > 1.8f);
    }

    [Fact]
    public void Integration_ConservesEnergy_WithNoDamping()
    {
        // Arrange: Particle with initial velocity, no gravity, no damping
        var world = new WorldState();
        int p = world.AddParticle(0f, 0f, 1f, 0f, 1f, 0.1f);

        var config = new SimulationConfig
        {
            GravityX = 0f,
            GravityY = 0f,
            GlobalDamping = 0f,
            VelocityStabilizationBeta = 0f
        };

        float initialKE = 0.5f * 1f * 1f; // 0.5 * m * v^2

        // Act: Integrate for many steps
        var stepper = new CPUStepper();
        for (int i = 0; i < 1000; i++)
        {
            stepper.Step(world, config);
        }

        // Assert: Kinetic energy should be approximately conserved
        float finalSpeed = MathF.Sqrt(world.VelX[p] * world.VelX[p] + world.VelY[p] * world.VelY[p]);
        float finalKE = 0.5f * 1f * finalSpeed * finalSpeed;

        Assert.InRange(finalKE, initialKE - 0.01f, initialKE + 0.01f);
    }

    [Fact]
    public void Friction_ReducesTangentialVelocity()
    {
        // Arrange: Particle on ground with velocity
        var world = new WorldState();
        int p = world.AddParticle(0f, -0.4f, 2f, -0.5f, 1f, 0.1f);
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -0.5f, 10f, 0.1f));

        // Apply contact solve first to establish contact
        for (int i = 0; i < 20; i++)
        {
            XPBDSolver.SolveContacts(world, 1f / 60f, 1e-8f);
        }

        float initialVelX = world.VelX[p];

        // Ensure particle has some tangential velocity
        if (MathF.Abs(initialVelX) < 0.1f)
        {
            world.VelX[p] = 2f;
            initialVelX = 2f;
        }

        // Act: Apply friction
        Friction.ApplyFriction(world, frictionMu: 0.5f);

        // Assert: Velocity should be reduced
        Assert.True(MathF.Abs(world.VelX[p]) < MathF.Abs(initialVelX));
    }
}