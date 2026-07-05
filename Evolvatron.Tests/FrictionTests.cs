using System;
using Evolvatron.Core;
using Evolvatron.Core.Templates;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests;

/// <summary>
/// Rigorous Coulomb-friction validation for rigid-body contacts (the path Walker's feet use).
///
/// Earlier friction coverage in <see cref="RigidBodyStabilityTests"/> predates the
/// <c>ImpulseContactSolver</c> friction and leans on <c>GlobalDamping</c> to stop bodies ("friction
/// isn't implemented yet"), so it never actually checks Coulomb behaviour. These tests zero BOTH
/// damping terms so friction is the only tangential force, and assert the textbook predictions:
///   • a body at rest on flat ground does not drift (static friction holds);
///   • a sliding body decelerates at ≈ μg and stops (kinetic friction);
///   • on an incline of angle θ (modelled by tilting gravity over flat ground) the body STICKS when
///     μ &gt; tanθ and SLIDES at g(sinθ − μcosθ) when μ &lt; tanθ (the critical-angle law).
/// </summary>
public class FrictionTests
{
    private readonly ITestOutputHelper _output;

    public FrictionTests(ITestOutputHelper output) => _output = output;

    private static SimulationConfig FrictionOnly(float mu, float gx = 0f, float gy = -9.81f) => new()
    {
        Dt = 1f / 240f,
        Substeps = 1,
        XpbdIterations = 12,
        GravityX = gx,
        GravityY = gy,
        FrictionMu = mu,
        Restitution = 0f,
        VelocityStabilizationBeta = 1.0f,
        GlobalDamping = 0f,   // isolate friction: no linear damping masking it
        AngularDamping = 0f,  // and no angular damping
    };

    // Settle a freshly created body onto the ground so a warm-started resting contact exists.
    private static void Settle(WorldState world, CPUStepper stepper, SimulationConfig cfg, float seconds)
    {
        int steps = (int)(seconds / cfg.Dt);
        for (int i = 0; i < steps; i++) stepper.Step(world, cfg);
    }

    [Fact]
    public void RestingBox_OnFlatGround_DoesNotDrift()
    {
        var world = new WorldState(128);
        var cfg = FrictionOnly(0.9f);
        var stepper = new CPUStepper();
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -0.5f, 20f, 0.5f)); // top surface at y=0
        RigidBodyFactory.CreateBox(world, 0f, 0.12f, 0.5f, 0.1f, 1f, 0f);

        Settle(world, stepper, cfg, 0.5f);
        float x0 = world.RigidBodies[0].X;
        Settle(world, stepper, cfg, 2.0f);
        var rb = world.RigidBodies[0];

        _output.WriteLine($"rest drift: dx={rb.X - x0:F4} m, VelX={rb.VelX:F4} m/s");
        Assert.True(MathF.Abs(rb.X - x0) < 0.02f, $"box drifted {rb.X - x0:F4} m on flat ground");
        Assert.True(MathF.Abs(rb.VelX) < 0.05f, $"box has residual VelX={rb.VelX:F4} m/s at rest");
    }

    [Fact]
    public void SlidingBox_DeceleratesAtCoulombRate_AndStops()
    {
        const float mu = 0.7f;
        var world = new WorldState(128);
        var cfg = FrictionOnly(mu);
        var stepper = new CPUStepper();
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -0.5f, 20f, 0.5f)); // top surface at y=0
        RigidBodyFactory.CreateBox(world, 0f, 0.12f, 0.5f, 0.1f, 1f, 0f); // low flat slab → minimal pitching

        Settle(world, stepper, cfg, 0.5f);
        var rb = world.RigidBodies[0];
        rb.VelX = 3f;
        world.RigidBodies[0] = rb;

        float expectedDecel = mu * 9.81f;          // Coulomb kinetic: a = μg
        // Measure deceleration over a short window while clearly sliding.
        float vStart = 3f, tWindow = 0.1f;
        int wSteps = (int)(tWindow / cfg.Dt);
        for (int i = 0; i < wSteps; i++) stepper.Step(world, cfg);
        float vMid = world.RigidBodies[0].VelX;
        float observedDecel = (vStart - vMid) / tWindow;

        // Run to a stop and record stopping distance/time.
        float xAtLaunch = 0f; // COM started ~0 after settle (symmetric)
        float t = tWindow;
        while (world.RigidBodies[0].VelX > 0.05f && t < 5f) { stepper.Step(world, cfg); t += cfg.Dt; }
        var fin = world.RigidBodies[0];
        float stopDist = fin.X - xAtLaunch;
        float theoryDist = (vStart * vStart) / (2f * expectedDecel);

        _output.WriteLine($"μ={mu}: expected decel {expectedDecel:F2} m/s², observed (first {tWindow}s) {observedDecel:F2} m/s²");
        _output.WriteLine($"stop: t={t:F2}s, dist={stopDist:F3} m (theory {theoryDist:F3} m), finalVelX={fin.VelX:F3}");

        Assert.True(fin.VelX <= 0.06f, $"box never stopped: VelX={fin.VelX:F3} after {t:F2}s");
        Assert.InRange(observedDecel, 0.6f * expectedDecel, 1.5f * expectedDecel);
    }

    [Theory]
    [InlineData(20f, 0.9f, false)] // μ=0.9 > tan20°=0.364 → STICK
    [InlineData(20f, 0.2f, true)]  // μ=0.2 < tan20°=0.364 → SLIDE
    public void BoxOnIncline_ObeysCriticalAngle(float degrees, float mu, bool shouldSlide)
    {
        float theta = degrees * MathF.PI / 180f;
        const float g = 9.81f;
        // Incline modelled as flat ground with gravity tilted by θ (down-slope is +x here).
        var world = new WorldState(128);
        var cfg = FrictionOnly(mu, gx: g * MathF.Sin(theta), gy: -g * MathF.Cos(theta));
        var stepper = new CPUStepper();
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -0.5f, 20f, 0.5f)); // top surface at y=0
        RigidBodyFactory.CreateBox(world, 0f, 0.5f, 0.5f, 0.5f, 1f, 0f); // square box (won't topple at 20°)

        // Let it land/settle first (vertical), then time the slide from rest.
        Settle(world, stepper, cfg, 0.3f);
        var rb = world.RigidBodies[0];
        rb.VelX = 0f; rb.VelY = 0f; rb.AngularVel = 0f;
        world.RigidBodies[0] = rb;
        float x0 = world.RigidBodies[0].X;

        Settle(world, stepper, cfg, 1.0f);
        var fin = world.RigidBodies[0];
        float disp = fin.X - x0;
        float theoryAccel = g * (MathF.Sin(theta) - mu * MathF.Cos(theta)); // >0 if sliding
        float theoryDisp = shouldSlide ? 0.5f * theoryAccel * 1f * 1f : 0f;

        _output.WriteLine($"θ={degrees}° μ={mu} (tanθ={MathF.Tan(theta):F3}): disp over 1s = {disp:F3} m, " +
                          $"VelX={fin.VelX:F3} | predicted {(shouldSlide ? $"SLIDE ~{theoryDisp:F3} m @ {theoryAccel:F2} m/s²" : "STICK ~0 m")}");

        if (shouldSlide)
            Assert.True(disp > 0.3f, $"box should have slid (μ<tanθ) but only moved {disp:F3} m");
        else
            Assert.True(MathF.Abs(disp) < 0.05f, $"box should have stuck (μ>tanθ) but moved {disp:F3} m");
    }
}
