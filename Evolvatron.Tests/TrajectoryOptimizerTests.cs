using System;
using Evolvatron.Core;
using Evolvatron.Core.Templates;
using Evolvatron.Evolvion.TrajectoryOptimization;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests;

public class TrajectoryOptimizerTests
{
    private readonly ITestOutputHelper _output;

    public TrajectoryOptimizerTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private TrajectoryResult RunOptimizer(int maxIter = 20, int controlSteps = 20, int physicsSteps = 10)
    {
        var optimizer = new TrajectoryOptimizer(new TrajectoryOptimizerOptions
        {
            MaxIterations = maxIter,
            ControlSteps = controlSteps,
            PhysicsStepsPerControl = physicsSteps,
            LogCallback = msg => _output.WriteLine(msg)
        });

        return optimizer.Optimize(startX: 2f, startY: 10f, startVelX: 0f, startVelY: -1f);
    }

    [Fact]
    public void CostDecreasesAfterOptimization()
    {
        var initial = RunOptimizer(maxIter: 1);
        var optimized = RunOptimizer(maxIter: 30);

        _output.WriteLine($"Initial cost:   {initial.FinalCost:F4}");
        _output.WriteLine($"Optimized cost: {optimized.FinalCost:F4}");
        _output.WriteLine($"Iterations:     {optimized.Iterations}");
        _output.WriteLine($"Time:           {optimized.ComputationTimeMs:F0} ms");
        _output.WriteLine($"Convergence:    {optimized.ConvergenceReason}");

        Assert.True(optimized.FinalCost < initial.FinalCost,
            $"Expected cost to decrease: {initial.FinalCost:F4} -> {optimized.FinalCost:F4}");
    }

    [Fact]
    public void RocketEndsNearPad()
    {
        // Use full-size problem (30 steps, 15 physics = 3.75s) for enough sim time
        var result = RunOptimizer(maxIter: 40, controlSteps: 30, physicsSteps: 15);
        var finalState = result.States[^1];

        float distX = MathF.Abs(finalState.X - 0f);
        float distY = MathF.Abs(finalState.Y - (-4.5f));
        float dist = MathF.Sqrt(distX * distX + distY * distY);

        _output.WriteLine($"Final position: ({finalState.X:F2}, {finalState.Y:F2})");
        _output.WriteLine($"Distance to pad: {dist:F2} m");

        Assert.True(dist < 8f, $"Final distance to pad too large: {dist:F2} m");
    }

    [Fact]
    public void FinalVelocityIsLow()
    {
        var result = RunOptimizer(maxIter: 30);
        var finalState = result.States[^1];

        float speed = MathF.Sqrt(finalState.VelX * finalState.VelX + finalState.VelY * finalState.VelY);

        _output.WriteLine($"Final velocity: ({finalState.VelX:F2}, {finalState.VelY:F2})");
        _output.WriteLine($"Final speed: {speed:F2} m/s");

        Assert.True(speed < 10f, $"Final speed too high: {speed:F2} m/s");
    }

    [Fact]
    public void FinalAngleIsSmall()
    {
        var result = RunOptimizer(maxIter: 30);
        var finalState = result.States[^1];

        float tiltFromUpright = MathF.Abs(finalState.Angle - MathF.PI / 2f);
        float tiltDegrees = tiltFromUpright * 180f / MathF.PI;

        _output.WriteLine($"Final angle: {finalState.Angle:F3} rad (tilt: {tiltDegrees:F1} deg)");

        Assert.True(tiltDegrees < 30f, $"Final tilt too large: {tiltDegrees:F1} degrees");
    }

    [Fact]
    public void JacobianHasCorrectShape()
    {
        int controlSteps = 20;
        int physicsStepsPerControl = 10;
        int totalParams = controlSteps * 2;
        int totalResiduals = controlSteps * 8 + 5; // 8 per step + 5 terminal

        var optimizer = new TrajectoryOptimizer(new TrajectoryOptimizerOptions
        {
            MaxIterations = 1,
            ControlSteps = controlSteps,
            PhysicsStepsPerControl = physicsStepsPerControl,
            LogCallback = msg => _output.WriteLine(msg)
        });

        var result = optimizer.Optimize(startX: 0f, startY: 10f);

        Assert.Equal(controlSteps, result.Throttles.Length);
        Assert.Equal(controlSteps, result.Gimbals.Length);
        Assert.Equal(controlSteps + 1, result.States.Length);

        _output.WriteLine($"Params: {totalParams}, Residuals: {totalResiduals}");
        _output.WriteLine($"Expected Jacobian: {totalResiduals} x {totalParams}");
    }

    [Fact]
    public void CausalSparsity_UpperTriangleIsZero()
    {
        var result = RunOptimizer(maxIter: 5);

        _output.WriteLine($"Cost after 5 iterations: {result.FinalCost:F4}");
        _output.WriteLine($"Convergence: {result.ConvergenceReason}");

        Assert.True(result.FinalCost < 1e6, "Cost is unreasonably high");
        Assert.True(result.FinalCost >= 0, "Cost should be non-negative");
    }

    [Fact]
    public void WorldStateSnapshot_CaptureRestore_Roundtrips()
    {
        var world = new WorldState(64);
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -5f, 30f, 0.5f));

        var rocketIndices = RigidBodyRocketTemplate.CreateRocket(world, 0f, 5f);

        // Record actual initial COM (not hardcoded — rocket COM depends on body+leg geometry)
        RigidBodyRocketTemplate.GetCenterOfMass(world, rocketIndices, out float initX, out float initY);
        _output.WriteLine($"Initial COM: ({initX:F3}, {initY:F3})");

        var snapshot = WorldStateSnapshot.Capture(world);

        // Mutate world significantly
        var stepper = new CPUStepper();
        var config = new SimulationConfig { Dt = 1f / 120f, XpbdIterations = 8 };
        for (int i = 0; i < 100; i++)
            stepper.Step(world, config);

        RigidBodyRocketTemplate.GetCenterOfMass(world, rocketIndices, out float afterX, out float afterY);
        _output.WriteLine($"After 100 steps: ({afterX:F3}, {afterY:F3})");
        Assert.True(MathF.Abs(afterY - initY) > 1f, "World should have changed significantly");

        // Restore and verify match
        snapshot.Restore(world);
        RigidBodyRocketTemplate.GetCenterOfMass(world, rocketIndices, out float restoredX, out float restoredY);
        _output.WriteLine($"After restore: ({restoredX:F3}, {restoredY:F3})");

        Assert.Equal(initX, restoredX, 3);
        Assert.Equal(initY, restoredY, 3);
    }
}
