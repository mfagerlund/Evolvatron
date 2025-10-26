using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.Scenes;
using System;
using System.Diagnostics;

namespace Evolvatron.Demo;

/// <summary>
/// Benchmarks GPU vs CPU performance.
/// M5: GPU acceleration with ILGPU.
/// </summary>
public static class GPUBenchmark
{
    public static void Run()
    {
        Console.WriteLine("=== Evolvatron GPU Benchmark (M5) ===");
        Console.WriteLine("Comparing GPU vs CPU performance");
        Console.WriteLine();

        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            XpbdIterations = 12,
            Substeps = 1,
            GravityY = -9.81f,
            ContactCompliance = 1e-8f,
            RodCompliance = 0f,
            AngleCompliance = 0f,
            MotorCompliance = 1e-6f,
            FrictionMu = 0.6f,
            VelocityStabilizationBeta = 1.0f,
            GlobalDamping = 0.02f
        };

        // Run benchmark with different particle counts
        int[] particleCounts = { 50, 100, 200, 400 };

        Console.WriteLine("Benchmark configuration:");
        Console.WriteLine($"  - dt: {config.Dt:F6}s ({1f / config.Dt:F0} Hz)");
        Console.WriteLine($"  - XPBD iterations: {config.XpbdIterations}");
        Console.WriteLine($"  - Steps per test: 1000");
        Console.WriteLine();

        foreach (int targetParticles in particleCounts)
        {
            Console.WriteLine($"--- Testing with ~{targetParticles} particles ---");

            // Create world
            var world = CreateBenchmarkWorld(targetParticles);

            Console.WriteLine($"Created world: {world.ParticleCount} particles, {world.Rods.Count} rods, " +
                            $"{world.Angles.Count} angles, {world.Obbs.Count} OBBs");

            // Benchmark CPU
            var worldCPU = CloneWorld(world);
            var cpuTime = BenchmarkCPU(worldCPU, config, steps: 1000);

            // Benchmark GPU
            var worldGPU = CloneWorld(world);
            var gpuTime = BenchmarkGPU(worldGPU, config, steps: 1000);

            // Results
            Console.WriteLine();
            Console.WriteLine($"Results:");
            Console.WriteLine($"  CPU: {cpuTime:F2}ms total, {cpuTime / 1000f:F3}ms/step");
            Console.WriteLine($"  GPU: {gpuTime:F2}ms total, {gpuTime / 1000f:F3}ms/step");
            Console.WriteLine($"  Speedup: {cpuTime / gpuTime:F2}x");
            Console.WriteLine();

            // Validate results match
            Console.WriteLine("Validating CPU vs GPU accuracy...");
            ValidateResults(world, worldCPU, worldGPU);
            Console.WriteLine();
        }

        Console.WriteLine("Benchmark complete!");
    }

    private static double BenchmarkCPU(WorldState world, SimulationConfig config, int steps)
    {
        var stepper = new CPUStepper();
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < steps; i++)
        {
            stepper.Step(world, config);
        }

        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static double BenchmarkGPU(WorldState world, SimulationConfig config, int steps)
    {
        using var stepper = new GPUStepper();

        // Warmup (first call includes compilation)
        Console.Write("  GPU warmup... ");
        stepper.Step(world, config);
        Console.WriteLine("done");

        // Actual benchmark
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < steps; i++)
        {
            stepper.Step(world, config);
        }

        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static WorldState CreateBenchmarkWorld(int targetParticles)
    {
        var world = new WorldState(targetParticles * 2);

        // Create funnel scene
        FunnelSceneBuilder.BuildFunnelScene(world, 20f, 15f, 30f, -10f, 4f);

        // Spawn contraptions until we hit target
        var spawner = new ContraptionSpawner(seed: 42);
        FunnelSceneBuilder.GetSpawnBounds(20f, 15f, -10f, out float minX, out float maxX, out float minY, out float maxY);

        while (world.ParticleCount < targetParticles)
        {
            spawner.SpawnRandomContraption(world, minX, maxX, minY, maxY, 4, 10);
        }

        return world;
    }

    private static WorldState CloneWorld(WorldState original)
    {
        var clone = new WorldState(original.Capacity);

        for (int i = 0; i < original.ParticleCount; i++)
        {
            clone.AddParticle(
                original.PosX[i], original.PosY[i],
                original.VelX[i], original.VelY[i],
                original.InvMass[i] > 0f ? 1f / original.InvMass[i] : 0f,
                original.Radius[i]
            );
        }

        clone.Rods.AddRange(original.Rods);
        clone.Angles.AddRange(original.Angles);
        clone.Motors.AddRange(original.Motors);
        clone.Circles.AddRange(original.Circles);
        clone.Capsules.AddRange(original.Capsules);
        clone.Obbs.AddRange(original.Obbs);

        return clone;
    }

    private static void ValidateResults(WorldState original, WorldState cpu, WorldState gpu)
    {
        float maxPosDiff = 0f;
        float maxVelDiff = 0f;

        for (int i = 0; i < original.ParticleCount; i++)
        {
            float posDiffX = MathF.Abs(cpu.PosX[i] - gpu.PosX[i]);
            float posDiffY = MathF.Abs(cpu.PosY[i] - gpu.PosY[i]);
            float velDiffX = MathF.Abs(cpu.VelX[i] - gpu.VelX[i]);
            float velDiffY = MathF.Abs(cpu.VelY[i] - gpu.VelY[i]);

            maxPosDiff = MathF.Max(maxPosDiff, MathF.Max(posDiffX, posDiffY));
            maxVelDiff = MathF.Max(maxVelDiff, MathF.Max(velDiffX, velDiffY));
        }

        Console.WriteLine($"  Max position difference: {maxPosDiff:E3}");
        Console.WriteLine($"  Max velocity difference: {maxVelDiff:E3}");

        if (maxPosDiff < 0.01f && maxVelDiff < 0.1f)
        {
            Console.WriteLine("  ✓ Results match within tolerance");
        }
        else
        {
            Console.WriteLine("  ⚠ Results differ more than expected (may be due to atomic operations)");
        }
    }
}
