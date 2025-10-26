using Evolvatron.Core;
using Evolvatron.Core.Scenes;
using System;
using System.Collections.Generic;

namespace Evolvatron.Demo;

/// <summary>
/// Demo showing the funnel scene with random contraptions falling through.
/// M4: Complete funnel demo with spawning, culling, and landing stats.
/// </summary>
public static class FunnelDemo
{
    public static void Run()
    {
        Console.WriteLine("=== Evolvatron Funnel Demo (M4) ===");
        Console.WriteLine("Random contraptions falling through a funnel");
        Console.WriteLine();

        var world = new WorldState(initialCapacity: 512);
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
            GlobalDamping = 0.02f // Slightly more damping for stability
        };

        var stepper = new CPUStepper();

        // Build funnel scene
        float funnelWidth = 20f;
        float funnelHeight = 15f;
        float groundY = -10f;
        float padWidth = 4f;

        FunnelSceneBuilder.BuildFunnelScene(world, funnelWidth, funnelHeight, 30f, groundY, padWidth);

        FunnelSceneBuilder.GetSpawnBounds(funnelWidth, funnelHeight, groundY,
            out float spawnMinX, out float spawnMaxX, out float spawnMinY, out float spawnMaxY);

        FunnelSceneBuilder.GetPadBounds(groundY, padWidth,
            out float padMinX, out float padMaxX, out float padMinY, out float padMaxY);

        FunnelSceneBuilder.GetCullBounds(funnelWidth, funnelHeight, groundY,
            out float cullMinX, out float cullMaxX, out float cullMinY, out float cullMaxY);

        Console.WriteLine($"Funnel scene created:");
        Console.WriteLine($"  - Spawn area: X=[{spawnMinX:F1}, {spawnMaxX:F1}] Y=[{spawnMinY:F1}, {spawnMaxY:F1}]");
        Console.WriteLine($"  - Landing pad: X=[{padMinX:F1}, {padMaxX:F1}] Y=[{padMinY:F1}, {padMaxY:F1}]");
        Console.WriteLine($"  - Funnel: {world.Obbs.Count} OBBs, {world.Circles.Count} circles, {world.Capsules.Count} capsules");
        Console.WriteLine();

        var spawner = new ContraptionSpawner(seed: 42);

        // Tracking
        List<List<int>> activeContraptions = new List<List<int>>();
        float nextSpawnTime = 0f;
        float spawnInterval = 2f; // Spawn every 2 seconds
        int totalSpawned = 0;
        int landed = 0;
        int outOfBounds = 0;

        float simTime = 0f;
        float maxTime = 30f;
        int stepCount = 0;

        Console.WriteLine($"Running simulation for {maxTime}s...");
        Console.WriteLine($"Spawning contraptions every {spawnInterval}s");
        Console.WriteLine();

        while (simTime < maxTime)
        {
            // Spawn new contraption periodically
            if (simTime >= nextSpawnTime)
            {
                var contraption = spawner.SpawnRandomContraption(
                    world, spawnMinX, spawnMaxX, spawnMinY, spawnMaxY,
                    minParticles: 4, maxParticles: 10);

                activeContraptions.Add(contraption);
                totalSpawned++;
                nextSpawnTime = simTime + spawnInterval;

                Console.WriteLine($"[t={simTime:F2}s] Spawned contraption #{totalSpawned} ({contraption.Count} particles)");
            }

            // Step simulation
            stepper.Step(world, config);
            simTime += config.Dt;
            stepCount++;

            // Check contraptions for landing or culling
            for (int i = activeContraptions.Count - 1; i >= 0; i--)
            {
                var contraption = activeContraptions[i];

                // Compute COM
                float comX = 0f, comY = 0f;
                float totalMass = 0f;
                bool allValid = true;

                foreach (int idx in contraption)
                {
                    if (idx >= world.ParticleCount)
                    {
                        allValid = false;
                        break;
                    }

                    float mass = world.InvMass[idx] > 0f ? 1f / world.InvMass[idx] : 0f;
                    totalMass += mass;
                    comX += world.PosX[idx] * mass;
                    comY += world.PosY[idx] * mass;
                }

                if (!allValid || totalMass <= 0f)
                {
                    activeContraptions.RemoveAt(i);
                    continue;
                }

                comX /= totalMass;
                comY /= totalMass;

                // Check if landed on pad
                if (comX >= padMinX && comX <= padMaxX && comY >= padMinY && comY <= padMaxY)
                {
                    // Check velocity
                    float velX = 0f, velY = 0f;
                    foreach (int idx in contraption)
                    {
                        if (idx < world.ParticleCount)
                        {
                            float mass = world.InvMass[idx] > 0f ? 1f / world.InvMass[idx] : 0f;
                            velX += world.VelX[idx] * mass;
                            velY += world.VelY[idx] * mass;
                        }
                    }
                    velX /= totalMass;
                    velY /= totalMass;

                    if (MathF.Abs(velY) < 0.5f && MathF.Abs(velX) < 0.5f)
                    {
                        landed++;
                        activeContraptions.RemoveAt(i);
                        Console.WriteLine($"[t={simTime:F2}s] Contraption LANDED! (Total: {landed}/{totalSpawned})");
                        continue;
                    }
                }

                // Check if out of bounds
                if (comX < cullMinX || comX > cullMaxX || comY < cullMinY || comY > cullMaxY)
                {
                    outOfBounds++;
                    activeContraptions.RemoveAt(i);
                    Console.WriteLine($"[t={simTime:F2}s] Contraption out of bounds (Total OOB: {outOfBounds})");
                }
            }

            // Print status every 5s
            if (stepCount % 1200 == 0)
            {
                Console.WriteLine($"[t={simTime:F2}s] Active: {activeContraptions.Count}, Particles: {world.ParticleCount}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("=== Simulation Complete ===");
        Console.WriteLine($"Ran {stepCount} steps ({simTime:F2}s)");
        Console.WriteLine($"Total spawned: {totalSpawned}");
        Console.WriteLine($"Landed on pad: {landed} ({(float)landed / totalSpawned * 100f:F1}%)");
        Console.WriteLine($"Out of bounds: {outOfBounds}");
        Console.WriteLine($"Still active: {activeContraptions.Count}");
        Console.WriteLine($"Final particles: {world.ParticleCount}");
        Console.WriteLine($"Rods: {world.Rods.Count}, Angles: {world.Angles.Count}, Motors: {world.Motors.Count}");
    }
}
