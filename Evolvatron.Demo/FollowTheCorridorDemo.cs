using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Raylib_cs;
using System.Numerics;
using System.Diagnostics;
using RaylibVector2 = System.Numerics.Vector2;
using GodotVector2 = Godot.Vector2;
using RaylibColor = Raylib_cs.Color;
using Colonel.Tests.HagridTests.FollowTheCorridor;
using static Colonel.Tests.HagridTests.FollowTheCorridor.SimpleCarWorld;

namespace Evolvatron.Demo;

/// <summary>
/// Real-time visualization of FollowTheCorridor evolution.
/// Shows multiple cars from the population evolving over generations.
/// </summary>
public static class FollowTheCorridorDemo
{
    private const int ScreenWidth = 1600;
    private const int ScreenHeight = 900;
    private const float Scale = 2.5f; // Pixels per world unit

    public static void Run()
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning); // Reduce startup noise
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Follow The Corridor");
        Raylib.SetTargetFPS(60);

        // Create topology
        var topology = CreateCorridorTopology();

        // Configure evolution
        var config = new EvolutionConfig
        {
            SpeciesCount = 40,
            IndividualsPerSpecies = 40, // 1600 total agents
            Elites = 1, // Only keep top 1 unchanged
            TournamentSize = 4,
            ParentPoolPercentage = 0.2f // Only top 20% eligible as parents
        };

        // Initialize population
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);
        var baseEnvironment = new FollowTheCorridorEnvironment(maxSteps: 320);

        int generation = 0;
        int maxGenerations = 1000;
        float bestFitness = float.MinValue;

        // Accumulated timing stats
        long totalFitnessEvalMs = 0;
        long totalEvolutionMs = 0;

        // Overall timer for elapsed time display
        var overallStopwatch = Stopwatch.StartNew();

        object stateLock = new object();

        // Background simulation state
        Task? simulationTask = null;
        bool isSimulating = false;
        bool solved = false;

        // Control whether to recompute elite fitness each generation
        // When false, elites keep their fitness from previous generation (faster, guarantees monotonic increase)
        // When true, elites are re-evaluated for visualization purposes (slower, but shows all agents)
        bool recomputeElites = false;

        // Track which individuals are elites (to avoid re-evaluating them)
        var isElite = new bool[config.SpeciesCount * config.IndividualsPerSpecies];

        // Camera offset
        RaylibVector2 cameraOffset = new RaylibVector2(200, ScreenHeight - 100);

        // Create 1600 environments (one per agent)
        var environments = new List<FollowTheCorridorEnvironment>();
        var individuals = new List<Individual>();
        var evaluators = new List<CPUEvaluator>();
        var totalRewards = new List<float>();

        foreach (var species in population.AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                environments.Add(new FollowTheCorridorEnvironment(maxSteps: 320));
                individuals.Add(individual);
                evaluators.Add(new CPUEvaluator(species.Topology)); // One evaluator per individual for thread safety
                totalRewards.Add(0f);
            }
        }

        int currentStep = 0;

        // Helper to start new generation
        void StartNewGeneration()
        {
            // Reset environments
            for (int i = 0; i < environments.Count; i++)
            {
                if (!isElite[i] || recomputeElites)
                {
                    environments[i].Reset(seed: generation);
                    totalRewards[i] = 0f;
                }
                // Elites (when recomputeElites=false): don't reset environment or totalRewards
            }

            currentStep = 0;

            // Start background simulation thread
            isSimulating = true;

            simulationTask = Task.Run(() =>
            {
                // Simulate all non-elite agents step by step
                var swEval = Stopwatch.StartNew();
                var observations = new float[environments[0].InputCount];

                while (currentStep < 320)
                {
                    lock (stateLock)
                    {
                        for (int i = 0; i < environments.Count; i++)
                        {
                            // Skip elites unless recomputeElites is true
                            if (isElite[i] && !recomputeElites)
                                continue;

                            if (!environments[i].IsTerminal())
                            {
                                environments[i].GetObservations(observations);
                                var actions = evaluators[i].Evaluate(individuals[i], observations);
                                float reward = environments[i].Step(actions);
                                totalRewards[i] += reward;
                            }
                        }
                        currentStep++;
                    }
                }

                // Collect fitnesses (skip elites unless recomputeElites is true)
                for (int i = 0, idx = 0; i < population.AllSpecies.Count; i++)
                {
                    var species = population.AllSpecies[i];
                    for (int j = 0; j < species.Individuals.Count; j++, idx++)
                    {
                        if (!isElite[idx] || recomputeElites)
                        {
                            var ind = species.Individuals[j];
                            ind.Fitness = totalRewards[idx];
                            species.Individuals[j] = ind;
                            individuals[idx] = ind;
                        }
                    }
                }

                swEval.Stop();

                lock (stateLock)
                {
                    totalFitnessEvalMs += swEval.ElapsedMilliseconds;

                    var best = population.GetBestIndividual();
                    bestFitness = best.HasValue ? best.Value.individual.Fitness : 0f;

                    // Check if solved
                    if (bestFitness > 0.9f)
                    {
                        solved = true;
                    }
                }

                isSimulating = false;
            });
        }

        // Start initial generation
        StartNewGeneration();

        while (!Raylib.WindowShouldClose() && !solved)
        {
            // Auto-advance to next generation as soon as simulation finishes
            lock (stateLock)
            {
                bool shouldAdvance = !isSimulating && generation < maxGenerations;

                if (shouldAdvance)
                {
                    var sw = Stopwatch.StartNew();
                    evolver.StepGeneration(population);
                    sw.Stop();
                    totalEvolutionMs += sw.ElapsedMilliseconds;
                    generation++;

                    // Mark elites (first config.Elites individuals in each species)
                    Array.Fill(isElite, false);
                    for (int i = 0, idx = 0; i < population.AllSpecies.Count; i++)
                    {
                        for (int j = 0; j < population.AllSpecies[i].Individuals.Count; j++, idx++)
                        {
                            isElite[idx] = (j < config.Elites);
                        }
                    }

                    StartNewGeneration();
                }
            }

            // === RENDER ===
            Raylib.BeginDrawing();
            Raylib.ClearBackground(RaylibColor.Black);

            // Render track
            RenderTrack(baseEnvironment.World, cameraOffset);

            // Render ALL 1600 cars (dead ones in red, alive ones in blue)
            lock (stateLock)
            {
                int activeCars = 0;
                for (int i = 0; i < environments.Count; i++)
                {
                    var position = environments[i].GetCarPosition();
                    var heading = environments[i].GetCarHeading();

                    if (!environments[i].IsTerminal())
                    {
                        activeCars++;
                        // Alive cars: blue
                        var carColor = new RaylibColor(100, 200, 255, 180);
                        RenderCar(position, heading, cameraOffset, carColor);
                    }
                    else
                    {
                        // Dead cars: red (at their final position)
                        var carColor = new RaylibColor(255, 100, 100, 120);
                        RenderCar(position, heading, cameraOffset, carColor);
                    }
                }

                // Render UI
                RenderUI(generation, bestFitness, currentStep, activeCars, isSimulating, overallStopwatch.Elapsed);
            }

            Raylib.EndDrawing();
        }

        // Cleanup
        simulationTask?.Wait();
        Raylib.CloseWindow();

        // Print summary
        long totalMs = totalFitnessEvalMs + totalEvolutionMs;
        Console.WriteLine($"\n=== SUMMARY ===");
        Console.WriteLine($"Generations: {generation}");
        Console.WriteLine($"Final fitness: {bestFitness:F3} ({bestFitness * 100:F1}%)");
        Console.WriteLine($"Status: {(solved ? "SOLVED!" : "Stopped")}");
        Console.WriteLine($"\nTime breakdown ({totalMs}ms total):");
        Console.WriteLine($"  Fitness evaluation: {totalFitnessEvalMs}ms ({100.0 * totalFitnessEvalMs / totalMs:F1}%)");
        Console.WriteLine($"  Evolution (mutation/crossover): {totalEvolutionMs}ms ({100.0 * totalEvolutionMs / totalMs:F1}%)");
    }

    private static void RenderTrack(SimpleCarWorld world, RaylibVector2 cameraOffset)
    {
        // Draw walls in white
        foreach (var lineSegment in world.WallGrid.LineSegments)
        {
            RaylibVector2 start = WorldToScreen(lineSegment.Start, cameraOffset);
            RaylibVector2 end = WorldToScreen(lineSegment.End, cameraOffset);
            Raylib.DrawLineV(start, end, RaylibColor.White);
        }

        // Draw start line in blue
        RaylibVector2 startBegin = WorldToScreen(world.Start.Start, cameraOffset);
        RaylibVector2 startEnd = WorldToScreen(world.Start.End, cameraOffset);
        Raylib.DrawLineEx(startBegin, startEnd, 3f, RaylibColor.SkyBlue);

        // Draw finish line in green
        RaylibVector2 finishBegin = WorldToScreen(world.Finish.Start, cameraOffset);
        RaylibVector2 finishEnd = WorldToScreen(world.Finish.End, cameraOffset);
        Raylib.DrawLineEx(finishBegin, finishEnd, 3f, RaylibColor.Lime);

        // Draw progress markers (every 10th)
        for (int i = 0; i < world.ProgressMarkers.Count; i += 10)
        {
            var marker = world.ProgressMarkers[i];
            RaylibVector2 screenPos = WorldToScreen(marker.Position, cameraOffset);
            Raylib.DrawCircleV(screenPos, 2f, new RaylibColor(100, 100, 100, 100));
        }
    }

    private static void RenderCar(GodotVector2 position, float heading, RaylibVector2 cameraOffset, RaylibColor color)
    {
        RaylibVector2 screenPos = WorldToScreen(position, cameraOffset);

        // Draw car body (circle)
        Raylib.DrawCircleV(screenPos, SimpleCar.Radius * Scale, color);

        // Draw heading indicator (nose) in white
        RaylibVector2 nose = new RaylibVector2(
            position.X + MathF.Cos(heading) * SimpleCar.Radius,
            position.Y + MathF.Sin(heading) * SimpleCar.Radius
        );
        RaylibVector2 screenNose = WorldToScreen(nose, cameraOffset);
        Raylib.DrawLineV(screenPos, screenNose, RaylibColor.White);
    }

    private static void RenderUI(int generation, float bestFitness, int step, int activeCars, bool isSimulating, TimeSpan elapsed)
    {
        int y = 10;
        int lineHeight = 25;

        // Format elapsed time as mm:ss
        string timeString = $"{(int)elapsed.TotalMinutes}:{elapsed.Seconds:D2}";

        Raylib.DrawText($"Generation: {generation} | Time: {timeString}", 10, y, 20, RaylibColor.White);
        y += lineHeight;

        Raylib.DrawText($"Best Fitness: {bestFitness:F3} ({bestFitness * 100:F0}% of track)", 10, y, 20, RaylibColor.White);
        y += lineHeight;

        Raylib.DrawText($"Step: {step} / 320", 10, y, 20, RaylibColor.White);
        y += lineHeight;

        Raylib.DrawText($"Active agents: {activeCars} / 1600", 10, y, 18, RaylibColor.White);
        y += lineHeight;

        if (isSimulating)
        {
            Raylib.DrawText("Status: Simulating MAX SPEED...", 10, y, 20, RaylibColor.Yellow);
        }
        else
        {
            Raylib.DrawText("Status: Ready for next generation", 10, y, 20, RaylibColor.Lime);
        }
    }

    private static RaylibVector2 WorldToScreen(GodotVector2 worldPos, RaylibVector2 cameraOffset)
    {
        return new RaylibVector2(
            cameraOffset.X + worldPos.X * Scale,
            cameraOffset.Y - worldPos.Y * Scale
        );
    }

    private static RaylibVector2 WorldToScreen(RaylibVector2 worldPos, RaylibVector2 cameraOffset)
    {
        return new RaylibVector2(
            cameraOffset.X + worldPos.X * Scale,
            cameraOffset.Y - worldPos.Y * Scale
        );
    }

    private static SpeciesSpec CreateCorridorTopology()
    {
        var topology = new SpeciesSpec
        {
            RowCounts = new[] { 1, 9, 12, 2 },
            AllowedActivationsPerRow = new uint[]
            {
                0b00000000001,
                0b11111111111,
                0b11111111111,
                0b00000000011
            },
            MaxInDegree = 12,
            Edges = new List<(int, int)>()
        };

        for (int src = 0; src < 10; src++)
        {
            for (int dst = 10; dst < 22; dst++)
            {
                topology.Edges.Add((src, dst));
            }
        }

        for (int src = 10; src < 22; src++)
        {
            topology.Edges.Add((src, 22));
            topology.Edges.Add((src, 23));
        }

        topology.BuildRowPlans();
        return topology;
    }
}
