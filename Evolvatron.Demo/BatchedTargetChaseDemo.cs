using Evolvatron.Core;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;
using Raylib_cs;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Visual demo showing a GRID of AI-controlled rockets evolved using GPU batched physics.
/// Each rocket is controlled by a different neural network from the population.
///
/// Key difference from TargetChaseDemo:
/// - ALL fitness evaluations happen on GPU via GPUBatchedFitnessEvaluator
/// - Only the TOP 30 performers are visualized (using CPU environments for display)
/// - This enables evaluating 1000+ individuals per generation efficiently
///
/// Controls:
///   SPACE - Pause/Resume
///   R - Reset evolution
///   +/- - Change simulation speed
///   E - Force evolution step
/// </summary>
public static class BatchedTargetChaseDemo
{
    private const int ScreenWidth = 1920;
    private const int ScreenHeight = 1080;

    // Grid layout - 30 rockets for visualization
    private const int GridCols = 6;
    private const int GridRows = 5;
    private const int RocketCount = GridCols * GridRows; // 30 rockets displayed

    // Cell size
    private static readonly int CellWidth = ScreenWidth / GridCols;
    private static readonly int CellHeight = ScreenHeight / GridRows;

    public static void Run()
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - GPU Batched Evolution (1000+ Population)");
        Raylib.SetTargetFPS(60);

        // Evolution setup - LARGE population with GPU evaluation
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 5,             // 5 species competing
            IndividualsPerSpecies = 200,  // 200 each = 1000 total!
            MinSpeciesCount = 5,          // Don't cull any species
            Elites = 10,
            TournamentSize = 5,
            GraceGenerations = 10000,     // Effectively disable species culling
            StagnationThreshold = 10000   // (avoids topology mismatch from diversification)
        };

        var evolver = new Evolver(seed: DateTime.Now.Millisecond);
        var topology = new SpeciesBuilder()
            .AddInputRow(8) // TargetChaseEnvironment.InputCount (dir, vel, up, dist, angVel)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var population = evolver.InitializePopulation(evolutionConfig, topology);
        var networkEvaluator = new CPUEvaluator(topology);

        // Create GPU fitness evaluator for batched evaluation
        GPUBatchedFitnessEvaluator? gpuEvaluator = null;
        bool gpuAvailable = false;
        string gpuStatus = "Initializing...";

        try
        {
            gpuEvaluator = new GPUBatchedFitnessEvaluator(maxIndividuals: 1500);
            gpuAvailable = true;
            gpuStatus = $"GPU: {gpuEvaluator.Accelerator.Name}";
        }
        catch (Exception ex)
        {
            gpuStatus = $"GPU unavailable: {ex.Message}";
            Console.WriteLine($"GPU initialization failed, falling back to CPU: {ex.Message}");
        }

        // Create environments and state for each displayed rocket (CPU visualization)
        var environments = new TargetChaseEnvironment[RocketCount];
        var individuals = new Individual[RocketCount];
        var individualIndices = new int[RocketCount]; // Track which individual from population

        for (int i = 0; i < RocketCount; i++)
        {
            environments[i] = new TargetChaseEnvironment();
            environments[i].MaxSteps = 400;
        }

        // State
        int generation = 0;
        float bestFitness = float.NegativeInfinity;
        float avgFitness = 0f;
        int bestTargetsHit = 0;
        int totalTargetsThisGen = 0;
        bool paused = false;
        int simulationSpeed = 2;
        float episodeTime = 0f;
        int episodeSeed = 0;
        double lastEvalTime = 0;
        int evaluationsPerSecond = 0;

        // Fitness history for graphing
        var bestFitnessHistory = new List<float>();
        var avgFitnessHistory = new List<float>();
        const int MaxHistoryPoints = 100;

        // Initialize
        EvaluatePopulationOnGPU();
        StartNewEpisodes();

        while (!Raylib.WindowShouldClose())
        {
            // Input
            if (Raylib.IsKeyPressed(KeyboardKey.Space))
                paused = !paused;

            if (Raylib.IsKeyPressed(KeyboardKey.R))
            {
                generation = 0;
                bestFitness = float.NegativeInfinity;
                avgFitness = 0f;
                bestTargetsHit = 0;
                bestFitnessHistory.Clear();
                avgFitnessHistory.Clear();
                population = evolver.InitializePopulation(evolutionConfig, topology);
                EvaluatePopulationOnGPU();
                StartNewEpisodes();
            }

            if (Raylib.IsKeyPressed(KeyboardKey.Equal) || Raylib.IsKeyPressed(KeyboardKey.KpAdd))
                simulationSpeed = Math.Min(simulationSpeed + 1, 10);

            if (Raylib.IsKeyPressed(KeyboardKey.Minus) || Raylib.IsKeyPressed(KeyboardKey.KpSubtract))
                simulationSpeed = Math.Max(simulationSpeed - 1, 1);

            if (Raylib.IsKeyPressed(KeyboardKey.E))
            {
                EvolveGeneration();
                StartNewEpisodes();
            }

            // Simulation (visualization only, actual fitness comes from GPU)
            if (!paused)
            {
                bool allTerminated = true;

                var observations = new float[8]; // Reuse buffer
                for (int step = 0; step < simulationSpeed; step++)
                {
                    for (int i = 0; i < RocketCount; i++)
                    {
                        var env = environments[i];
                        if (!env.IsTerminal())
                        {
                            allTerminated = false;

                            // Get observations
                            env.GetObservations(observations);

                            // Run neural network
                            var actions = networkEvaluator.Evaluate(individuals[i], observations);

                            // Step environment
                            env.Step(actions);
                        }
                    }
                    episodeTime += 1f / 120f;
                }

                // Check if all displayed rockets are done
                if (allTerminated)
                {
                    // Count total targets hit (for display)
                    totalTargetsThisGen = 0;
                    for (int i = 0; i < RocketCount; i++)
                    {
                        totalTargetsThisGen += environments[i].TargetsHit;
                        if (environments[i].TargetsHit > bestTargetsHit)
                            bestTargetsHit = environments[i].TargetsHit;
                    }

                    // Evolve and restart
                    EvolveGeneration();
                    StartNewEpisodes();
                }
            }

            // Rendering
            Raylib.BeginDrawing();
            Raylib.ClearBackground(new Color(15, 15, 25, 255));

            // Draw grid of rockets
            for (int i = 0; i < RocketCount; i++)
            {
                int col = i % GridCols;
                int row = i / GridCols;
                int cellX = col * CellWidth;
                int cellY = row * CellHeight;

                DrawCell(environments[i], cellX, cellY, CellWidth, CellHeight, i, individualIndices[i]);
            }

            // Draw grid lines
            for (int col = 1; col < GridCols; col++)
            {
                Raylib.DrawLine(col * CellWidth, 0, col * CellWidth, ScreenHeight, new Color(40, 40, 60, 255));
            }
            for (int row = 1; row < GridRows; row++)
            {
                Raylib.DrawLine(0, row * CellHeight, ScreenWidth, row * CellHeight, new Color(40, 40, 60, 255));
            }

            // Draw global UI overlay
            int totalPop = evolutionConfig.SpeciesCount * evolutionConfig.IndividualsPerSpecies;
            DrawGlobalUI(generation, bestFitness, avgFitness, bestTargetsHit, totalTargetsThisGen, episodeTime,
                paused, simulationSpeed, totalPop, evolutionConfig.SpeciesCount, gpuStatus, lastEvalTime, evaluationsPerSecond,
                bestFitnessHistory, avgFitnessHistory);

            Raylib.EndDrawing();
        }

        // Cleanup
        gpuEvaluator?.Dispose();
        Raylib.CloseWindow();

        // Local functions
        void EvaluatePopulationOnGPU()
        {
            var startTime = DateTime.Now;

            // Gather all individuals from all species
            var allIndividuals = new List<Individual>();
            foreach (var species in population.AllSpecies)
            {
                allIndividuals.AddRange(species.Individuals);
            }

            float[] fitnessValues;

            if (gpuAvailable && gpuEvaluator != null)
            {
                // GPU batched evaluation - evaluate ALL individuals in one GPU pass
                try
                {
                    fitnessValues = gpuEvaluator.EvaluatePopulation(
                        topology,
                        allIndividuals,
                        seed: episodeSeed);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"GPU evaluation failed, falling back to CPU: {ex.Message}");
                    gpuAvailable = false;
                    gpuStatus = "GPU failed, using CPU";
                    fitnessValues = EvaluateOnCPU(allIndividuals);
                }
            }
            else
            {
                // CPU fallback
                fitnessValues = EvaluateOnCPU(allIndividuals);
            }

            // Apply fitness values back to individuals and compute stats
            int idx = 0;
            float genBestFitness = float.NegativeInfinity;
            float genTotalFitness = 0f;
            foreach (var species in population.AllSpecies)
            {
                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = fitnessValues[idx++];
                    species.Individuals[i] = ind;

                    genTotalFitness += ind.Fitness;
                    if (ind.Fitness > genBestFitness)
                        genBestFitness = ind.Fitness;
                    if (ind.Fitness > bestFitness)
                        bestFitness = ind.Fitness;
                }
            }

            // Update average fitness
            avgFitness = genTotalFitness / allIndividuals.Count;

            // Add to history (limit size)
            bestFitnessHistory.Add(genBestFitness);
            avgFitnessHistory.Add(avgFitness);
            if (bestFitnessHistory.Count > MaxHistoryPoints)
            {
                bestFitnessHistory.RemoveAt(0);
                avgFitnessHistory.RemoveAt(0);
            }

            var elapsed = (DateTime.Now - startTime).TotalMilliseconds;
            lastEvalTime = elapsed;
            evaluationsPerSecond = (int)(allIndividuals.Count / (elapsed / 1000.0));
        }

        float[] EvaluateOnCPU(List<Individual> allIndividuals)
        {
            // Parallel CPU evaluation as fallback
            var results = new float[allIndividuals.Count];
            Parallel.For(0, allIndividuals.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                evalIndex =>
                {
                    var ind = allIndividuals[evalIndex];

                    // Thread-local environment and evaluator
                    var localEnv = new TargetChaseEnvironment();
                    localEnv.MaxSteps = 200; // Shorter for faster eval
                    var localEvaluator = new CPUEvaluator(topology);
                    var obs = new float[8];

                    // Evaluate on 2 seeds for robustness
                    float totalFitness = 0f;
                    for (int seed = 0; seed < 2; seed++)
                    {
                        localEnv.Reset(seed * 1000 + evalIndex);
                        while (!localEnv.IsTerminal())
                        {
                            localEnv.GetObservations(obs);
                            var acts = localEvaluator.Evaluate(ind, obs);
                            localEnv.Step(acts);
                        }
                        totalFitness += localEnv.GetFinalFitness();
                    }

                    results[evalIndex] = totalFitness / 2f;
                });

            return results;
        }

        void EvolveGeneration()
        {
            evolver.StepGeneration(population);
            generation++;
            episodeSeed++;
            EvaluatePopulationOnGPU();
        }

        void StartNewEpisodes()
        {
            episodeSeed++;
            episodeTime = 0f;

            // Gather all individuals from all species with their fitness
            var allIndividuals = new List<(Individual ind, int speciesIdx, int idx)>();
            for (int s = 0; s < population.AllSpecies.Count; s++)
            {
                var species = population.AllSpecies[s];
                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    allIndividuals.Add((species.Individuals[i], s, i));
                }
            }

            // Sort by fitness and pick top performers for display
            var sorted = allIndividuals
                .OrderByDescending(x => x.ind.Fitness)
                .ToList();

            for (int i = 0; i < RocketCount; i++)
            {
                if (i < sorted.Count)
                {
                    individuals[i] = sorted[i].ind;
                    individualIndices[i] = sorted[i].speciesIdx * 1000 + sorted[i].idx; // Encode species
                }

                environments[i].Reset(episodeSeed * 100 + i);
            }
        }
    }

    private static void DrawCell(TargetChaseEnvironment env, int cellX, int cellY, int cellW, int cellH, int index, int popIndex)
    {
        // Cell center
        float centerX = cellX + cellW / 2f;
        float centerY = cellY + cellH / 2f;

        // Draw arena bounds (scaled) - symmetric arena
        float arenaSize = TargetChaseEnvironment.ArenaHalfSize * 2;
        float scaleX = (cellW - 20) / arenaSize;
        float scaleY = (cellH - 40) / arenaSize;
        float scale = MathF.Min(scaleX, scaleY) * 0.9f;

        Vector2 WorldToCell(float wx, float wy)
        {
            float sx = centerX + wx * scale;
            float sy = centerY - wy * scale;
            return new Vector2(sx, sy);
        }

        // Draw arena border (no ground line for symmetric arena)
        var topLeft = WorldToCell(-TargetChaseEnvironment.ArenaHalfSize, TargetChaseEnvironment.ArenaHalfSize);
        var topRight = WorldToCell(TargetChaseEnvironment.ArenaHalfSize, TargetChaseEnvironment.ArenaHalfSize);
        var bottomLeft = WorldToCell(-TargetChaseEnvironment.ArenaHalfSize, -TargetChaseEnvironment.ArenaHalfSize);
        var bottomRight = WorldToCell(TargetChaseEnvironment.ArenaHalfSize, -TargetChaseEnvironment.ArenaHalfSize);
        Raylib.DrawLineEx(topLeft, topRight, 1f, Color.DarkGray);
        Raylib.DrawLineEx(bottomLeft, bottomRight, 1f, Color.DarkGray);
        Raylib.DrawLineEx(topLeft, bottomLeft, 1f, Color.DarkGray);
        Raylib.DrawLineEx(topRight, bottomRight, 1f, Color.DarkGray);

        // Draw target
        var (tx, ty) = env.TargetPosition;
        var targetPos = WorldToCell(tx, ty);
        float targetR = 1.0f * scale;

        // Pulsing glow
        float pulse = 1f + 0.15f * MathF.Sin((float)Raylib.GetTime() * 6f + index);
        Raylib.DrawCircleV(targetPos, targetR * pulse, new Color(255, 200, 0, 100));
        Raylib.DrawCircleV(targetPos, targetR * 0.7f * pulse, new Color(255, 220, 50, 180));
        Raylib.DrawCircleV(targetPos, targetR * 0.3f, Color.White);

        // Draw rocket
        env.GetRocketState(out float rx, out float ry, out float vx, out float vy, out float upX, out float upY, out float angle);
        var rocketPos = WorldToCell(rx, ry);

        // Rocket body (simple triangle)
        float rocketSize = 0.8f * scale;
        float cosA = MathF.Cos(angle);
        float sinA = MathF.Sin(angle);

        // Triangle points (pointing in direction of angle)
        Vector2 nose = new Vector2(rocketPos.X + cosA * rocketSize, rocketPos.Y - sinA * rocketSize);
        Vector2 left = new Vector2(rocketPos.X - cosA * rocketSize * 0.5f - sinA * rocketSize * 0.4f,
                                    rocketPos.Y + sinA * rocketSize * 0.5f - cosA * rocketSize * 0.4f);
        Vector2 right = new Vector2(rocketPos.X - cosA * rocketSize * 0.5f + sinA * rocketSize * 0.4f,
                                     rocketPos.Y + sinA * rocketSize * 0.5f + cosA * rocketSize * 0.4f);

        // Color based on whether it's terminated
        Color rocketColor = env.IsTerminal() ? new Color(100, 100, 100, 150) : new Color(100, 180, 255, 255);
        Raylib.DrawTriangle(nose, right, left, rocketColor);
        Raylib.DrawTriangleLines(nose, right, left, Color.White);

        // Draw thrust flame
        if (!env.IsTerminal() && env.CurrentThrottle > 0.1f)
        {
            float flameLen = env.CurrentThrottle * rocketSize * 1.5f;
            Vector2 flameBase = new Vector2(rocketPos.X - cosA * rocketSize * 0.4f, rocketPos.Y + sinA * rocketSize * 0.4f);
            Vector2 flameTip = new Vector2(flameBase.X - cosA * flameLen, flameBase.Y + sinA * flameLen);

            // Flickering flame
            float flicker = 0.7f + 0.3f * MathF.Sin((float)Raylib.GetTime() * 30f + index * 7);
            Raylib.DrawLineEx(flameBase, flameTip, 4f * flicker, new Color(255, 150, 50, 200));
            Raylib.DrawLineEx(flameBase, flameTip, 2f * flicker, new Color(255, 255, 100, 255));
        }

        // Cell info - show species and individual
        int speciesIdx = popIndex / 1000;
        int indIdx = popIndex % 1000;
        Raylib.DrawText($"S{speciesIdx}#{indIdx}", cellX + 5, cellY + 5, 12, Color.LightGray);
        Raylib.DrawText($"Tgt:{env.TargetsHit}", cellX + 5, cellY + 18, 12,
            env.TargetsHit > 0 ? Color.Green : Color.Gray);
    }

    private static void DrawGlobalUI(int generation, float bestFitness, float avgFitness, int bestTargetsHit, int totalTargets,
        float episodeTime, bool paused, int speed, int popSize, int speciesCount, string gpuStatus,
        double evalTime, int evalsPerSec, List<float> bestHistory, List<float> avgHistory)
    {
        // Semi-transparent overlay at top
        Raylib.DrawRectangle(0, 0, ScreenWidth, 55, new Color(0, 0, 0, 200));

        // Stats - Row 1
        Raylib.DrawText($"Gen: {generation}", 10, 6, 20, Color.White);
        Raylib.DrawText($"Best: {bestFitness:F0}", 130, 6, 20, Color.Green);
        Raylib.DrawText($"Avg: {avgFitness:F0}", 280, 6, 20, Color.Yellow);
        Raylib.DrawText($"Pop: {popSize}", 420, 6, 20, new Color(0, 255, 255, 255));
        Raylib.DrawText($"Speed: {speed}x", 520, 6, 20, Color.LightGray);

        if (paused)
        {
            Raylib.DrawText("PAUSED", ScreenWidth / 2 - 50, 6, 24, Color.Red);
        }

        // Stats - Row 2 (GPU info)
        Raylib.DrawText(gpuStatus, 10, 32, 16, new Color(100, 200, 255, 255));
        Raylib.DrawText($"Eval: {evalTime:F0}ms ({evalsPerSec}/s)", 400, 32, 16, Color.LightGray);

        // Controls hint
        Raylib.DrawText("SPACE:Pause  R:Reset  E:Evolve  +/-:Speed", ScreenWidth - 400, 32, 14, Color.Gray);

        // FPS
        Raylib.DrawText($"FPS:{Raylib.GetFPS()}", ScreenWidth - 80, 6, 16, Color.Green);

        // Mode indicator
        Raylib.DrawText("GPU BATCHED", ScreenWidth - 150, 6, 16, new Color(255, 200, 50, 255));

        // Draw fitness graph in bottom-right corner
        DrawFitnessGraph(bestHistory, avgHistory);
    }

    private static void DrawFitnessGraph(List<float> bestHistory, List<float> avgHistory)
    {
        if (bestHistory.Count < 2) return;

        // Graph dimensions
        const int graphWidth = 300;
        const int graphHeight = 150;
        int graphX = ScreenWidth - graphWidth - 20;
        int graphY = ScreenHeight - graphHeight - 20;

        // Background
        Raylib.DrawRectangle(graphX - 5, graphY - 25, graphWidth + 10, graphHeight + 35, new Color(0, 0, 0, 200));
        Raylib.DrawRectangleLines(graphX, graphY, graphWidth, graphHeight, Color.DarkGray);

        // Title
        Raylib.DrawText("Fitness History", graphX, graphY - 20, 16, Color.White);

        // Find min/max for scaling
        float minVal = float.MaxValue;
        float maxVal = float.MinValue;
        foreach (var v in bestHistory) { minVal = Math.Min(minVal, v); maxVal = Math.Max(maxVal, v); }
        foreach (var v in avgHistory) { minVal = Math.Min(minVal, v); maxVal = Math.Max(maxVal, v); }

        // Add some padding
        float range = maxVal - minVal;
        if (range < 100) range = 100;
        minVal -= range * 0.1f;
        maxVal += range * 0.1f;
        range = maxVal - minVal;

        // Draw axis labels
        Raylib.DrawText($"{maxVal:F0}", graphX - 40, graphY - 5, 12, Color.Gray);
        Raylib.DrawText($"{minVal:F0}", graphX - 40, graphY + graphHeight - 10, 12, Color.Gray);

        // Draw lines
        int pointCount = bestHistory.Count;
        float xStep = (float)graphWidth / Math.Max(pointCount - 1, 1);

        // Average fitness (yellow)
        for (int i = 1; i < avgHistory.Count; i++)
        {
            float x1 = graphX + (i - 1) * xStep;
            float y1 = graphY + graphHeight - ((avgHistory[i - 1] - minVal) / range * graphHeight);
            float x2 = graphX + i * xStep;
            float y2 = graphY + graphHeight - ((avgHistory[i] - minVal) / range * graphHeight);
            Raylib.DrawLineEx(new Vector2(x1, y1), new Vector2(x2, y2), 2f, Color.Yellow);
        }

        // Best fitness (green)
        for (int i = 1; i < bestHistory.Count; i++)
        {
            float x1 = graphX + (i - 1) * xStep;
            float y1 = graphY + graphHeight - ((bestHistory[i - 1] - minVal) / range * graphHeight);
            float x2 = graphX + i * xStep;
            float y2 = graphY + graphHeight - ((bestHistory[i] - minVal) / range * graphHeight);
            Raylib.DrawLineEx(new Vector2(x1, y1), new Vector2(x2, y2), 2f, Color.Green);
        }

        // Legend
        Raylib.DrawText("Best", graphX + graphWidth - 80, graphY + 5, 12, Color.Green);
        Raylib.DrawText("Avg", graphX + graphWidth - 40, graphY + 5, 12, Color.Yellow);
    }
}
