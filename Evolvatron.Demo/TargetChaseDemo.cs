using Evolvatron.Core;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Raylib_cs;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Visual demo showing a GRID of AI-controlled rockets chasing targets.
/// Each rocket is controlled by a different neural network from the population.
/// Evolution happens in real-time - watch them improve!
///
/// Controls:
///   SPACE - Pause/Resume
///   R - Reset evolution
///   +/- - Change simulation speed
///   E - Force evolution step
/// </summary>
public static class TargetChaseDemo
{
    private const int ScreenWidth = 1920;
    private const int ScreenHeight = 1080;

    // Grid layout - 30 rockets!
    private const int GridCols = 6;
    private const int GridRows = 5;
    private const int RocketCount = GridCols * GridRows; // 30 rockets

    // Cell size
    private static readonly int CellWidth = ScreenWidth / GridCols;
    private static readonly int CellHeight = ScreenHeight / GridRows;
    private const float MetersToPixels = 18f; // Smaller scale to fit cells

    public static void Run()
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - AI Target Chase (Population View)");
        Raylib.SetTargetFPS(60);

        // Evolution setup - LARGE population with parallel evaluation
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 5,             // 5 species competing
            IndividualsPerSpecies = 200,  // 200 each = 1000 total!
            MinSpeciesCount = 3,
            Elites = 10,
            TournamentSize = 5,
            GraceGenerations = 5,
            StagnationThreshold = 30
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

        // Create environments and state for each displayed rocket
        var environments = new TargetChaseEnvironment[RocketCount];
        var individuals = new Individual[RocketCount];
        var individualIndices = new int[RocketCount]; // Track which individual from population

        for (int i = 0; i < RocketCount; i++)
        {
            environments[i] = new TargetChaseEnvironment();
            environments[i].MaxSteps = 400;
        }

        // Background evaluation environment
        var evalEnv = new TargetChaseEnvironment();
        evalEnv.MaxSteps = 300;

        // State
        int generation = 0;
        float bestFitness = float.NegativeInfinity;
        int bestTargetsHit = 0;
        int totalTargetsThisGen = 0;
        bool paused = false;
        int simulationSpeed = 2;
        float episodeTime = 0f;
        int episodeSeed = 0;

        // Initialize
        EvaluatePopulationBackground();
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
                bestTargetsHit = 0;
                population = evolver.InitializePopulation(evolutionConfig, topology);
                EvaluatePopulationBackground();
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

            // Simulation
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

                // Check if all rockets are done
                if (allTerminated)
                {
                    // Count total targets hit
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
            DrawGlobalUI(generation, bestFitness, bestTargetsHit, totalTargetsThisGen, episodeTime, paused, simulationSpeed, totalPop, evolutionConfig.SpeciesCount);

            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();

        // Local functions
        void EvaluatePopulationBackground()
        {
            // Parallel evaluation of all individuals across all species
            var allIndividuals = new List<(Species species, int idx, Individual ind)>();
            foreach (var species in population.AllSpecies)
            {
                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    allIndividuals.Add((species, i, species.Individuals[i]));
                }
            }

            // Parallel evaluation using all CPU cores
            var results = new float[allIndividuals.Count];
            Parallel.For(0, allIndividuals.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                evalIndex =>
                {
                    var (species, idx, ind) = allIndividuals[evalIndex];

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

            // Apply results back
            for (int i = 0; i < allIndividuals.Count; i++)
            {
                var (species, idx, ind) = allIndividuals[i];
                ind.Fitness = results[i];
                species.Individuals[idx] = ind;

                if (ind.Fitness > bestFitness)
                    bestFitness = ind.Fitness;
            }
        }

        void EvolveGeneration()
        {
            evolver.StepGeneration(population);
            generation++;
            EvaluatePopulationBackground();
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

        // Draw arena bounds (scaled)
        float scaleX = (cellW - 20) / (TargetChaseEnvironment.ArenaHalfWidth * 2);
        float scaleY = (cellH - 40) / (TargetChaseEnvironment.ArenaHalfHeight - TargetChaseEnvironment.GroundY);
        float scale = MathF.Min(scaleX, scaleY) * 0.9f;

        Vector2 WorldToCell(float wx, float wy)
        {
            float sx = centerX + wx * scale;
            float sy = centerY - (wy - 1f) * scale; // Shift up a bit
            return new Vector2(sx, sy);
        }

        // Draw ground
        var groundLeft = WorldToCell(-TargetChaseEnvironment.ArenaHalfWidth, TargetChaseEnvironment.GroundY);
        var groundRight = WorldToCell(TargetChaseEnvironment.ArenaHalfWidth, TargetChaseEnvironment.GroundY);
        Raylib.DrawLineEx(groundLeft, groundRight, 2f, Color.Gray);

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

    private static void DrawGlobalUI(int generation, float bestFitness, int bestTargetsHit, int totalTargets,
        float episodeTime, bool paused, int speed, int popSize, int speciesCount)
    {
        // Semi-transparent overlay at top
        Raylib.DrawRectangle(0, 0, ScreenWidth, 35, new Color(0, 0, 0, 180));

        // Stats
        Raylib.DrawText($"Gen: {generation}", 10, 8, 20, Color.White);
        Raylib.DrawText($"Best: {bestFitness:F0}", 130, 8, 20, Color.Green);
        Raylib.DrawText($"Targets: {bestTargetsHit}", 280, 8, 20, Color.Yellow);
        Raylib.DrawText($"Pop: {popSize} ({speciesCount} species)", 420, 8, 20, new Color(0, 255, 255, 255));
        Raylib.DrawText($"Speed: {speed}x", 650, 8, 20, Color.LightGray);
        Raylib.DrawText($"Time: {episodeTime:F1}s", 760, 8, 20, Color.LightGray);

        if (paused)
        {
            Raylib.DrawText("PAUSED", ScreenWidth / 2 - 50, 8, 24, Color.Red);
        }

        // Controls hint
        Raylib.DrawText("SPACE:Pause  R:Reset  E:Evolve  +/-:Speed", ScreenWidth - 400, 8, 16, Color.Gray);

        // FPS
        Raylib.DrawText($"FPS:{Raylib.GetFPS()}", ScreenWidth - 80, 8, 16, Color.Green);
    }
}
