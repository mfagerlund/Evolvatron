using Evolvatron.Core;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;
using Raylib_cs;
using System.Diagnostics;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Neuroevolution lander demo — counterpart to LunarLanderDemo (LM trajectory optimization).
/// Evolves neural controllers for closed-loop rocket landing using Evolvion.
///
/// Phase 1: Headless evolution with console output (fast, no rendering)
///   - Uses GPU batched evaluation if available (1000+ parallel rockets on GPU)
///   - Falls back to CPU Parallel.For if no GPU
/// Phase 2: Visual grid replay of best evolved controllers
///
/// Controls (visual phase):
///   SPACE - Pause/Resume
///   R     - Reset evolution
///   +/-   - Change simulation speed
///   E     - Force evolution step
/// </summary>
public static class EvolutionLanderDemo
{
    private const int ScreenWidth = 1920;
    private const int ScreenHeight = 1080;
    private const int GridCols = 5;
    private const int GridRows = 3;
    private const int CellCount = GridCols * GridRows;
    private static readonly int CellWidth = ScreenWidth / GridCols;
    private static readonly int CellHeight = ScreenHeight / GridRows;
    private const int MaxTrailPoints = 600;

    private const int MaxGenerations = 500;
    private const int MaxSteps = 600; // 5.0s at 120Hz
    private const int EvalSeeds = 5; // Average fitness over N seeds for robustness
    private const int TestSeeds = 20; // Seeds for best-individual test
    private const int TestInterval = 10; // Test best individual every N generations

    public static void Run()
    {
        var sw = Stopwatch.StartNew();

        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 5,
            IndividualsPerSpecies = 1600,
            MinSpeciesCount = 3,
            Elites = 100,
            TournamentSize = 8,
            ParentPoolPercentage = 0.5f,
            GraceGenerations = 10000,
            StagnationThreshold = 10000,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.9f,
                WeightJitterStdDev = 0.12f,
                WeightReset = 0.08f,
                WeightL1Shrink = 0.02f,
                L1ShrinkFactor = 0.97f,
                ActivationSwap = 0.01f,
                NodeParamMutate = 0.03f,
                NodeParamStdDev = 0.1f,
            }
        };

        var evolver = new Evolver(seed: 42);
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var population = evolver.InitializePopulation(evolutionConfig, topology);
        int totalPop = evolutionConfig.SpeciesCount * evolutionConfig.IndividualsPerSpecies;

        // Try GPU evaluator first
        GPURocketLandingEvaluator? gpuEvaluator = null;
        bool useGPU = false;
        try
        {
            gpuEvaluator = new GPURocketLandingEvaluator(maxIndividuals: totalPop + 100);
            useGPU = true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  GPU unavailable, using CPU: {ex.Message}");
        }

        string evalMode = useGPU ? $"GPU ({gpuEvaluator!.Accelerator.Name})" : "CPU Parallel";

        Console.WriteLine($"\n--- Evolving neural controller (seed=42) ---");
        Console.WriteLine($"  Config: {evolutionConfig.SpeciesCount} species x {evolutionConfig.IndividualsPerSpecies} individuals = {totalPop} total");
        Console.WriteLine($"  Network: 8 -> 16(Tanh) -> 12(Tanh) -> 2(Tanh)");
        Console.WriteLine($"  Max steps: {MaxSteps} ({MaxSteps / 120f:F1}s at 120Hz), eval: {evalMode}, seeds: {EvalSeeds}");
        Console.WriteLine($"  Mutation: jitter={evolutionConfig.MutationRates.WeightJitterStdDev:F3}, " +
                          $"reset={evolutionConfig.MutationRates.WeightReset:F3}, " +
                          $"tournament={evolutionConfig.TournamentSize}");

        // Phase 1: Headless evolution — stop when best individual aces the test
        float bestFitness = float.NegativeInfinity;
        int finalGen = 0;

        for (int gen = 0; gen <= MaxGenerations; gen++)
        {
            finalGen = gen;
            int genLandings;
            if (useGPU)
                genLandings = EvaluatePopulationMultiSeedGPU(population, topology, gpuEvaluator!, gen, ref bestFitness);
            else
                genLandings = EvaluatePopulationCPU(population, topology, ref bestFitness);

            var stats = population.GetStatistics();

            Console.WriteLine($"  Gen {gen,3}: best={stats.BestFitness,7:F1}, mean={stats.MeanFitness,7:F1}, " +
                              $"landed={genLandings,4}/{totalPop}");

            // Test best individual periodically
            if (gen > 0 && gen % TestInterval == 0)
            {
                int testLandings = TestBestOnCPU(population, topology, gen);
                if (testLandings == TestSeeds)
                {
                    Console.WriteLine($"  ** SOLVED at gen {gen}: {testLandings}/{TestSeeds} landings!");
                    break;
                }
            }

            if (gen < MaxGenerations)
                evolver.StepGeneration(population);
        }

        sw.Stop();
        Console.WriteLine($"\n  Total time: {sw.ElapsedMilliseconds}ms, {finalGen} gens");

        // Final verbose test
        TestBestOnCPU(population, topology, finalGen, verbose: true);

        gpuEvaluator?.Dispose();
    }

    private static int EvaluatePopulationMultiSeedGPU(Population population, SpeciesSpec topology,
        GPURocketLandingEvaluator gpuEvaluator, int generation, ref float bestFitness)
    {
        var allIndividuals = new List<Individual>();
        var indexMap = new List<(Species species, int idx)>();

        foreach (var species in population.AllSpecies)
        {
            for (int i = 0; i < species.Individuals.Count; i++)
            {
                allIndividuals.Add(species.Individuals[i]);
                indexMap.Add((species, i));
            }
        }

        // Evaluate on multiple seeds and average fitness
        var totalFitness = new float[allIndividuals.Count];
        int totalLandings = 0;

        for (int s = 0; s < EvalSeeds; s++)
        {
            int seed = generation * 1000 + s;
            var (fitnessValues, landings) = gpuEvaluator.EvaluatePopulation(
                topology, allIndividuals, seed: seed, maxSteps: MaxSteps);

            for (int i = 0; i < allIndividuals.Count; i++)
                totalFitness[i] += fitnessValues[i];

            totalLandings += landings;
        }

        // Average and assign
        for (int i = 0; i < allIndividuals.Count; i++)
        {
            var (species, idx) = indexMap[i];
            var ind = species.Individuals[idx];
            ind.Fitness = totalFitness[i] / EvalSeeds;
            species.Individuals[idx] = ind;
            if (ind.Fitness > bestFitness)
                bestFitness = ind.Fitness;
        }

        return totalLandings / EvalSeeds; // Average landings per seed
    }

    private static int TestBestOnCPU(Population population, SpeciesSpec topology, int gen, bool verbose = false)
    {
        var best = population.GetBestIndividual();
        if (!best.HasValue) return 0;

        var (bestInd, _) = best.Value;
        var evaluator = new CPUEvaluator(topology);
        int testSeeds = verbose ? TestSeeds : TestSeeds;
        int cpuLandings = 0;
        float totalFitness = 0f;

        for (int s = 0; s < testSeeds; s++)
        {
            var testEnv = new RocketEnvironment();
            testEnv.MaxSteps = MaxSteps;
            testEnv.Reset(s * 7 + 13); // Different test seeds than training
            var obs = new float[8];
            while (!testEnv.IsTerminal())
            {
                testEnv.GetObservations(obs);
                var acts = evaluator.Evaluate(bestInd, obs);
                testEnv.Step(acts);
            }
            float fit = testEnv.GetFinalFitness();
            totalFitness += fit;
            if (testEnv.Landed) cpuLandings++;

            if (verbose)
            {
                testEnv.GetRocketState(out float fx, out float fy, out float fvx, out float fvy,
                    out float fux, out float fuy, out _);
                float tiltDeg = MathF.Atan2(fux, fuy) * 180f / MathF.PI;
                string status = testEnv.Landed ? "LANDED" : "CRASH";
                Console.WriteLine($"    seed={s * 7 + 13,3}: pos=({fx,6:F1},{fy,6:F1}) vel=({fvx,5:F1},{fvy,5:F1}) " +
                                  $"tilt={tiltDeg,5:F0}deg fit={fit,6:F1} {status}");
            }
        }

        Console.WriteLine($"  ** Test gen {gen}: {cpuLandings}/{testSeeds} landed, avg_fit={totalFitness / testSeeds:F1}");
        return cpuLandings;
    }

    private static int EvaluatePopulationCPU(Population population, SpeciesSpec topology, ref float bestFitness)
    {
        var allIndividuals = new List<(Species species, int idx, Individual ind)>();
        foreach (var species in population.AllSpecies)
            for (int i = 0; i < species.Individuals.Count; i++)
                allIndividuals.Add((species, i, species.Individuals[i]));

        var results = new float[allIndividuals.Count];
        var landings = new int[allIndividuals.Count];

        Parallel.For(0, allIndividuals.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            evalIndex =>
            {
                var (_, _, ind) = allIndividuals[evalIndex];
                var localEnv = new RocketEnvironment();
                localEnv.MaxSteps = MaxSteps;
                var localEvaluator = new CPUEvaluator(topology);
                var obs = new float[8];

                localEnv.Reset(evalIndex);
                while (!localEnv.IsTerminal())
                {
                    localEnv.GetObservations(obs);
                    var acts = localEvaluator.Evaluate(ind, obs);
                    localEnv.Step(acts);
                }
                results[evalIndex] = localEnv.GetFinalFitness();
                landings[evalIndex] = localEnv.Landed ? 1 : 0;
            });

        int totalLandings = 0;
        for (int i = 0; i < allIndividuals.Count; i++)
        {
            var (species, idx, ind) = allIndividuals[i];
            ind.Fitness = results[i];
            species.Individuals[idx] = ind;
            if (ind.Fitness > bestFitness)
                bestFitness = ind.Fitness;
            totalLandings += landings[i];
        }
        return totalLandings;
    }

    private static void RunVisualization(Population population, SpeciesSpec topology,
        EvolutionConfig evolutionConfig, Evolver evolver)
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Evolution Lander (Neuroevolution)");
        Raylib.SetTargetFPS(60);

        var networkEvaluator = new CPUEvaluator(topology);

        var environments = new RocketEnvironment[CellCount];
        var individuals = new Individual[CellCount];
        var speciesIndices = new int[CellCount];
        var trails = new List<(float x, float y)>[CellCount];
        var cellFitness = new float[CellCount];

        for (int i = 0; i < CellCount; i++)
        {
            environments[i] = new RocketEnvironment();
            environments[i].MaxSteps = MaxSteps;
            trails[i] = new List<(float x, float y)>(MaxTrailPoints);
        }

        int generation = MaxGenerations;
        float bestFitness = population.GetStatistics().BestFitness;
        bool paused = false;
        int simulationSpeed = 4;
        float episodeTime = 0f;
        int episodeSeed = 0;
        int landings = 0;
        int bestLandings = 0;

        StartNewEpisodes();

        while (!Raylib.WindowShouldClose())
        {
            if (Raylib.IsKeyPressed(KeyboardKey.Space))
                paused = !paused;

            if (Raylib.IsKeyPressed(KeyboardKey.Equal) || Raylib.IsKeyPressed(KeyboardKey.KpAdd))
                simulationSpeed = Math.Min(simulationSpeed + 1, 20);
            if (Raylib.IsKeyPressed(KeyboardKey.Minus) || Raylib.IsKeyPressed(KeyboardKey.KpSubtract))
                simulationSpeed = Math.Max(simulationSpeed - 1, 1);

            if (Raylib.IsKeyPressed(KeyboardKey.E))
            {
                evolver.StepGeneration(population);
                generation++;
                float dummy = float.NegativeInfinity;
                EvaluatePopulationCPU(population, topology, ref dummy);
                var stats = population.GetStatistics();
                Console.WriteLine($"  [Evo] Gen {generation,3}: best={stats.BestFitness,7:F1}, mean={stats.MeanFitness,7:F1}");
                StartNewEpisodes();
            }

            if (!paused)
            {
                bool allTerminated = true;
                var observations = new float[8];

                for (int step = 0; step < simulationSpeed; step++)
                {
                    for (int i = 0; i < CellCount; i++)
                    {
                        var env = environments[i];
                        if (!env.IsTerminal())
                        {
                            allTerminated = false;
                            env.GetObservations(observations);
                            var actions = networkEvaluator.Evaluate(individuals[i], observations);
                            env.Step(actions);

                            if (trails[i].Count < MaxTrailPoints)
                            {
                                env.GetRocketState(out float rx, out float ry, out _, out _, out _, out _, out _);
                                trails[i].Add((rx, ry));
                            }
                        }
                        else if (float.IsNaN(cellFitness[i]))
                        {
                            cellFitness[i] = env.GetFinalFitness();
                        }
                    }
                    episodeTime += 1f / 120f;
                }

                if (allTerminated)
                {
                    landings = 0;
                    for (int i = 0; i < CellCount; i++)
                    {
                        if (float.IsNaN(cellFitness[i]))
                            cellFitness[i] = environments[i].GetFinalFitness();
                        if (environments[i].Landed)
                            landings++;
                    }
                    if (landings > bestLandings)
                        bestLandings = landings;

                    // Auto-evolve: step generation and re-evaluate
                    evolver.StepGeneration(population);
                    generation++;
                    float dummy = float.NegativeInfinity;
                    EvaluatePopulationCPU(population, topology, ref dummy);
                    bestFitness = Math.Max(bestFitness, population.GetStatistics().BestFitness);

                    StartNewEpisodes();
                }
            }

            Raylib.BeginDrawing();
            Raylib.ClearBackground(new Color(10, 10, 30, 255));

            for (int i = 0; i < CellCount; i++)
            {
                int col = i % GridCols;
                int row = i / GridCols;
                DrawCell(environments[i], trails[i], col * CellWidth, row * CellHeight,
                    CellWidth, CellHeight, speciesIndices[i], cellFitness[i]);
            }

            for (int col = 1; col < GridCols; col++)
                Raylib.DrawLine(col * CellWidth, 0, col * CellWidth, ScreenHeight, new Color(40, 40, 60, 255));
            for (int row = 1; row < GridRows; row++)
                Raylib.DrawLine(0, row * CellHeight, ScreenWidth, row * CellHeight, new Color(40, 40, 60, 255));

            int totalPop = evolutionConfig.SpeciesCount * evolutionConfig.IndividualsPerSpecies;
            DrawHUD(generation, bestFitness, landings, bestLandings, episodeTime, paused, simulationSpeed, totalPop);

            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();

        void StartNewEpisodes()
        {
            episodeSeed++;
            episodeTime = 0f;

            var allIndividuals = new List<(Individual ind, int speciesIdx)>();
            for (int s = 0; s < population.AllSpecies.Count; s++)
            {
                var species = population.AllSpecies[s];
                for (int i = 0; i < species.Individuals.Count; i++)
                    allIndividuals.Add((species.Individuals[i], s));
            }

            var sorted = allIndividuals.OrderByDescending(x => x.ind.Fitness).ToList();

            for (int i = 0; i < CellCount; i++)
            {
                if (i < sorted.Count)
                {
                    individuals[i] = sorted[i].ind;
                    speciesIndices[i] = sorted[i].speciesIdx;
                }
                environments[i].Reset(episodeSeed * 100 + i);
                trails[i].Clear();
                cellFitness[i] = float.NaN;

                environments[i].GetRocketState(out float rx, out float ry, out _, out _, out _, out _, out _);
                trails[i].Add((rx, ry));
            }
        }
    }

    private static void DrawCell(RocketEnvironment env, List<(float x, float y)> trail,
        int cellX, int cellY, int cellW, int cellH, int speciesIdx, float fitness)
    {
        float groundSurface = env.GroundSurfaceY;

        float worldMinX = -12f;
        float worldMaxX = 12f;
        float worldMinY = groundSurface - 1f;
        float worldMaxY = 22f;

        float worldW = worldMaxX - worldMinX;
        float worldH = worldMaxY - worldMinY;

        float scaleX = (cellW - 10) / worldW;
        float scaleY = (cellH - 40) / worldH;
        float scale = MathF.Min(scaleX, scaleY);

        float centerScreenX = cellX + cellW / 2f;
        float centerScreenY = cellY + cellH / 2f + 10f;

        float worldCenterX = (worldMinX + worldMaxX) / 2f;
        float worldCenterY = (worldMinY + worldMaxY) / 2f;

        Vector2 WorldToCell(float wx, float wy)
        {
            float sx = centerScreenX + (wx - worldCenterX) * scale;
            float sy = centerScreenY - (wy - worldCenterY) * scale;
            return new Vector2(sx, sy);
        }

        // Ground line
        var gl = WorldToCell(-15f, groundSurface);
        var gr = WorldToCell(15f, groundSurface);
        Raylib.DrawLineEx(gl, gr, 2f, new Color(100, 100, 100, 255));

        // Landing pad
        float padLeft = env.PadX - env.PadHalfWidth;
        float padRight = env.PadX + env.PadHalfWidth;
        var pl = WorldToCell(padLeft, groundSurface);
        var pr = WorldToCell(padRight, groundSurface + 0.3f);
        float padW = pr.X - pl.X;
        float padH = pl.Y - pr.Y;
        Raylib.DrawRectangleV(new Vector2(pl.X, pr.Y), new Vector2(padW, padH), new Color(200, 60, 20, 200));

        // Trail
        if (trail.Count > 1)
        {
            for (int i = 1; i < trail.Count; i++)
            {
                float progress = (float)i / trail.Count;
                byte alpha = (byte)(60 + (int)(140 * progress));
                var p0 = WorldToCell(trail[i - 1].x, trail[i - 1].y);
                var p1 = WorldToCell(trail[i].x, trail[i].y);
                Raylib.DrawLineEx(p0, p1, 1.5f, new Color((byte)80, (byte)160, (byte)255, alpha));
            }
        }

        // Rocket — circle geoms matching LunarLanderDemo
        var world = env.World;
        var rocketIndices = env.RocketIndices;
        byte rocketAlpha = env.IsTerminal() ? (byte)120 : (byte)220;

        foreach (int idx in rocketIndices)
        {
            var rb = world.RigidBodies[idx];
            float cos = MathF.Cos(rb.Angle);
            float sin = MathF.Sin(rb.Angle);
            bool isBody = (idx == rocketIndices[0]);
            var color = isBody
                ? new Color((byte)220, (byte)220, (byte)240, rocketAlpha)
                : new Color((byte)180, (byte)180, (byte)180, rocketAlpha);

            for (int g = 0; g < rb.GeomCount; g++)
            {
                var geom = world.RigidBodyGeoms[rb.GeomStartIndex + g];
                float wx = rb.X + geom.LocalX * cos - geom.LocalY * sin;
                float wy = rb.Y + geom.LocalX * sin + geom.LocalY * cos;
                var sp = WorldToCell(wx, wy);
                float sr = geom.Radius * scale;
                Raylib.DrawCircleV(sp, sr, color);
                if (isBody)
                    Raylib.DrawCircleLinesV(sp, sr, new Color((byte)255, (byte)255, (byte)255, (byte)(rocketAlpha / 2)));
            }
        }

        // Thrust flame — rotated by gimbal angle to show engine vectoring
        float throttle = env.CurrentThrottle;
        if (!env.IsTerminal() && throttle > 0.01f)
        {
            var body = world.RigidBodies[rocketIndices[0]];
            float bodyCos = MathF.Cos(body.Angle);
            float bodySin = MathF.Sin(body.Angle);
            float halfBody = 1.5f * 0.5f;
            float botX = body.X - halfBody * bodyCos;
            float botY = body.Y - halfBody * bodySin;

            // Gimbal rotates flame direction relative to body axis
            float gimbal = env.CurrentGimbal;
            float flameAngle = body.Angle + gimbal * 0.5f; // ~30 deg max deflection
            float flameCos = MathF.Cos(flameAngle);
            float flameSin = MathF.Sin(flameAngle);

            float flameLen = throttle * 2.0f;
            float flameTipX = botX - flameCos * flameLen;
            float flameTipY = botY - flameSin * flameLen;
            float perpX = -flameSin * 0.3f * throttle;
            float perpY = flameCos * 0.3f * throttle;

            var tip = WorldToCell(flameTipX, flameTipY);
            var left = WorldToCell(botX + perpX, botY + perpY);
            var right = WorldToCell(botX - perpX, botY - perpY);

            byte fa1 = (byte)(180 * throttle);
            byte fa2 = (byte)(100 * throttle);
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)180, (byte)30, fa1));
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)100, (byte)20, fa2));
        }

        // Cell info
        Raylib.DrawText($"S{speciesIdx}", cellX + 5, cellY + 5, 14, Color.LightGray);
        if (!float.IsNaN(fitness))
        {
            Color fitColor = fitness > 30 ? Color.Green : (fitness > 15 ? Color.Yellow : Color.Red);
            Raylib.DrawText($"F:{fitness:F1}", cellX + 5, cellY + 20, 14, fitColor);
        }
    }

    private static void DrawHUD(int generation, float bestFitness, int landings, int bestLandings,
        float episodeTime, bool paused, int speed, int popSize)
    {
        Raylib.DrawRectangle(0, 0, ScreenWidth, 35, new Color(0, 0, 0, 180));

        Raylib.DrawText($"Gen: {generation}", 10, 8, 20, Color.White);
        Raylib.DrawText($"Best: {bestFitness:F0}", 130, 8, 20, Color.Green);
        Raylib.DrawText($"Pop: {popSize}", 310, 8, 20, new Color(0, 255, 255, 255));
        Raylib.DrawText($"Speed: {speed}x", 440, 8, 20, Color.LightGray);
        Raylib.DrawText($"Time: {episodeTime:F1}s", 550, 8, 20, Color.LightGray);
        Raylib.DrawText($"Landings: {landings}/{bestLandings}", 680, 8, 20, Color.Yellow);

        if (paused)
            Raylib.DrawText("PAUSED", ScreenWidth / 2 - 50, 8, 24, Color.Red);

        Raylib.DrawText("SPACE:Pause  E:Evolve  +/-:Speed", ScreenWidth - 350, 8, 16, Color.Gray);
        Raylib.DrawText($"FPS:{Raylib.GetFPS()}", ScreenWidth - 80, 8, 16, Color.Green);
    }
}
