using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using Raylib_cs;
using System.Diagnostics;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Obstacle lander demo — evolved controllers navigating a funnel with sensor rays.
///
/// Phase 1: Headless GPU evolution (10K pop, 12-input NN with 4 distance sensors)
/// Phase 2: Visual 5x3 grid replay using GPU step-by-step (identical physics to training)
///
/// Controls (visual phase):
///   SPACE - Pause/Resume
///   R     - Reset episodes
///   +/-   - Change simulation speed
/// </summary>
public static class ObstacleLanderDemo
{
    private const int ScreenWidth = 1920;
    private const int ScreenHeight = 1080;
    private const int GridCols = 5;
    private const int GridRows = 3;
    private const int CellCount = GridCols * GridRows;
    private static readonly int CellWidth = ScreenWidth / GridCols;
    private static readonly int CellHeight = ScreenHeight / GridRows;
    private const int MaxTrailPoints = 600;
    private const int MaxSteps = 900;
    private const int EvalRounds = 3;
    private const int SensorCount = 4;
    private const int BodiesPerWorld = 3;
    private const int GeomsPerWorld = 19;

    private const float WallAngleDeg = 25f;
    private const float WallAngle = WallAngleDeg * MathF.PI / 180f;
    private const float GroundY = -5f;
    private const float GroundSurfaceY = GroundY + 0.5f;
    private const float PadX = 0f;
    private const float PadHalfWidth = 2f;

    private static readonly OBBCollider[] ObstacleOBBs =
    {
        OBBCollider.FromAngle(-8f, 8f, 4f, 0.3f, WallAngle),
        OBBCollider.FromAngle(8f, 8f, 4f, 0.3f, -WallAngle),
    };

    private static readonly GPUOBBCollider[] GpuObstacles =
    {
        new GPUOBBCollider
        {
            CX = -8f, CY = 8f,
            UX = MathF.Cos(WallAngle), UY = MathF.Sin(WallAngle),
            HalfExtentX = 4f, HalfExtentY = 0.3f
        },
        new GPUOBBCollider
        {
            CX = 8f, CY = 8f,
            UX = MathF.Cos(-WallAngle), UY = MathF.Sin(-WallAngle),
            HalfExtentX = 4f, HalfExtentY = 0.3f
        },
    };

    public static void Run()
    {
        var sw = Stopwatch.StartNew();

        // --- Phase 1: Headless GPU evolution ---
        int popSize = 10000;
        int inputSize = 8 + SensorCount;

        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 1,
            IndividualsPerSpecies = popSize,
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

        var topology = new SpeciesBuilder()
            .AddInputRow(inputSize)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(evolutionConfig, topology);

        Console.WriteLine("\n--- Obstacle Lander: GPU Evolution Phase ---");
        Console.WriteLine($"  Pop: {popSize}, Network: {inputSize} -> 16 -> 12 -> 2");
        Console.WriteLine($"  Obstacles: {GpuObstacles.Length}, Sensors: {SensorCount}");
        Console.WriteLine($"  MaxSteps: {MaxSteps}, EvalRounds: {EvalRounds}");

        GPURocketLandingMegaEvaluator? megaEval = null;
        try
        {
            megaEval = new GPURocketLandingMegaEvaluator(maxIndividuals: popSize + 100);
            megaEval.SpawnXMin = 3f;
            megaEval.SpawnXRange = 8f;
            megaEval.SpawnAngleRange = 0.3f;
            megaEval.InitialVelXRange = 2f;
            megaEval.InitialVelYMax = 2f;
            megaEval.MaxThrust = 130f;
            megaEval.MaxLandingAngle = 8f * MathF.PI / 180f;
            megaEval.Obstacles = new List<GPUOBBCollider>(GpuObstacles);
            megaEval.SensorCount = SensorCount;
            megaEval.ObstacleDeathEnabled = true;
            megaEval.WagglePenalty = 0.01f;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  GPU unavailable: {ex.Message}");
            Console.WriteLine("  This demo requires a GPU. Exiting.");
            return;
        }

        Console.WriteLine($"  GPU: {megaEval.Accelerator.Name}");

        // Warmup
        var allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
        megaEval.EvaluatePopulation(topology, allIndividuals, seed: 99, maxSteps: MaxSteps);

        const int timeBudgetSeconds = 60;
        var evolSw = Stopwatch.StartNew();
        int generation = 0;
        float bestFitness = float.NegativeInfinity;
        int bestLandings = 0;

        while (evolSw.Elapsed.TotalSeconds < timeBudgetSeconds)
        {
            allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
            int n = allIndividuals.Count;
            var avgFitness = new float[n];
            int totalLandings = 0;

            for (int k = 0; k < EvalRounds; k++)
            {
                var (fitness, landings, _) = megaEval.EvaluatePopulation(
                    topology, allIndividuals, seed: generation * EvalRounds + k, maxSteps: MaxSteps);
                for (int i = 0; i < n; i++)
                    avgFitness[i] += fitness[i];
                totalLandings += landings;
            }
            for (int i = 0; i < n; i++)
                avgFitness[i] /= EvalRounds;

            int idx = 0;
            foreach (var species in population.AllSpecies)
            {
                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = avgFitness[idx++];
                    species.Individuals[i] = ind;
                }
            }

            float genBest = avgFitness.Max();
            if (genBest > bestFitness) bestFitness = genBest;
            if (totalLandings > bestLandings) bestLandings = totalLandings;

            if (generation % 5 == 0)
            {
                float landRate = 100f * totalLandings / (popSize * EvalRounds);
                Console.WriteLine($"  Gen {generation,3}: best={genBest,7:F1}, land={totalLandings,5} ({landRate:F1}%), " +
                                  $"t={evolSw.Elapsed.TotalSeconds:F1}s");
            }

            evolver.StepGeneration(population);
            generation++;
        }

        sw.Stop();
        Console.WriteLine($"\n  Evolution complete: {generation} gens in {evolSw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"  Best fitness: {bestFitness:F1}, Peak landings: {bestLandings}");

        // Final evaluation to rank individuals by landing ability
        allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
        int finalN = allIndividuals.Count;
        var finalAvgFitness = new float[finalN];
        var finalLandCount = new int[finalN];
        for (int k = 0; k < EvalRounds; k++)
        {
            var (fitness, _, landed) = megaEval.EvaluatePopulation(
                topology, allIndividuals, seed: 999 + k, maxSteps: MaxSteps);
            for (int i = 0; i < finalN; i++)
            {
                finalAvgFitness[i] += fitness[i];
                if (landed[i]) finalLandCount[i]++;
            }
        }
        for (int i = 0; i < finalN; i++)
            finalAvgFitness[i] /= EvalRounds;

        // Select top CellCount individuals: landers first, then by fitness
        var ranked = allIndividuals
            .Select((ind, i) => (ind, lands: finalLandCount[i], fit: finalAvgFitness[i]))
            .OrderByDescending(x => x.lands)
            .ThenByDescending(x => x.fit)
            .Take(CellCount)
            .Select(x => x.ind)
            .ToList();

        // --- Phase 2: Visual replay using GPU (identical physics) ---
        Console.WriteLine($"\n  Starting GPU visual replay (top-{CellCount} controllers)...");
        RunGPUVisualization(megaEval, topology, ranked, bestFitness);
        megaEval.Dispose();
    }

    private static void RunGPUVisualization(
        GPURocketLandingMegaEvaluator megaEval,
        SpeciesSpec topology,
        List<Individual> vizIndividuals,
        float bestFitness)
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Obstacle Lander (GPU Replay)");
        Raylib.SetTargetFPS(60);

        var trails = new List<(float x, float y)>[CellCount];
        for (int i = 0; i < CellCount; i++)
            trails[i] = new List<(float x, float y)>(MaxTrailPoints);

        var terminal = new byte[CellCount];
        var landed = new byte[CellCount];
        var throttle = new float[CellCount];
        var gimbal = new float[CellCount];
        GPURigidBody[]? bodies = null;
        GPURigidBodyGeom[]? geoms = null;

        bool paused = false;
        int simulationSpeed = 4;
        float episodeTime = 0f;
        int episodeSeed = 0;
        int landingCount = 0;

        PrepareEpisode();

        while (!Raylib.WindowShouldClose())
        {
            if (Raylib.IsKeyPressed(KeyboardKey.Space))
                paused = !paused;
            if (Raylib.IsKeyPressed(KeyboardKey.Equal) || Raylib.IsKeyPressed(KeyboardKey.KpAdd))
                simulationSpeed = Math.Min(simulationSpeed + 1, 20);
            if (Raylib.IsKeyPressed(KeyboardKey.Minus) || Raylib.IsKeyPressed(KeyboardKey.KpSubtract))
                simulationSpeed = Math.Max(simulationSpeed - 1, 1);
            if (Raylib.IsKeyPressed(KeyboardKey.R))
                PrepareEpisode();

            if (!paused)
            {
                for (int step = 0; step < simulationSpeed; step++)
                {
                    // Check if all terminated
                    bool allDone = true;
                    for (int i = 0; i < CellCount; i++)
                        if (terminal[i] == 0) { allDone = false; break; }

                    if (allDone)
                    {
                        landingCount = 0;
                        for (int i = 0; i < CellCount; i++)
                            if (landed[i] != 0) landingCount++;
                        PrepareEpisode();
                        break;
                    }

                    megaEval.StepVisualization();
                    megaEval.ReadVisualizationState(out bodies, out geoms, terminal, landed, throttle, gimbal);

                    // Update trails from COM
                    for (int i = 0; i < CellCount; i++)
                    {
                        if (terminal[i] != 0 || trails[i].Count >= MaxTrailPoints) continue;
                        float comX = 0, comY = 0, totalMass = 0;
                        for (int b = 0; b < BodiesPerWorld; b++)
                        {
                            var body = bodies![i * BodiesPerWorld + b];
                            if (body.InvMass > 0)
                            {
                                float mass = 1f / body.InvMass;
                                comX += body.X * mass;
                                comY += body.Y * mass;
                                totalMass += mass;
                            }
                        }
                        if (totalMass > 0) { comX /= totalMass; comY /= totalMass; }
                        trails[i].Add((comX, comY));
                    }
                    episodeTime += 1f / 120f;
                }
            }

            Raylib.BeginDrawing();
            Raylib.ClearBackground(new Color(10, 10, 30, 255));

            if (bodies != null && geoms != null)
            {
                for (int i = 0; i < CellCount; i++)
                {
                    int col = i % GridCols;
                    int row = i / GridCols;
                    DrawCellGPU(i, bodies, geoms, terminal[i] != 0, landed[i] != 0,
                        throttle[i], gimbal[i], trails[i],
                        col * CellWidth, row * CellHeight, CellWidth, CellHeight);
                }
            }

            for (int c = 1; c < GridCols; c++)
                Raylib.DrawLine(c * CellWidth, 0, c * CellWidth, ScreenHeight, new Color(40, 40, 60, 255));
            for (int r = 1; r < GridRows; r++)
                Raylib.DrawLine(0, r * CellHeight, ScreenWidth, r * CellHeight, new Color(40, 40, 60, 255));

            DrawHUD(bestFitness, landingCount, episodeTime, paused, simulationSpeed);
            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();

        void PrepareEpisode()
        {
            episodeSeed++;
            episodeTime = 0f;
            for (int i = 0; i < CellCount; i++)
                trails[i].Clear();
            Array.Clear(terminal);
            Array.Clear(landed);

            megaEval.PrepareVisualization(topology, vizIndividuals, episodeSeed * 100, MaxSteps);
            megaEval.ReadVisualizationState(out bodies, out geoms, terminal, landed, throttle, gimbal);

            // Initial trail point
            for (int i = 0; i < CellCount; i++)
            {
                float comX = 0, comY = 0, totalMass = 0;
                for (int b = 0; b < BodiesPerWorld; b++)
                {
                    var body = bodies![i * BodiesPerWorld + b];
                    if (body.InvMass > 0)
                    {
                        float mass = 1f / body.InvMass;
                        comX += body.X * mass;
                        comY += body.Y * mass;
                        totalMass += mass;
                    }
                }
                if (totalMass > 0) { comX /= totalMass; comY /= totalMass; }
                trails[i].Add((comX, comY));
            }
        }
    }

    private static void DrawCellGPU(int worldIdx,
        GPURigidBody[] bodies, GPURigidBodyGeom[] geoms,
        bool isTerminal, bool hasLanded, float curThrottle, float curGimbal,
        List<(float x, float y)> trail,
        int cellX, int cellY, int cellW, int cellH)
    {
        float worldMinX = -18f, worldMaxX = 18f;
        float worldMinY = GroundSurfaceY - 1f, worldMaxY = 22f;
        float worldW = worldMaxX - worldMinX;
        float worldH = worldMaxY - worldMinY;
        float scaleX = (cellW - 10) / worldW;
        float scaleY = (cellH - 40) / worldH;
        float scale = MathF.Min(scaleX, scaleY);
        float centerScreenX = cellX + cellW / 2f;
        float centerScreenY = cellY + cellH / 2f + 10f;
        float worldCenterX = (worldMinX + worldMaxX) / 2f;
        float worldCenterY = (worldMinY + worldMaxY) / 2f;

        Vector2 W2S(float wx, float wy)
        {
            return new Vector2(
                centerScreenX + (wx - worldCenterX) * scale,
                centerScreenY - (wy - worldCenterY) * scale);
        }

        // Ground
        Raylib.DrawLineEx(W2S(-20f, GroundSurfaceY), W2S(20f, GroundSurfaceY), 2f, new Color(100, 100, 100, 255));

        // Landing pad
        var pl = W2S(PadX - PadHalfWidth, GroundSurfaceY);
        var pr = W2S(PadX + PadHalfWidth, GroundSurfaceY + 0.3f);
        Raylib.DrawRectangleV(new Vector2(pl.X, pr.Y), new Vector2(pr.X - pl.X, pl.Y - pr.Y),
            new Color(200, 60, 20, 200));

        // Obstacles
        foreach (var obs in ObstacleOBBs)
            DrawOBB(obs, W2S);

        // Trail
        if (trail.Count > 1)
        {
            for (int i = 1; i < trail.Count; i++)
            {
                byte alpha = (byte)(60 + (int)(140f * i / trail.Count));
                Raylib.DrawLineEx(W2S(trail[i - 1].x, trail[i - 1].y), W2S(trail[i].x, trail[i].y),
                    1.5f, new Color((byte)80, (byte)160, (byte)255, alpha));
            }
        }

        // Rocket bodies + geoms
        int bodyBase = worldIdx * BodiesPerWorld;
        int geomBase = worldIdx * GeomsPerWorld;
        byte rocketAlpha = isTerminal ? (byte)120 : (byte)220;

        // COM for sensor rays
        float comX = 0, comY = 0, totalMass = 0;
        for (int b = 0; b < BodiesPerWorld; b++)
        {
            var body = bodies[bodyBase + b];
            if (body.InvMass > 0)
            {
                float mass = 1f / body.InvMass;
                comX += body.X * mass;
                comY += body.Y * mass;
                totalMass += mass;
            }
        }
        if (totalMass > 0) { comX /= totalMass; comY /= totalMass; }

        for (int b = 0; b < BodiesPerWorld; b++)
        {
            var body = bodies[bodyBase + b];
            float cos = MathF.Cos(body.Angle);
            float sin = MathF.Sin(body.Angle);
            bool isMain = (b == 0);
            var color = isMain
                ? new Color((byte)220, (byte)220, (byte)240, rocketAlpha)
                : new Color((byte)180, (byte)180, (byte)180, rocketAlpha);

            for (int g = 0; g < body.GeomCount; g++)
            {
                var geom = geoms[geomBase + body.GeomStartIndex + g];
                float wx = body.X + geom.LocalX * cos - geom.LocalY * sin;
                float wy = body.Y + geom.LocalX * sin + geom.LocalY * cos;
                var sp = W2S(wx, wy);
                float sr = geom.Radius * scale;
                Raylib.DrawCircleV(sp, sr, color);
                if (isMain)
                    Raylib.DrawCircleLinesV(sp, sr, new Color((byte)255, (byte)255, (byte)255, (byte)(rocketAlpha / 2)));
            }
        }

        // Sensor rays (visual only, computed on CPU for display)
        if (!isTerminal)
        {
            var body0 = bodies[bodyBase];
            float upX = MathF.Cos(body0.Angle);
            float upY = MathF.Sin(body0.Angle);
            float[] rayDirs = { upX, upY, -upX, -upY, -upY, upX, upY, -upX };
            var groundObb = OBBCollider.AxisAligned(0f, GroundY, 30f, 0.5f);
            const float maxRange = 30f;

            for (int s = 0; s < SensorCount; s++)
            {
                float dirX = rayDirs[s * 2], dirY = rayDirs[s * 2 + 1];
                float hitDist = maxRange;
                float d = RayVsOBBCpu(comX, comY, dirX, dirY, groundObb, maxRange);
                if (d < hitDist) hitDist = d;
                foreach (var obs in ObstacleOBBs)
                {
                    d = RayVsOBBCpu(comX, comY, dirX, dirY, obs, maxRange);
                    if (d < hitDist) hitDist = d;
                }
                float t = hitDist / maxRange;
                var rayColor = new Color((byte)(255 * (1f - t)), (byte)(200 * t), (byte)50, (byte)150);
                Raylib.DrawLineEx(W2S(comX, comY), W2S(comX + dirX * hitDist, comY + dirY * hitDist), 1f, rayColor);
            }
        }

        // Thrust flame
        if (!isTerminal && curThrottle > 0.01f)
        {
            var body0 = bodies[bodyBase];
            float bodyCos = MathF.Cos(body0.Angle);
            float bodySin = MathF.Sin(body0.Angle);
            float halfBody = 0.75f;
            float botX = body0.X - halfBody * bodyCos;
            float botY = body0.Y - halfBody * bodySin;

            float flameAngle = body0.Angle + curGimbal * 0.5f;
            float flameCos = MathF.Cos(flameAngle);
            float flameSin = MathF.Sin(flameAngle);

            float flameLen = curThrottle * 2.0f;
            float tipX = botX - flameCos * flameLen;
            float tipY = botY - flameSin * flameLen;
            float px = -flameSin * 0.3f * curThrottle;
            float py = flameCos * 0.3f * curThrottle;

            var tip = W2S(tipX, tipY);
            var left = W2S(botX + px, botY + py);
            var right = W2S(botX - px, botY - py);

            Raylib.DrawTriangle(tip, right, left,
                new Color((byte)255, (byte)180, (byte)30, (byte)(180 * curThrottle)));
            Raylib.DrawTriangle(tip, right, left,
                new Color((byte)255, (byte)100, (byte)20, (byte)(100 * curThrottle)));
        }

        // Status labels
        if (isTerminal)
        {
            if (hasLanded)
                Raylib.DrawText("LANDED", cellX + 5, cellY + 5, 16, Color.Green);
            else
                Raylib.DrawText("CRASHED", cellX + 5, cellY + 5, 14, new Color((byte)200, (byte)80, (byte)80, (byte)180));
        }
    }

    private static void DrawOBB(OBBCollider obb, Func<float, float, Vector2> W2S)
    {
        float px = -obb.UY, py = obb.UX;
        var s1 = W2S(obb.CX + obb.UX * obb.HalfExtentX + px * obb.HalfExtentY,
                      obb.CY + obb.UY * obb.HalfExtentX + py * obb.HalfExtentY);
        var s2 = W2S(obb.CX - obb.UX * obb.HalfExtentX + px * obb.HalfExtentY,
                      obb.CY - obb.UY * obb.HalfExtentX + py * obb.HalfExtentY);
        var s3 = W2S(obb.CX - obb.UX * obb.HalfExtentX - px * obb.HalfExtentY,
                      obb.CY - obb.UY * obb.HalfExtentX - py * obb.HalfExtentY);
        var s4 = W2S(obb.CX + obb.UX * obb.HalfExtentX - px * obb.HalfExtentY,
                      obb.CY + obb.UY * obb.HalfExtentX - py * obb.HalfExtentY);

        var fill = new Color((byte)60, (byte)60, (byte)70, (byte)200);
        Raylib.DrawTriangle(s1, s3, s2, fill);
        Raylib.DrawTriangle(s1, s4, s3, fill);

        var outline = new Color((byte)120, (byte)120, (byte)140, (byte)255);
        Raylib.DrawLineEx(s1, s2, 1.5f, outline);
        Raylib.DrawLineEx(s2, s3, 1.5f, outline);
        Raylib.DrawLineEx(s3, s4, 1.5f, outline);
        Raylib.DrawLineEx(s4, s1, 1.5f, outline);
    }

    private static float RayVsOBBCpu(float ox, float oy, float dx, float dy,
        OBBCollider obb, float maxRange)
    {
        float relX = ox - obb.CX, relY = oy - obb.CY;
        float px = -obb.UY, py = obb.UX;
        float localOX = relX * obb.UX + relY * obb.UY;
        float localOY = relX * px + relY * py;
        float localDX = dx * obb.UX + dy * obb.UY;
        float localDY = dx * px + dy * py;

        float tMin = 0f, tMax = maxRange;

        if (MathF.Abs(localDX) < 1e-8f)
        {
            if (localOX < -obb.HalfExtentX || localOX > obb.HalfExtentX) return maxRange;
        }
        else
        {
            float t1 = (-obb.HalfExtentX - localOX) / localDX;
            float t2 = (obb.HalfExtentX - localOX) / localDX;
            if (t1 > t2) (t1, t2) = (t2, t1);
            tMin = MathF.Max(tMin, t1);
            tMax = MathF.Min(tMax, t2);
            if (tMin > tMax) return maxRange;
        }

        if (MathF.Abs(localDY) < 1e-8f)
        {
            if (localOY < -obb.HalfExtentY || localOY > obb.HalfExtentY) return maxRange;
        }
        else
        {
            float t1 = (-obb.HalfExtentY - localOY) / localDY;
            float t2 = (obb.HalfExtentY - localOY) / localDY;
            if (t1 > t2) (t1, t2) = (t2, t1);
            tMin = MathF.Max(tMin, t1);
            tMax = MathF.Min(tMax, t2);
            if (tMin > tMax) return maxRange;
        }

        return tMin > 0 ? tMin : (tMax > 0 ? tMax : maxRange);
    }

    private static void DrawHUD(float bestFitness, int landings, float episodeTime,
        bool paused, int speed)
    {
        Raylib.DrawRectangle(0, 0, ScreenWidth, 35, new Color(0, 0, 0, 180));
        Raylib.DrawText("OBSTACLE LANDER (GPU)", 10, 8, 20, Color.White);
        Raylib.DrawText($"Best: {bestFitness:F0}", 250, 8, 20, Color.Green);
        Raylib.DrawText($"Speed: {speed}x", 380, 8, 20, Color.LightGray);
        Raylib.DrawText($"Time: {episodeTime:F1}s", 500, 8, 20, Color.LightGray);
        Raylib.DrawText($"Landings: {landings}/{CellCount}", 630, 8, 20, Color.Yellow);
        if (paused)
            Raylib.DrawText("PAUSED", ScreenWidth / 2 - 50, 8, 24, Color.Red);
        Raylib.DrawText("SPACE:Pause  R:Reset  +/-:Speed", ScreenWidth - 350, 8, 16, Color.Gray);
        Raylib.DrawText($"FPS:{Raylib.GetFPS()}", ScreenWidth - 80, 8, 16, Color.Green);
    }
}
