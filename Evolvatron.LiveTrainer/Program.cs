using System.Diagnostics;
using System.Numerics;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.GPU.MegaKernel;
using Evolvatron.Evolvion.World;
using Raylib_cs;

if (args.Length == 0)
{
    Console.WriteLine("Usage: dotnet run -- <world.json> [--generations N] [--spawns N]");
    return 1;
}

string jsonPath = Path.GetFullPath(args[0]);
if (!File.Exists(jsonPath))
{
    Console.Error.WriteLine($"File not found: {jsonPath}");
    return 1;
}

int maxGenerations = 300;
int numSpawns = 10;

for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--generations" && i + 1 < args.Length)
        maxGenerations = int.Parse(args[++i]);
    else if (args[i] == "--spawns" && i + 1 < args.Length)
        numSpawns = int.Parse(args[++i]);
}

// Shared training state
const int MaxReplayCount = 30;
float[]? latestTopNParams = null; // flat: [rocket0_params, rocket1_params, ...], sorted best-first
float[]? latestTopNFitness = null; // fitness values for top N, sorted best-first
int currentGeneration = 0;
float bestFitness = 0f;
float landingRate = 0f;
bool trainingDone = false;
int solvedGeneration = -1; // gen where first individual landed ALL spawns (-1 = unsolved)
object championLock = new();

// Graph data (landing rate history)
var landingRateHistory = new List<float>();
var fitnessHistory = new List<float>();

// File-watch reload state
var trainCts = new CancellationTokenSource();
int worldChangedFlag = 0; // 0=no, 1=yes — use Interlocked for thread safety
Thread? trainingThread = null;

// Load world + start training + configure viz
SimWorld world = null!;
DenseTopology topology = null!;
GPUDenseRocketLandingEvaluator? vizEval = null;

void LoadWorldAndStart()
{
    string json = File.ReadAllText(jsonPath);
    world = SimWorldLoader.FromJson(json);
    int sensorCount = world.SimulationConfig.SensorCount;
    topology = DenseTopology.ForRocket(new[] { 16, 12 }, sensorCount: sensorCount);

    Console.WriteLine($"LiveTrainer: {jsonPath}");
    Console.WriteLine($"  Generations: {maxGenerations}, Spawns: {numSpawns}, Sensors: {sensorCount}");

    // Reset shared state
    lock (championLock)
    {
        latestTopNParams = null;
        latestTopNFitness = null;
        currentGeneration = 0;
        bestFitness = 0f;
        landingRate = 0f;
        trainingDone = false;
        solvedGeneration = -1;
        landingRateHistory.Clear();
        fitnessHistory.Clear();
    }

    // Configure viz evaluator
    vizEval?.Dispose();
    vizEval = new GPUDenseRocketLandingEvaluator(topology);
    vizEval.Configure(world);

    // Start training thread
    trainCts = new CancellationTokenSource();
    var ct = trainCts.Token;
    trainingThread = new Thread(() => RunTraining(ct));
    trainingThread.IsBackground = true;
    trainingThread.Start();
}

void RunTraining(CancellationToken ct)
{
    using var trainEval = new GPUDenseRocketLandingEvaluator(topology);
    trainEval.Configure(world);

    var config = new IslandConfig
    {
        IslandCount = 1,
        Strategy = UpdateStrategyType.CEM,
        InitialSigma = 0.25f,
        MinSigma = 0.08f,
        MaxSigma = 2.0f,
        CEMEliteFraction = 0.01f,
        CEMSigmaSmoothing = 0.3f,
        CEMMuSmoothing = 0.2f,
        StagnationThreshold = 9999,
    };

    int gpuCapacity = trainEval.OptimalPopulationSize;
    var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
    var rng = new Random(42);
    var sw = Stopwatch.StartNew();

    int actualSpawns = trainEval.SpawnCount > 0 ? trainEval.SpawnCount : numSpawns;
    Console.WriteLine($"  Using {actualSpawns} deterministic spawn positions (fixed seeds)");

    // Elitism: carry forward best individual
    float[]? eliteParams = null;
    float eliteFitness = float.NegativeInfinity;

    for (int gen = 0; gen < maxGenerations; gen++)
    {
        if (ct.IsCancellationRequested) break;

        var paramVectors = optimizer.GeneratePopulation(rng);
        int paramCount = topology.TotalParams;
        int totalPop = optimizer.TotalPopulation;

        // Inject elite into last slot
        if (eliteParams != null)
            Array.Copy(eliteParams, 0, paramVectors, (totalPop - 1) * paramCount, paramCount);

        // Fixed baseSeed → same spawn positions every generation (from world config)
        var (fitness, landings, maxLandingCount) = trainEval.EvaluateMultiSpawn(
            paramVectors, totalPop, actualSpawns, baseSeed: trainEval.SpawnSeed);

        // Track elite (best ever)
        for (int i = 0; i < totalPop; i++)
        {
            if (fitness[i] > eliteFitness)
            {
                eliteFitness = fitness[i];
                eliteParams ??= new float[paramCount];
                Array.Copy(paramVectors, i * paramCount, eliteParams, 0, paramCount);
            }
        }

        optimizer.Update(fitness, paramVectors);
        optimizer.ManageIslands(rng);

        float maxFit = fitness.Max();
        float rate = (float)landings / (totalPop * actualSpawns) * 100f;

        // Extract top N from this generation's population
        int topN = Math.Min(MaxReplayCount, totalPop);

        // Get indices sorted by fitness descending
        var indices = Enumerable.Range(0, totalPop).ToArray();
        Array.Sort(indices, (a, b) => fitness[b].CompareTo(fitness[a]));

        var topParams = new float[topN * paramCount];
        var topFitness = new float[topN];
        for (int k = 0; k < topN; k++)
        {
            Array.Copy(paramVectors, indices[k] * paramCount, topParams, k * paramCount, paramCount);
            topFitness[k] = fitness[indices[k]];
        }

        // Guarantee elite is always slot 0 (best-ever individual never lost)
        if (eliteParams != null && eliteFitness > topFitness[0])
        {
            Array.Copy(eliteParams, 0, topParams, 0, paramCount);
            topFitness[0] = eliteFitness;
            maxFit = eliteFitness;
        }

        lock (championLock)
        {
            latestTopNParams = topParams;
            latestTopNFitness = topFitness;
            currentGeneration = gen + 1;
            bestFitness = MathF.Max(bestFitness, maxFit);
            landingRate = rate;
            landingRateHistory.Add(rate);
            fitnessHistory.Add(MathF.Max(bestFitness, maxFit));
            if (solvedGeneration < 0 && maxLandingCount >= actualSpawns)
            {
                solvedGeneration = gen + 1;
                Console.WriteLine($"  *** MAP SOLVED at Gen {solvedGeneration}! ({maxLandingCount}/{actualSpawns} landings) ***");
            }
        }

        if (gen % 10 == 0)
            Console.WriteLine($"  Gen {gen,3}: fit={MathF.Max(eliteFitness, maxFit),8:F1}  land={rate,5:F1}%  [{sw.Elapsed.TotalSeconds:F0}s]");
    }

    if (!ct.IsCancellationRequested)
    {
        var (finalMu, _) = optimizer.GetBestSolution();
        using var champEval = new GPUDenseRocketLandingEvaluator(topology);
        champEval.Configure(world);
        var (champLandings, champTotal) = champEval.EvaluateChampion(finalMu, numSpawns: 100, baseSeed: 9999);
        Console.WriteLine($"Champion: {champLandings}/{champTotal} ({(float)champLandings / champTotal * 100:F0}%)");
        trainingDone = true;
    }
    else
    {
        Console.WriteLine("  Training cancelled (world file changed).");
    }
}

// Initial load
LoadWorldAndStart();

// File watcher
var watcher = new FileSystemWatcher(Path.GetDirectoryName(jsonPath)!, Path.GetFileName(jsonPath));
watcher.Changed += (_, _) => Interlocked.Exchange(ref worldChangedFlag, 1);
watcher.EnableRaisingEvents = true;

const int ScreenWidth = 1280;
const int ScreenHeight = 800;
const int BodiesPerWorld = 3;
const int GeomsPerWorld = 19;
const int MaxTrailPoints = 600;

Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron Live Trainer");
Raylib.SetTargetFPS(60);
Raylib.SetTraceLogLevel(TraceLogLevel.Warning);

// Replay state (N rockets)
float[]? currentTopNParams = null;
int replaySeed = 0;
bool replayActive = false;
int replayStep = 0;
int postTerminalFrames = 0;
int activeReplayCount = 0;
GPURigidBody[]? bodies = null;
GPURigidBodyGeom[]? geoms = null;
byte[]? rTerminal = null;
byte[]? rLanded = null;
float[]? rThrottle = null;
float[]? rGimbal = null;
var trail = new List<(float x, float y)>(MaxTrailPoints);

// Training stats snapshot (read under lock)
int dispGen = 0;
float dispFit = 0f;
float dispRate = 0f;
bool dispDone = false;
int dispSolved = -1;
List<float> dispRateHistory = new();
List<float> dispFitHistory = new();

int simSpeed = 2; // physics steps per frame (2 = real-time at 60fps/120hz)
bool paused = false;
int displayedLosers = 29; // how many non-winner rockets to show (0 = winner only)
bool spawnSpreadMode = false; // S key: show best individual from all training spawn positions

// Camera
float padX = world.LandingPad.PadX;
float padY = world.LandingPad.PadY;
float groundSurfaceY = world.GroundY + 0.5f;
float spawnY = world.Spawn.Y + world.Spawn.HeightRange;

float worldMinX = -20f, worldMaxX = 20f;
float worldMinY = groundSurfaceY - 2f;
float worldMaxY = spawnY + 5f;

while (!Raylib.WindowShouldClose())
{
    // File-change reload
    if (Interlocked.CompareExchange(ref worldChangedFlag, 0, 1) == 1)
    {
        // Debounce — editors may write in multiple steps
        Thread.Sleep(300);
        Interlocked.Exchange(ref worldChangedFlag, 0);

        Console.WriteLine("\n  World file changed — reloading...");

        // Stop training thread
        trainCts.Cancel();
        trainingThread?.Join(5000);

        // Reset replay state
        currentTopNParams = null;
        replayActive = false;
        bodies = null;
        geoms = null;
        trail.Clear();
        postTerminalFrames = 0;

        // Reload everything
        try
        {
            LoadWorldAndStart();
            // Update camera bounds
            padX = world.LandingPad.PadX;
            padY = world.LandingPad.PadY;
            groundSurfaceY = world.GroundY + 0.5f;
            spawnY = world.Spawn.Y + world.Spawn.HeightRange;
            worldMinX = -20f; worldMaxX = 20f;
            worldMinY = groundSurfaceY - 2f;
            worldMaxY = spawnY + 5f;
            Console.WriteLine("  Reload complete — training restarted.");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"  Reload failed: {ex.Message}");
        }
    }

    // Input
    if (Raylib.IsKeyPressed(KeyboardKey.Space)) paused = !paused;
    if (Raylib.IsKeyPressed(KeyboardKey.Equal) || Raylib.IsKeyPressed(KeyboardKey.KpAdd))
        simSpeed = Math.Min(simSpeed + 1, 20);
    if (Raylib.IsKeyPressed(KeyboardKey.Minus) || Raylib.IsKeyPressed(KeyboardKey.KpSubtract))
        simSpeed = Math.Max(simSpeed - 1, 1);
    if (Raylib.IsKeyPressed(KeyboardKey.R)) StartNewReplay();
    if (Raylib.IsKeyPressed(KeyboardKey.S)) { spawnSpreadMode = !spawnSpreadMode; StartNewReplay(); }
    if (Raylib.IsKeyPressed(KeyboardKey.Up))
        displayedLosers = Math.Min(displayedLosers + 5, MaxReplayCount - 1);
    if (Raylib.IsKeyPressed(KeyboardKey.Down))
        displayedLosers = Math.Max(displayedLosers - 5, 0);

    // Read training stats
    float[]? pendingParams = null;
    lock (championLock)
    {
        dispGen = currentGeneration;
        dispFit = bestFitness;
        dispRate = landingRate;
        dispDone = trainingDone;
        dispSolved = solvedGeneration;

        if (landingRateHistory.Count > dispRateHistory.Count)
        {
            dispRateHistory = new List<float>(landingRateHistory);
            dispFitHistory = new List<float>(fitnessHistory);
        }

        if (latestTopNParams != null && !ReferenceEquals(latestTopNParams, currentTopNParams))
            pendingParams = latestTopNParams;
    }

    // Only switch to new generation when current replay is done
    if (pendingParams != null && !replayActive && postTerminalFrames <= 0)
    {
        currentTopNParams = pendingParams;
        StartNewReplay();
    }
    else if (!replayActive && currentTopNParams != null && postTerminalFrames <= 0)
        StartNewReplay();

    // Step physics (all N rockets)
    if (!paused && replayActive)
    {
        bool allTerminal = rTerminal != null && rTerminal.Length >= activeReplayCount;
        if (allTerminal)
        {
            for (int r = 0; r < activeReplayCount; r++)
                if (rTerminal![r] == 0) { allTerminal = false; break; }
        }

        for (int s = 0; s < simSpeed; s++)
        {
            if (allTerminal) break;
            vizEval!.StepReplay();
            replayStep++;
        }
        vizEval!.ReadMultiReplayState(out bodies!, out geoms!,
            out rTerminal!, out rLanded!, out rThrottle!, out rGimbal!);

        // Re-check after stepping
        allTerminal = true;
        for (int r = 0; r < activeReplayCount; r++)
            if (rTerminal![r] == 0) { allTerminal = false; break; }

        // Update trail for winner (rocket 0)
        if (rTerminal![0] == 0 && trail.Count < MaxTrailPoints)
        {
            var (cx, cy) = ComputeCOM(bodies!, 0);
            trail.Add((cx, cy));
        }

        if (allTerminal && replayActive)
        {
            replayActive = false;
            postTerminalFrames = 90;
        }
    }

    if (postTerminalFrames > 0)
        postTerminalFrames--;

    // --- Draw ---
    Raylib.BeginDrawing();
    Raylib.ClearBackground(new Color(10, 10, 30, 255));

    // World transform
    float worldW = worldMaxX - worldMinX;
    float worldH = worldMaxY - worldMinY;
    float drawAreaY = 40;
    float drawAreaH = ScreenHeight - 70;
    float scaleX = (ScreenWidth - 20) / worldW;
    float scaleY = drawAreaH / worldH;
    float scale = MathF.Min(scaleX, scaleY);
    float cScreenX = ScreenWidth / 2f;
    float cScreenY = drawAreaY + drawAreaH / 2f;
    float cWorldX = (worldMinX + worldMaxX) / 2f;
    float cWorldY = (worldMinY + worldMaxY) / 2f;

    Vector2 W2S(float wx, float wy)
    {
        return new Vector2(
            cScreenX + (wx - cWorldX) * scale,
            cScreenY - (wy - cWorldY) * scale);
    }

    float WorldToScreenSize(float worldSize) => worldSize * scale;

    // Ground
    Raylib.DrawLineEx(W2S(-25f, groundSurfaceY), W2S(25f, groundSurfaceY), 2f, new Color(100, 100, 100, 255));

    // Landing pad
    float padBottom = padY - world.LandingPad.PadHalfHeight;
    float padTop = padY + world.LandingPad.PadHalfHeight;
    var pl = W2S(padX - world.LandingPad.PadHalfWidth, padBottom);
    var pr = W2S(padX + world.LandingPad.PadHalfWidth, padTop);
    Raylib.DrawRectangleV(new Vector2(pl.X, pr.Y), new Vector2(pr.X - pl.X, pl.Y - pr.Y),
        new Color(200, 100, 20, 220));

    // Obstacles
    foreach (var obs in vizEval.Obstacles)
        DrawOBB(obs, W2S);

    // Danger zones
    foreach (var dz in vizEval.DangerZones)
    {
        var dzTL = W2S(dz.X - dz.HalfExtentX, dz.Y + dz.HalfExtentY);
        var dzBR = W2S(dz.X + dz.HalfExtentX, dz.Y - dz.HalfExtentY);
        Raylib.DrawRectangleV(dzTL, new Vector2(dzBR.X - dzTL.X, dzBR.Y - dzTL.Y),
            new Color(200, 40, 40, 60));
        Raylib.DrawRectangleLinesEx(new Rectangle(dzTL.X, dzTL.Y, dzBR.X - dzTL.X, dzBR.Y - dzTL.Y),
            1.5f, new Color(200, 60, 60, 150));
        // Influence radius
        if (dz.InfluenceRadius > 0)
        {
            var irTL = W2S(dz.X - dz.HalfExtentX - dz.InfluenceRadius, dz.Y + dz.HalfExtentY + dz.InfluenceRadius);
            var irBR = W2S(dz.X + dz.HalfExtentX + dz.InfluenceRadius, dz.Y - dz.HalfExtentY - dz.InfluenceRadius);
            Raylib.DrawRectangleLinesEx(new Rectangle(irTL.X, irTL.Y, irBR.X - irTL.X, irBR.Y - irTL.Y),
                1f, new Color(200, 60, 60, 50));
        }
    }

    // Speed zones
    foreach (var sz in vizEval.SpeedZones)
    {
        var szTL = W2S(sz.X - sz.HalfExtentX, sz.Y + sz.HalfExtentY);
        var szBR = W2S(sz.X + sz.HalfExtentX, sz.Y - sz.HalfExtentY);
        Raylib.DrawRectangleV(szTL, new Vector2(szBR.X - szTL.X, szBR.Y - szTL.Y),
            new Color(40, 120, 200, 40));
        Raylib.DrawRectangleLinesEx(new Rectangle(szTL.X, szTL.Y, szBR.X - szTL.X, szBR.Y - szTL.Y),
            1.5f, new Color(60, 140, 220, 120));
    }

    // Attractors
    foreach (var att in vizEval.Attractors)
    {
        var aTL = W2S(att.X - att.HalfExtentX, att.Y + att.HalfExtentY);
        var aBR = W2S(att.X + att.HalfExtentX, att.Y - att.HalfExtentY);
        Raylib.DrawRectangleV(aTL, new Vector2(aBR.X - aTL.X, aBR.Y - aTL.Y),
            new Color(40, 200, 80, 40));
        Raylib.DrawRectangleLinesEx(new Rectangle(aTL.X, aTL.Y, aBR.X - aTL.X, aBR.Y - aTL.Y),
            1.5f, new Color(60, 220, 100, 120));
        // Influence radius
        if (att.InfluenceRadius > 0)
        {
            var center = W2S(att.X, att.Y);
            float ir = WorldToScreenSize(att.InfluenceRadius + MathF.Max(att.HalfExtentX, att.HalfExtentY));
            Raylib.DrawCircleLinesV(center, ir, new Color(60, 220, 100, 40));
        }
    }

    // Checkpoints
    int cpIdx = 0;
    foreach (var cp in vizEval.Checkpoints)
    {
        var center = W2S(cp.X, cp.Y);
        float r = WorldToScreenSize(cp.Radius);
        var fillColor = new Color(220, 200, 40, 30);
        var lineColor = new Color(220, 200, 40, 120);
        Raylib.DrawCircleV(center, r, fillColor);
        Raylib.DrawCircleLinesV(center, r, lineColor);
        // Influence radius
        if (cp.InfluenceRadius > 0)
            Raylib.DrawCircleLinesV(center, WorldToScreenSize(cp.Radius + cp.InfluenceRadius),
                new Color(220, 200, 40, 40));
        Raylib.DrawText($"{cp.Order}", (int)center.X - 5, (int)center.Y - 8, 16, new Color(220, 200, 40, 180));
        cpIdx++;
    }

    // Winner trail
    if (trail.Count > 1)
    {
        for (int i = 1; i < trail.Count; i++)
        {
            int alpha = 60 + (int)(160f * i / trail.Count);
            Raylib.DrawLineEx(W2S(trail[i - 1].x, trail[i - 1].y), W2S(trail[i].x, trail[i].y),
                1.5f, new Color(80, 160, 255, alpha));
        }
    }

    // Multi-rocket rendering
    if (bodies != null && geoms != null && rTerminal != null)
    {
        // Draw losers first (faintest last-place → brighter near top), then winner on top
        int maxIdx = Math.Min(1 + displayedLosers, activeReplayCount);
        for (int rocketIdx = maxIdx - 1; rocketIdx >= 0; rocketIdx--)
        {
            bool isWinner = rocketIdx == 0;
            bool isTerminal = rTerminal[rocketIdx] != 0;
            bool isLanded = rLanded != null && rLanded[rocketIdx] != 0;

            // Alpha: winner=220, losers fade from 80 (rank 1) to 15 (rank N)
            int baseAlpha;
            if (isWinner)
                baseAlpha = isTerminal ? 150 : 220;
            else
                baseAlpha = Math.Max(15, 80 - (rocketIdx - 1) * 2);

            if (isTerminal && !isWinner)
                baseAlpha = baseAlpha / 3;

            int bodyBase = rocketIdx * BodiesPerWorld;

            // Body geoms
            for (int b = 0; b < BodiesPerWorld; b++)
            {
                var body = bodies[bodyBase + b];
                float cos = MathF.Cos(body.Angle);
                float sin = MathF.Sin(body.Angle);
                bool isMainBody = (b == 0);

                Color color;
                bool isCrashed = isTerminal && !isLanded;
                if (isCrashed)
                    color = isWinner
                        ? new Color(220, 60, 60, baseAlpha)
                        : new Color(200, 50, 50, baseAlpha);
                else if (isWinner)
                    color = isMainBody
                        ? new Color(100, 200, 255, baseAlpha)
                        : new Color(80, 160, 220, baseAlpha);
                else
                    color = new Color(180, 180, 200, baseAlpha);

                for (int g = 0; g < body.GeomCount; g++)
                {
                    var geom = geoms[body.GeomStartIndex + g];
                    float wx = body.X + geom.LocalX * cos - geom.LocalY * sin;
                    float wy = body.Y + geom.LocalX * sin + geom.LocalY * cos;
                    var sp = W2S(wx, wy);
                    float sr = geom.Radius * scale;
                    Raylib.DrawCircleV(sp, sr, color);
                    if (isWinner && isMainBody)
                    {
                        var outlineColor = isCrashed
                            ? new Color(255, 100, 100, baseAlpha / 2)
                            : new Color(255, 255, 255, baseAlpha / 2);
                        Raylib.DrawCircleLinesV(sp, sr, outlineColor);
                    }
                }
            }

            // Thrust flame (winner only)
            if (isWinner && !isTerminal && rThrottle != null && rThrottle[0] > 0.01f)
            {
                var body0 = bodies[bodyBase];
                float bodyCos = MathF.Cos(body0.Angle);
                float bodySin = MathF.Sin(body0.Angle);
                float halfBody = 0.75f;
                float botX = body0.X - halfBody * bodyCos;
                float botY = body0.Y - halfBody * bodySin;

                float gimbalVal = rGimbal != null ? rGimbal[0] : 0f;
                float flameAngle = body0.Angle + gimbalVal * 0.5f;
                float flameCos = MathF.Cos(flameAngle);
                float flameSin = MathF.Sin(flameAngle);
                float throttle = rThrottle[0];

                float flameLen = throttle * 2.0f;
                float tipX = botX - flameCos * flameLen;
                float tipY = botY - flameSin * flameLen;
                float px = -flameSin * 0.3f * throttle;
                float py = flameCos * 0.3f * throttle;

                var tip = W2S(tipX, tipY);
                var left = W2S(botX + px, botY + py);
                var right = W2S(botX - px, botY - py);

                Raylib.DrawTriangle(tip, right, left,
                    new Color(255, 180, 30, (int)(180 * throttle)));
                Raylib.DrawTriangle(tip, right, left,
                    new Color(255, 100, 20, (int)(100 * throttle)));
            }

            // Sensor rays (winner only)
            if (isWinner && !isTerminal && world.SimulationConfig.SensorCount >= 4)
            {
                var (comX, comY) = ComputeCOM(bodies, 0);
                var body0 = bodies[bodyBase];
                float upX = MathF.Cos(body0.Angle);
                float upY = MathF.Sin(body0.Angle);
                float[] rayDirs = { upX, upY, -upX, -upY, -upY, upX, upY, -upX };
                float maxRange = vizEval!.MaxSensorRange;

                for (int s = 0; s < 4; s++)
                {
                    float dirX = rayDirs[s * 2], dirY = rayDirs[s * 2 + 1];
                    float hitDist = maxRange;
                    hitDist = MathF.Min(hitDist, RayVsAABB(comX, comY, dirX, dirY, 0f, world.GroundY, 30f, 0.5f, maxRange));
                    foreach (var obs in vizEval.Obstacles)
                        hitDist = MathF.Min(hitDist, RayVsOBBCpu(comX, comY, dirX, dirY, obs, maxRange));
                    float t = hitDist / maxRange;
                    var rayColor = new Color((int)(255 * (1f - t)), (int)(200 * t), 50, 100);
                    Raylib.DrawLineEx(W2S(comX, comY), W2S(comX + dirX * hitDist, comY + dirY * hitDist), 1f, rayColor);
                }
            }

            // Status label (winner only)
            if (isWinner && isTerminal)
            {
                var (cx, cy) = ComputeCOM(bodies, 0);
                string label = isLanded ? "LANDED" : "CRASHED";
                var labelColor = isLanded ? Color.Green : new Color(200, 80, 80, 200);
                var pos = W2S(cx, cy + 2f);
                Raylib.DrawText(label, (int)pos.X - 30, (int)pos.Y - 10, 20, labelColor);
            }
        }
    }
    else if (currentTopNParams == null)
    {
        Raylib.DrawText("Waiting for first generation...", ScreenWidth / 2 - 150, ScreenHeight / 2, 20, Color.Gray);
    }

    // Landing rate graph (bottom-right corner)
    if (dispRateHistory.Count > 1)
    {
        int graphW = 240, graphH = 100;
        int graphX = ScreenWidth - graphW - 10;
        int graphY = ScreenHeight - graphH - 34;

        Raylib.DrawRectangle(graphX, graphY, graphW, graphH, new Color(0, 0, 0, 160));
        Raylib.DrawRectangleLinesEx(new Rectangle(graphX, graphY, graphW, graphH), 1f, new Color(60, 60, 80, 200));
        Raylib.DrawText("Landing %", graphX + 4, graphY + 2, 10, new Color(200, 200, 200, 150));

        // Find max for scaling
        float maxRate = 1f;
        foreach (var r in dispRateHistory)
            if (r > maxRate) maxRate = r;
        maxRate = MathF.Ceiling(maxRate / 10f) * 10f;
        if (maxRate < 10f) maxRate = 10f;

        // Grid lines at 25%, 50%, 75%
        for (int pct = 25; pct < (int)maxRate; pct += 25)
        {
            float gy = graphY + graphH - (pct / maxRate) * graphH;
            Raylib.DrawLineEx(new Vector2(graphX, gy), new Vector2(graphX + graphW, gy), 1f, new Color(50, 50, 70, 100));
            Raylib.DrawText($"{pct}%", graphX + graphW - 30, (int)gy - 5, 8, new Color(120, 120, 140, 120));
        }

        int count = dispRateHistory.Count;
        float step = (float)graphW / Math.Max(count - 1, 1);
        for (int i = 1; i < count; i++)
        {
            float x0 = graphX + (i - 1) * step;
            float y0 = graphY + graphH - (dispRateHistory[i - 1] / maxRate) * graphH;
            float x1 = graphX + i * step;
            float y1 = graphY + graphH - (dispRateHistory[i] / maxRate) * graphH;
            Raylib.DrawLineEx(new Vector2(x0, y0), new Vector2(x1, y1), 1.5f, new Color(80, 220, 120, 200));
        }

        // Fitness line (right axis, yellow)
        if (dispFitHistory.Count > 1)
        {
            float maxFitGraph = 1f;
            foreach (var f in dispFitHistory)
                if (f > maxFitGraph) maxFitGraph = f;
            maxFitGraph = MathF.Ceiling(maxFitGraph / 100f) * 100f;
            if (maxFitGraph < 100f) maxFitGraph = 100f;

            Raylib.DrawText($"Fit ({maxFitGraph:F0})", graphX + graphW - 70, graphY + 2, 10, new Color(220, 200, 80, 150));
            for (int i = 1; i < dispFitHistory.Count; i++)
            {
                float x0 = graphX + (i - 1) * step;
                float y0 = graphY + graphH - (dispFitHistory[i - 1] / maxFitGraph) * graphH;
                float x1 = graphX + i * step;
                float y1 = graphY + graphH - (dispFitHistory[i] / maxFitGraph) * graphH;
                Raylib.DrawLineEx(new Vector2(x0, y0), new Vector2(x1, y1), 1.5f, new Color(220, 200, 80, 180));
            }
        }
    }

    // HUD
    Raylib.DrawRectangle(0, 0, ScreenWidth, 36, new Color(0, 0, 0, 200));
    Raylib.DrawText("EVOLVATRON LIVE", 10, 8, 20, Color.White);
    string genText = dispDone ? $"Gen: {dispGen} (done)" : $"Gen: {dispGen}/{maxGenerations}";
    Raylib.DrawText(genText, 190, 8, 18, Color.LightGray);
    Raylib.DrawText($"Fit: {dispFit:F0}", 400, 8, 18, Color.Green);
    Raylib.DrawText($"Land: {dispRate:F1}%", 520, 8, 18, Color.Yellow);
    float replayTime = replayStep / 120f;
    float maxTime = world.SimulationConfig.MaxSteps / 120f;
    Raylib.DrawText($"T:{replayTime:F1}/{maxTime:F1}s", 660, 8, 16, Color.LightGray);
    Raylib.DrawText($"Speed: {simSpeed}x", 800, 8, 18, Color.LightGray);
    string losersText = displayedLosers > 0 ? $"Show: {1 + displayedLosers}" : "Show: 1";
    Raylib.DrawText(losersText, 920, 8, 18, Color.LightGray);
    if (spawnSpreadMode)
        Raylib.DrawText("SPREAD", 1020, 8, 18, Color.Orange);
    if (dispSolved >= 0)
        Raylib.DrawText($"SOLVED @{dispSolved}", 1100, 8, 18, Color.Green);
    if (paused)
        Raylib.DrawText("PAUSED", ScreenWidth / 2 - 40, 8, 20, Color.Red);
    Raylib.DrawText($"FPS:{Raylib.GetFPS()}", ScreenWidth - 80, 8, 16, Color.Green);

    Raylib.DrawRectangle(0, ScreenHeight - 28, ScreenWidth, 28, new Color(0, 0, 0, 180));
    Raylib.DrawText("SPACE:Pause  +/-:Speed  R:Restart  S:SpawnSpread  Up/Down:Rockets", 10, ScreenHeight - 22, 14, Color.Gray);
    Raylib.DrawText(Path.GetFileName(jsonPath), ScreenWidth - 200, ScreenHeight - 22, 14, Color.DarkGray);

    Raylib.EndDrawing();
}

trainCts.Cancel();
trainingThread?.Join(5000);
watcher.Dispose();
vizEval?.Dispose();
Raylib.CloseWindow();
return 0;

// --- Helper functions ---

void StartNewReplay()
{
    if (currentTopNParams == null) return;
    replaySeed++;
    trail.Clear();
    replayStep = 0;
    postTerminalFrames = 0;

    if (spawnSpreadMode)
    {
        int paramCount = topology.TotalParams;
        var bestParams = new float[paramCount];
        Array.Copy(currentTopNParams, 0, bestParams, 0, paramCount);
        int spawns = vizEval!.SpawnCount > 0 ? vizEval.SpawnCount : numSpawns;
        vizEval.PrepareSpawnSpreadReplay(bestParams, spawns, vizEval.SpawnSeed);
        activeReplayCount = spawns;
    }
    else
    {
        activeReplayCount = currentTopNParams.Length / topology.TotalParams;
        vizEval!.PrepareMultiReplay(currentTopNParams, activeReplayCount, replaySeed);
    }

    vizEval!.ReadMultiReplayState(out bodies!, out geoms!, out rTerminal!, out rLanded!, out rThrottle!, out rGimbal!);
    var (cx, cy) = ComputeCOM(bodies!, 0);
    trail.Add((cx, cy));
    replayActive = true;
}

static (float x, float y) ComputeCOM(GPURigidBody[] bodies, int rocketIdx)
{
    int bodyBase = rocketIdx * BodiesPerWorld;
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
    return (comX, comY);
}

static void DrawOBB(GPUOBBCollider obb, Func<float, float, Vector2> W2S)
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

    var fill = new Color(60, 60, 70, 200);
    Raylib.DrawTriangle(s1, s3, s2, fill);
    Raylib.DrawTriangle(s1, s4, s3, fill);

    var outline = new Color(120, 120, 140, 255);
    Raylib.DrawLineEx(s1, s2, 1.5f, outline);
    Raylib.DrawLineEx(s2, s3, 1.5f, outline);
    Raylib.DrawLineEx(s3, s4, 1.5f, outline);
    Raylib.DrawLineEx(s4, s1, 1.5f, outline);
}

static float RayVsOBBCpu(float ox, float oy, float dx, float dy, GPUOBBCollider obb, float maxRange)
{
    float relX = ox - obb.CX, relY = oy - obb.CY;
    float px = -obb.UY, py = obb.UX;
    float localOX = relX * obb.UX + relY * obb.UY;
    float localOY = relX * px + relY * py;
    float localDX = dx * obb.UX + dy * obb.UY;
    float localDY = dx * px + dy * py;

    float tMin = 0f, tMax = maxRange;
    if (MathF.Abs(localDX) < 1e-8f)
    { if (localOX < -obb.HalfExtentX || localOX > obb.HalfExtentX) return maxRange; }
    else
    {
        float t1 = (-obb.HalfExtentX - localOX) / localDX;
        float t2 = (obb.HalfExtentX - localOX) / localDX;
        if (t1 > t2) (t1, t2) = (t2, t1);
        tMin = MathF.Max(tMin, t1); tMax = MathF.Min(tMax, t2);
        if (tMin > tMax) return maxRange;
    }
    if (MathF.Abs(localDY) < 1e-8f)
    { if (localOY < -obb.HalfExtentY || localOY > obb.HalfExtentY) return maxRange; }
    else
    {
        float t1 = (-obb.HalfExtentY - localOY) / localDY;
        float t2 = (obb.HalfExtentY - localOY) / localDY;
        if (t1 > t2) (t1, t2) = (t2, t1);
        tMin = MathF.Max(tMin, t1); tMax = MathF.Min(tMax, t2);
        if (tMin > tMax) return maxRange;
    }
    return tMin > 0 ? tMin : (tMax > 0 ? tMax : maxRange);
}

static float RayVsAABB(float ox, float oy, float dx, float dy,
    float cx, float cy, float hx, float hy, float maxRange)
{
    float tMin = 0f, tMax = maxRange;
    if (MathF.Abs(dx) < 1e-8f)
    { if (ox < cx - hx || ox > cx + hx) return maxRange; }
    else
    {
        float t1 = (cx - hx - ox) / dx;
        float t2 = (cx + hx - ox) / dx;
        if (t1 > t2) (t1, t2) = (t2, t1);
        tMin = MathF.Max(tMin, t1); tMax = MathF.Min(tMax, t2);
        if (tMin > tMax) return maxRange;
    }
    if (MathF.Abs(dy) < 1e-8f)
    { if (oy < cy - hy || oy > cy + hy) return maxRange; }
    else
    {
        float t1 = (cy - hy - oy) / dy;
        float t2 = (cy + hy - oy) / dy;
        if (t1 > t2) (t1, t2) = (t2, t1);
        tMin = MathF.Max(tMin, t1); tMax = MathF.Min(tMax, t2);
        if (tMin > tMax) return maxRange;
    }
    return tMin > 0 ? tMin : (tMax > 0 ? tMax : maxRange);
}
