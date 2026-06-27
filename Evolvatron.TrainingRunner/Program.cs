using System.Diagnostics;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.World;

if (args.Length == 0)
{
    Console.WriteLine("Usage: dotnet run -- <world.json> [--generations N] [--spawns N]");
    Console.WriteLine("  Watches the JSON file for changes and restarts training automatically.");
    Console.WriteLine();
    Console.WriteLine("Phase-1 controller: dotnet run -- --control [--generations N] [--conditions N] [--hidden 16,16] [--out controller.bin]");
    return 1;
}

if (args[0] == "--control")
{
    return RunControlMode(args);
}

if (args[0] == "--maze")
{
    return RunMazeMode(args);
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

Console.WriteLine($"Training runner: {jsonPath}");
Console.WriteLine($"  Generations: {maxGenerations}, Spawns: {numSpawns}");
Console.WriteLine($"  Watching for file changes. Press Ctrl+C to quit.");
Console.WriteLine();

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cts.Cancel();
    Console.WriteLine("\nShutting down...");
};

while (!cts.IsCancellationRequested)
{
    try
    {
        var fileChangeToken = new CancellationTokenSource();
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cts.Token, fileChangeToken.Token);

        using var watcher = new FileSystemWatcher(
            Path.GetDirectoryName(jsonPath)!,
            Path.GetFileName(jsonPath));
        watcher.NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size;
        watcher.Changed += (_, _) =>
        {
            Console.WriteLine("\n--- File changed, restarting training ---\n");
            fileChangeToken.Cancel();
        };
        watcher.EnableRaisingEvents = true;

        RunTraining(jsonPath, maxGenerations, numSpawns, linked.Token);

        // Training completed normally — wait for file change before restarting
        if (!linked.IsCancellationRequested)
        {
            Console.WriteLine("Training complete. Waiting for file change to retrain...");
            var waitTcs = new TaskCompletionSource();
            watcher.Changed += (_, _) => waitTcs.TrySetResult();
            try { waitTcs.Task.Wait(cts.Token); }
            catch (OperationCanceledException) { break; }
            Console.WriteLine("\n--- File changed, restarting training ---\n");
            Thread.Sleep(500);
        }
    }
    catch (OperationCanceledException) when (cts.IsCancellationRequested)
    {
        break;
    }
    catch (OperationCanceledException)
    {
        // File changed during training — loop will restart
        Thread.Sleep(500); // let file writes settle
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        Console.WriteLine("Waiting for file change to retry...");

        // Wait for next file change before retrying
        using var retryWatcher = new FileSystemWatcher(
            Path.GetDirectoryName(jsonPath)!,
            Path.GetFileName(jsonPath));
        retryWatcher.NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size;
        var retryTcs = new TaskCompletionSource();
        retryWatcher.Changed += (_, _) => retryTcs.TrySetResult();
        retryWatcher.EnableRaisingEvents = true;

        try { retryTcs.Task.Wait(cts.Token); }
        catch (OperationCanceledException) { break; }

        Thread.Sleep(500);
    }
}

Console.WriteLine("Training runner stopped.");
return 0;

// === Phase-1 maneuvering controller (see docs/phase1_controller_spec.md) ===
static int RunControlMode(string[] args)
{
    int maxGenerations = 200;
    int numConditions = 20;
    int[] hidden = { 16, 16 };
    int contextSize = 0;   // Elman recurrence width (0 = plain reactive controller)
    string outPath = Path.Combine(Directory.GetCurrentDirectory(), "controller.bin");

    // Optional overrides; defaults match the evaluator's own defaults (null = leave as-is).
    int? maxSteps = null;
    int? segmentLength = null;
    float? angVelPenalty = null;
    float? initAngVel = null;
    float? cmdSpeed = null;
    float? tumblePenalty = null;
    float? verrScale = null;
    float? effortWeight = null;

    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--generations" && i + 1 < args.Length) maxGenerations = int.Parse(args[++i]);
        else if (args[i] == "--conditions" && i + 1 < args.Length) numConditions = int.Parse(args[++i]);
        else if (args[i] == "--hidden" && i + 1 < args.Length) hidden = args[++i].Split(',').Select(int.Parse).ToArray();
        else if (args[i] == "--context" && i + 1 < args.Length) contextSize = int.Parse(args[++i]);
        else if (args[i] == "--out" && i + 1 < args.Length) outPath = Path.GetFullPath(args[++i]);
        else if (args[i] == "--max-steps" && i + 1 < args.Length) maxSteps = int.Parse(args[++i]);
        else if (args[i] == "--segment-length" && i + 1 < args.Length) segmentLength = int.Parse(args[++i]);
        else if (args[i] == "--angvel-penalty" && i + 1 < args.Length) angVelPenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--init-angvel" && i + 1 < args.Length) initAngVel = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--cmd-speed" && i + 1 < args.Length) cmdSpeed = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--tumble-penalty" && i + 1 < args.Length) tumblePenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--verr-scale" && i + 1 < args.Length) verrScale = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--effort-weight" && i + 1 < args.Length) effortWeight = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
    }

    var topology = DenseTopology.ForRocketController(hidden, contextSize);
    using var eval = new GPUDenseRocketControlEvaluator(topology);
    if (maxSteps.HasValue) eval.MaxSteps = maxSteps.Value;
    if (segmentLength.HasValue) eval.SegmentLength = segmentLength.Value;
    if (angVelPenalty.HasValue) eval.AngVelPenalty = angVelPenalty.Value;
    if (initAngVel.HasValue) eval.InitialAngVelMax = initAngVel.Value;
    if (cmdSpeed.HasValue) eval.CmdSpeedMax = cmdSpeed.Value;
    if (tumblePenalty.HasValue) eval.TumblePenalty = tumblePenalty.Value;
    if (verrScale.HasValue) eval.VErrScale = verrScale.Value;
    if (effortWeight.HasValue) eval.RewardEffortWeight = effortWeight.Value;
    int gpuCap = eval.OptimalPopulationSize;

    Console.WriteLine("Phase-1 maneuvering-controller training (free space, velocity tracking)");
    Console.WriteLine($"  Topology: {topology}");
    Console.WriteLine($"  GPU pop: {gpuCap}, conditions/gen: {numConditions}, " +
        $"segments/episode: {eval.SegmentsPerEpisode}, MaxSteps: {eval.MaxSteps}, Elman context: {eval.ContextSize}");
    Console.WriteLine($"  Reward: VErrScale={eval.VErrScale:F2}, EffortWeight={eval.RewardEffortWeight:F3}, " +
        $"AngVelPenalty={eval.AngVelPenalty:F3}, TumblePenalty={eval.TumblePenalty:F1}");
    Console.WriteLine($"  Task: InitAngVelMax={eval.InitialAngVelMax:F2}, CmdSpeedMax={eval.CmdSpeedMax:F1}, SegLen={eval.SegmentLength}, " +
        $"SolveThreshold={eval.TrackSolveThreshold:F2}");
    Console.WriteLine();

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

    var optimizer = new IslandOptimizer(config, topology, gpuCap);
    var rng = new Random(42);
    var sw = Stopwatch.StartNew();
    int totalEvals = optimizer.TotalPopulation * numConditions;

    for (int gen = 0; gen < maxGenerations; gen++)
    {
        var pv = optimizer.GeneratePopulation(rng);
        var (fit, solved, _, _) = eval.EvaluateMultiCondition(
            pv, optimizer.TotalPopulation, numConditions, baseSeed: gen * 100);
        optimizer.Update(fit, pv);

        if (gen % 10 == 0 || gen == maxGenerations - 1)
        {
            float solveRate = (float)solved / totalEvals * 100f;
            Console.WriteLine($"  Gen {gen,3}: fit={fit.Max(),9:F1}  " +
                $"solved={solved,7}/{totalEvals} ({solveRate,5:F1}%)  [{sw.Elapsed.TotalSeconds:F0}s]");
        }
    }

    var (mu, _) = optimizer.GetBestSolution();
    var (champSolved, champTotal) = eval.EvaluateChampion(mu, numConditions: 100, baseSeed: 999999);
    Console.WriteLine();
    Console.WriteLine($"Champion: {champSolved}/{champTotal} held-out conditions tracked " +
        $"({(float)champSolved / champTotal * 100:F0}%)");
    Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F1}s");

    using (var fs = File.Create(outPath))
    using (var bw = new BinaryWriter(fs))
    {
        bw.Write(mu.Length);
        foreach (float v in mu) bw.Write(v);
    }
    Console.WriteLine($"Controller saved to: {outPath}");
    return 0;
}

static float[] LoadBin(string path)
{
    using var fs = File.OpenRead(path);
    using var br = new BinaryReader(fs);
    int n = br.ReadInt32();
    var v = new float[n];
    for (int i = 0; i < n; i++) v[i] = br.ReadSingle();
    return v;
}

// === Phase-2 maze navigator (see docs/phase2_maze_spec.md) ===
static int RunMazeMode(string[] args)
{
    int maxGenerations = 200;
    int numMazes = 12;
    int[] mazeHidden = { 24, 24 };
    int[] ctrlHidden = { 16, 16 };
    string controllerPath = Path.Combine(Directory.GetCurrentDirectory(), "scratch", "controller_easy.bin");
    int sensors = 0;
    int obstacles = 0;
    int maxSteps = 600;
    float cmdSpeed = 3f;
    float goalRadius = 0.75f;
    string outPath = Path.Combine(Directory.GetCurrentDirectory(), "maze_policy.bin");

    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--generations" && i + 1 < args.Length) maxGenerations = int.Parse(args[++i]);
        else if (args[i] == "--mazes" && i + 1 < args.Length) numMazes = int.Parse(args[++i]);
        else if (args[i] == "--hidden" && i + 1 < args.Length) mazeHidden = args[++i].Split(',').Select(int.Parse).ToArray();
        else if (args[i] == "--controller" && i + 1 < args.Length) controllerPath = Path.GetFullPath(args[++i]);
        else if (args[i] == "--controller-hidden" && i + 1 < args.Length) ctrlHidden = args[++i].Split(',').Select(int.Parse).ToArray();
        else if (args[i] == "--sensors" && i + 1 < args.Length) sensors = int.Parse(args[++i]);
        else if (args[i] == "--obstacles" && i + 1 < args.Length) obstacles = int.Parse(args[++i]);
        else if (args[i] == "--max-steps" && i + 1 < args.Length) maxSteps = int.Parse(args[++i]);
        else if (args[i] == "--cmd-speed" && i + 1 < args.Length) cmdSpeed = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--goal-radius" && i + 1 < args.Length) goalRadius = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
        else if (args[i] == "--out" && i + 1 < args.Length) outPath = Path.GetFullPath(args[++i]);
    }

    if (!File.Exists(controllerPath))
    {
        Console.Error.WriteLine($"Frozen controller not found: {controllerPath}");
        return 1;
    }

    var ctrlParams = LoadBin(controllerPath);
    var ctrlTopology = DenseTopology.ForRocketController(ctrlHidden);
    if (ctrlParams.Length != ctrlTopology.TotalParams)
    {
        Console.Error.WriteLine($"Controller {controllerPath} has {ctrlParams.Length} params but topology " +
            $"{ctrlTopology} expects {ctrlTopology.TotalParams}. Pass --controller-hidden to match.");
        return 1;
    }

    int mazeInput = 6 + (sensors >= 8 ? 8 : (sensors >= 4 ? 4 : 0));
    var mazeLayers = new int[mazeHidden.Length + 2];
    mazeLayers[0] = mazeInput;
    for (int i = 0; i < mazeHidden.Length; i++) mazeLayers[i + 1] = mazeHidden[i];
    mazeLayers[^1] = 2;
    var mazeTopology = new DenseTopology(mazeLayers);

    using var eval = new GPUDenseMazeEvaluator(mazeTopology, ctrlTopology, ctrlParams)
    {
        MaxSteps = maxSteps,
        SensorCount = sensors,
        NumObstacles = obstacles,
        CmdSpeedMax = cmdSpeed,
        GoalRadius = goalRadius
    };
    int gpuCap = eval.OptimalPopulationSize;

    Console.WriteLine("Phase-2 maze navigation (frozen controller + evolved maze policy)");
    Console.WriteLine($"  Frozen controller: {controllerPath} ({ctrlTopology})");
    Console.WriteLine($"  GPU pop: {gpuCap}, mazes/gen: {numMazes}, MaxSteps: {maxSteps}");
    Console.WriteLine($"  Task: sensors={sensors}, obstacles={obstacles}, CmdSpeedMax={cmdSpeed}, GoalRadius={eval.GoalRadius}");
    Console.WriteLine();

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

    var optimizer = new IslandOptimizer(config, mazeTopology, gpuCap);
    var rng = new Random(42);
    var sw = Stopwatch.StartNew();
    int totalEvals = optimizer.TotalPopulation * numMazes;

    for (int gen = 0; gen < maxGenerations; gen++)
    {
        var pv = optimizer.GeneratePopulation(rng);
        var (fit, solved, _, _) = eval.EvaluateMultiMaze(
            pv, optimizer.TotalPopulation, numMazes, baseSeed: gen * 100);
        optimizer.Update(fit, pv);

        if (gen % 10 == 0 || gen == maxGenerations - 1)
        {
            float solveRate = (float)solved / totalEvals * 100f;
            Console.WriteLine($"  Gen {gen,3}: fit={fit.Max(),9:F1}  " +
                $"reached={solved,7}/{totalEvals} ({solveRate,5:F1}%)  [{sw.Elapsed.TotalSeconds:F0}s]");
        }
    }

    var (mu, _) = optimizer.GetBestSolution();
    var (champReached, champTotal) = eval.EvaluateChampion(mu, numMazes: 100, baseSeed: 999999);
    Console.WriteLine();
    Console.WriteLine($"Champion: {champReached}/{champTotal} held-out mazes reached " +
        $"({(float)champReached / champTotal * 100:F0}%)");
    Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F1}s");

    using (var fs = File.Create(outPath))
    using (var bw = new BinaryWriter(fs))
    {
        bw.Write(mu.Length);
        foreach (float v in mu) bw.Write(v);
    }
    Console.WriteLine($"Maze policy saved to: {outPath}");
    return 0;
}

static void RunTraining(string jsonPath, int maxGenerations, int numSpawns, CancellationToken cancel)
{
    string json = File.ReadAllText(jsonPath);
    var world = SimWorldLoader.FromJson(json);

    int sensorCount = world.SimulationConfig.SensorCount;
    var topology = DenseTopology.ForRocket(new[] { 16, 12 }, sensorCount: sensorCount);

    using var evaluator = new GPUDenseRocketLandingEvaluator(topology);
    evaluator.Configure(world);

    int gpuCapacity = evaluator.OptimalPopulationSize;
    Console.WriteLine($"Topology: {topology}");
    Console.WriteLine($"GPU pop: {gpuCapacity}, obstacles: {evaluator.Obstacles.Count}, sensors: {sensorCount}");
    Console.WriteLine($"Reward weights: pos={evaluator.RewardPositionWeight:F1} vel={evaluator.RewardVelocityWeight:F1} " +
        $"angle={evaluator.RewardAngleWeight:F1} angvel={evaluator.RewardAngVelWeight:F1} " +
        $"waggle={evaluator.WagglePenalty:F3}");
    Console.WriteLine();

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

    var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
    var rng = new Random(42);
    var sw = Stopwatch.StartNew();

    int bestLandings = 0;
    float bestFitness = float.MinValue;

    for (int gen = 0; gen < maxGenerations; gen++)
    {
        cancel.ThrowIfCancellationRequested();

        var paramVectors = optimizer.GeneratePopulation(rng);
        var (fitness, landings, _, _) = evaluator.EvaluateMultiSpawn(
            paramVectors, optimizer.TotalPopulation, numSpawns, baseSeed: gen * 100);
        optimizer.Update(fitness, paramVectors);
        optimizer.ManageIslands(rng);

        float maxFit = fitness.Max();
        if (maxFit > bestFitness) bestFitness = maxFit;
        if (landings > bestLandings) bestLandings = landings;

        float landingRate = (float)landings / (optimizer.TotalPopulation * numSpawns) * 100f;

        if (gen % 10 == 0 || gen == maxGenerations - 1)
        {
            Console.WriteLine($"  Gen {gen,3}: fit={maxFit,8:F1}  landings={landings,6}/{optimizer.TotalPopulation * numSpawns}" +
                $" ({landingRate,5:F1}%)  [{sw.Elapsed.TotalSeconds:F0}s]");
        }
    }

    // Evaluate champion
    var (mu, _) = optimizer.GetBestSolution();
    var (champLandings, champTotal) = evaluator.EvaluateChampion(mu, numSpawns: 100, baseSeed: 9999);
    Console.WriteLine();
    Console.WriteLine($"Champion: {champLandings}/{champTotal} landings ({(float)champLandings / champTotal * 100:F0}%)");
    Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F1}s");

    // Save champion params
    string champPath = Path.ChangeExtension(jsonPath, ".champion.bin");
    using var fs = File.Create(champPath);
    using var bw = new BinaryWriter(fs);
    bw.Write(mu.Length);
    foreach (float v in mu) bw.Write(v);
    Console.WriteLine($"Champion saved to: {champPath}");
    Console.WriteLine();
}
