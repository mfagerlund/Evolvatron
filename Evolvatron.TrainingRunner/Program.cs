using System.Diagnostics;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.World;

if (args.Length == 0)
{
    Console.WriteLine("Usage: dotnet run -- <world.json> [--generations N] [--spawns N]");
    Console.WriteLine("  Watches the JSON file for changes and restarts training automatically.");
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
