using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// CEM on rocket landing via GPUDenseRocketLandingEvaluator.
/// Tests whether the CEM results from DPNV transfer to the harder rocket landing task.
/// Uses generation budget (immune to GPU contention).
/// </summary>
public class DenseRocketLandingTest
{
    /// <summary>
    /// Smoke test: CEM + dense NN on basic rocket landing (no obstacles).
    /// 100 generations, report landing rate.
    /// </summary>
    [Fact]
    public void CEM_RocketLanding_SmokeTest()
    {
        // 8 inputs → 16 → 8 → 2 outputs (no sensors for smoke test)
        var topology = DenseTopology.ForRocket(new[] { 16, 8 });
        using var evaluator = new GPUDenseRocketLandingEvaluator(topology);
        evaluator.MaxSteps = 600;

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");

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
        int numSpawns = 10; // train on 10 spawn conditions

        for (int gen = 0; gen < 100; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, landings, _, _) = evaluator.EvaluateMultiSpawn(
                paramVectors, optimizer.TotalPopulation, numSpawns, baseSeed: gen * 100);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            if (landings > bestLandings)
                bestLandings = landings;

            if (gen % 20 == 0)
                Console.WriteLine($"  Gen {gen}: landings={landings}/{optimizer.TotalPopulation * numSpawns} " +
                    $"({sw.Elapsed.TotalSeconds:F1}s)");
        }

        // Test champion
        var (mu, _) = optimizer.GetBestSolution();
        var (champLandings, champTotal) = evaluator.EvaluateChampion(mu, numSpawns: 50, baseSeed: 9999);
        Console.WriteLine($"\nChampion: {champLandings}/{champTotal} landings across 50 spawns");
        Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F1}s");
    }

    /// <summary>
    /// Full CEM rocket landing with sensors and obstacles.
    /// 300 generations, multi-spawn(10), reports champion landing rate.
    /// </summary>
    [Fact]
    public void CEM_RocketLanding_WithObstacles()
    {
        // 12 inputs (8 base + 4 sensors) → 16 → 12 → 2 outputs
        var topology = DenseTopology.ForRocket(new[] { 16, 12 }, sensorCount: 4);
        using var evaluator = new GPUDenseRocketLandingEvaluator(topology);
        evaluator.MaxSteps = 900;
        evaluator.SensorCount = 4;
        evaluator.MaxSensorRange = 30f;
        evaluator.ObstacleDeathEnabled = true;
        evaluator.SpawnXRange = 5f;
        evaluator.SpawnAngleRange = 5f * MathF.PI / 180f;

        // Funnel obstacles
        float wallAngle = 25f * MathF.PI / 180f;
        evaluator.Obstacles = new List<Core.GPU.GPUOBBCollider>
        {
            new() { CX = -8f, CY = 8f,
                UX = MathF.Cos(wallAngle), UY = MathF.Sin(wallAngle),
                HalfExtentX = 4f, HalfExtentY = 0.3f },
            new() { CX = 8f, CY = 8f,
                UX = MathF.Cos(-wallAngle), UY = MathF.Sin(-wallAngle),
                HalfExtentX = 4f, HalfExtentY = 0.3f },
        };

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"Obstacles: {evaluator.Obstacles.Count}, sensors: {evaluator.SensorCount}");

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

        int numSpawns = 10;

        for (int gen = 0; gen < 300; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, landings, _, _) = evaluator.EvaluateMultiSpawn(
                paramVectors, optimizer.TotalPopulation, numSpawns, baseSeed: gen * 100);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            if (gen % 50 == 0)
                Console.WriteLine($"  Gen {gen}: landings={landings}/{optimizer.TotalPopulation * numSpawns} " +
                    $"({sw.Elapsed.TotalSeconds:F1}s)");
        }

        var (mu, _) = optimizer.GetBestSolution();
        var (champLandings, champTotal) = evaluator.EvaluateChampion(mu, numSpawns: 100, baseSeed: 9999);
        Console.WriteLine($"\nChampion: {champLandings}/{champTotal} landings across 100 spawns");
        Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F1}s");
    }
}
