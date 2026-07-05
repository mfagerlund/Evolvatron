using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// CEM on the goal-relative POSE-REACHING controller via GPUDenseRocketPoseEvaluator.
/// Verifies the new pose trainer converges (hit% rises) before the editor wires it in.
/// </summary>
public class DenseRocketPoseTest
{
    [Fact]
    public void CEM_RocketPose_SmokeTest()
    {
        var topology = DenseTopology.ForRocketPose(new[] { 24, 16 });   // 10 → 24 → 16 → 2
        using var evaluator = new GPUDenseRocketPoseEvaluator(topology);
        evaluator.MaxSteps = 400;

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

        int numSpawns = 12;
        int bestHits = 0;

        for (int gen = 0; gen < 120; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, hits, _, _) = evaluator.EvaluateMultiSpawn(
                paramVectors, optimizer.TotalPopulation, numSpawns, baseSeed: gen * 100);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            if (hits > bestHits) bestHits = hits;

            if (gen % 10 == 0)
            {
                float pct = 100f * hits / (optimizer.TotalPopulation * numSpawns);
                Console.WriteLine($"  Gen {gen}: hits={hits}/{optimizer.TotalPopulation * numSpawns} " +
                    $"({pct:F1}%)  ({sw.Elapsed.TotalSeconds:F1}s)");
            }
        }

        var (mu, _) = optimizer.GetBestSolution();
        var (champHits, champTotal) = evaluator.EvaluateChampion(mu, numSpawns: 100, baseSeed: 9999);
        Console.WriteLine($"\nChampion: {champHits}/{champTotal} pose hits across 100 targets " +
            $"({100f * champHits / champTotal:F1}%)");
        Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F1}s");

        Assert.True(champHits > 0, "Pose controller never hit a single target — trainer is not converging.");
    }
}
