using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Verifies that the evolutionary algorithm is fully deterministic.
/// Each seed should produce identical results across multiple runs.
/// </summary>
public class DeterminismVerificationTest
{
    private readonly ITestOutputHelper _output;

    public DeterminismVerificationTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void SameSeed_ProducesIdenticalResults()
    {
        const int maxGenerations = 50;
        const int seeds = 5;
        const int runsPerSeed = 2;

        _output.WriteLine("=== DETERMINISM VERIFICATION TEST ===");
        _output.WriteLine($"Testing {seeds} seeds, each run {runsPerSeed} times");
        _output.WriteLine($"All runs with same seed should produce identical results\n");

        var config = new EvolutionConfig();

        for (int seed = 0; seed < seeds; seed++)
        {
            _output.WriteLine($"\n=== SEED {seed} ===");

            var results = new List<RunResult>();

            for (int run = 0; run < runsPerSeed; run++)
            {
                var result = ExecuteRun(seed, maxGenerations, config);
                results.Add(result);

                _output.WriteLine($"Run {run + 1}: TopologyHash={result.TopologyHash:X8}, " +
                    $"Gen0Best={result.Gen0BestFitness:F6}, SolvedAt={result.SolvedAtGeneration?.ToString() ?? "N/A"}");
            }

            // Verify all runs with same seed are identical
            var firstRun = results[0];
            for (int run = 1; run < runsPerSeed; run++)
            {
                var currentRun = results[run];

                Assert.Equal(firstRun.TopologyHash, currentRun.TopologyHash);
                Assert.Equal(firstRun.Gen0BestFitness, currentRun.Gen0BestFitness);
                Assert.Equal(firstRun.SolvedAtGeneration, currentRun.SolvedAtGeneration);

                if (firstRun.Gen0BestFitness != currentRun.Gen0BestFitness)
                {
                    _output.WriteLine($"  ERROR: Run {run + 1} has different Gen0BestFitness! " +
                        $"Expected {firstRun.Gen0BestFitness:F6}, got {currentRun.Gen0BestFitness:F6}");
                }

                if (firstRun.TopologyHash != currentRun.TopologyHash)
                {
                    _output.WriteLine($"  ERROR: Run {run + 1} has different topology hash! " +
                        $"Expected {firstRun.TopologyHash:X8}, got {currentRun.TopologyHash:X8}");
                }

                if (firstRun.SolvedAtGeneration != currentRun.SolvedAtGeneration)
                {
                    _output.WriteLine($"  ERROR: Run {run + 1} solved at different generation! " +
                        $"Expected {firstRun.SolvedAtGeneration}, got {currentRun.SolvedAtGeneration}");
                }
            }

            _output.WriteLine($"  ✓ All {runsPerSeed} runs identical for seed {seed}");
        }

        _output.WriteLine("\n✓ Determinism verification complete!");
    }

    private RunResult ExecuteRun(int seed, int maxGenerations, EvolutionConfig config)
    {
        var evolver = new Evolver(seed);
        var random = new Random(seed);

        var topology = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)
            .Build();

        int topologyHash = ComputeTopologyHash(topology);

        var population = evolver.InitializePopulation(config, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Initial evaluation
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();
        float gen0BestFitness = gen0Stats.BestFitness;

        // Evolution loop
        int? solvedAtGeneration = null;
        for (int gen = 1; gen <= maxGenerations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: 0);
            evolver.StepGeneration(population);

            var stats = population.GetStatistics();

            if (stats.BestFitness > -0.01f) // MSE < 0.01
            {
                solvedAtGeneration = gen;
                break;
            }
        }

        return new RunResult
        {
            TopologyHash = topologyHash,
            Gen0BestFitness = gen0BestFitness,
            SolvedAtGeneration = solvedAtGeneration
        };
    }

    private static int ComputeTopologyHash(SpeciesSpec topology)
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + topology.TotalNodes;
            hash = hash * 31 + topology.TotalEdges;

            foreach (var edge in topology.Edges.OrderBy(e => e.Source).ThenBy(e => e.Dest))
            {
                hash = hash * 31 + edge.Source;
                hash = hash * 31 + edge.Dest;
            }

            return hash;
        }
    }

    private class RunResult
    {
        public int TopologyHash { get; set; }
        public float Gen0BestFitness { get; set; }
        public int? SolvedAtGeneration { get; set; }
    }
}
