using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// POST-BIAS-FIX CRITICAL TEST: Sparse vs Dense Initial Topology
///
/// PRE-bias-fix result: "Dense beats sparse by 12x"
/// BUT: That was when biases were frozen! Now that biases can adapt,
/// NEAT's sparse-to-dense philosophy may actually work.
///
/// This test answers THE most important question:
/// Should we start with minimal topology (NEAT-style) or fully connected?
///
/// 8 configs, 150 generations each, ~15 minutes total
/// </summary>
public class SparseDensitySweepTest
{
    private readonly ITestOutputHelper _output;

    public SparseDensitySweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact] // Critical post-bias-fix test
    public void SparseDensitySweep_PostBiasFix()
    {
        var batch = CreateSparseDensityBatch();

        _output.WriteLine("SPARSE vs DENSE TOPOLOGY SWEEP (POST-BIAS-FIX)");
        _output.WriteLine("==============================================");
        _output.WriteLine($"Testing {batch.Configs.Length} configurations with 150 generations each");
        _output.WriteLine($"Parallelism: 8 threads");
        _output.WriteLine($"Architecture: 2→3→3→3→3→3→3→1 (6-layer narrow, Tanh-only)");
        _output.WriteLine("");

        var startTime = DateTime.Now;
        var results = RunBatchInParallel(batch, parallelism: 8);
        var elapsed = DateTime.Now - startTime;

        // Print results
        _output.WriteLine("");
        _output.WriteLine($"RESULTS (completed in {elapsed.TotalMinutes:F1} minutes)");
        _output.WriteLine("=".PadRight(80, '='));
        _output.WriteLine("");

        var sorted = results.OrderByDescending(r => r.Improvement).ToList();

        foreach (var result in sorted)
        {
            _output.WriteLine($"{result.ConfigName,-30} | Gen0: {result.Gen0Best:F4} → Gen150: {result.Gen150Best:F4} | Δ: {result.Improvement:F4}");
        }

        _output.WriteLine("");
        _output.WriteLine("KEY FINDINGS");
        _output.WriteLine("=".PadRight(80, '='));
        _output.WriteLine($"WINNER: {sorted[0].ConfigName}");
        _output.WriteLine($"  Final Fitness: {sorted[0].Gen150Best:F4}");
        _output.WriteLine($"  Improvement: {sorted[0].Improvement:F4}");
        _output.WriteLine("");

        // Compare to fully dense (1.0)
        var fullyDense = sorted.FirstOrDefault(r => r.ConfigName.Contains("1.0"));
        if (fullyDense != null)
        {
            _output.WriteLine("COMPARISON TO FULLY DENSE (1.0):");
            foreach (var result in sorted.Where(r => r != fullyDense))
            {
                double ratio = result.Improvement / (fullyDense.Improvement + 0.0001f);
                string verdict = ratio > 1.05 ? "BETTER" : (ratio < 0.95 ? "WORSE" : "SIMILAR");
                _output.WriteLine($"  {result.ConfigName}: {ratio:F2}x ({verdict})");
            }
        }

        _output.WriteLine("");
        _output.WriteLine("CONCLUSION");
        _output.WriteLine("=".PadRight(80, '='));
        if (sorted[0].ConfigName.Contains("1.0") || sorted[0].ConfigName.Contains("0.95") || sorted[0].ConfigName.Contains("0.85"))
        {
            _output.WriteLine("Dense initialization still wins post-bias-fix.");
            _output.WriteLine("Recommendation: Keep using dense (0.75-1.0) initialization.");
        }
        else
        {
            _output.WriteLine("⚠️ SPARSE WINS! Bias fix unlocked NEAT-style sparse-to-dense evolution!");
            _output.WriteLine($"Recommendation: Use {sorted[0].ConfigName} for initialization.");
        }
    }

    private Batch CreateSparseDensityBatch()
    {
        var densities = new[]
        {
            ("density:0.1", 0.1f),    // Very sparse start (NEAT-like)
            ("density:0.2", 0.2f),    // Sparse
            ("density:0.3", 0.3f),    // Moderate-sparse
            ("density:0.5", 0.5f),    // Half connected
            ("density:0.7", 0.7f),    // Moderate-dense
            ("density:0.85", 0.85f),  // Dense
            ("density:0.95", 0.95f),  // Very dense
            ("density:1.0", 1.0f),    // Fully connected (current default)
        };

        return new Batch
        {
            Name = "Initial Topology Density",
            Configs = densities.Select(d => new Config
            {
                Name = d.Item1,
                EvolutionConfig = CreateBaseConfig(),
                Topology = CreateTopologyWithDensity(42, d.Item2)
            }).ToArray()
        };
    }

    private EvolutionConfig CreateBaseConfig()
    {
        // Use Phase 6 winner config as baseline
        return new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 200, // Total pop: 800
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,
            GraceGenerations = 3,
            StagnationThreshold = 15,
            SpeciesDiversityThreshold = 0.15f,
            RelativePerformanceThreshold = 0.5f,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightJitterStdDev = 0.3f,
                WeightReset = 0.10f,
                WeightL1Shrink = 0.1f,
                L1ShrinkFactor = 0.9f,
                ActivationSwap = 0.0f, // Tanh-only
                NodeParamMutate = 0.0f, // Tanh has no params
                NodeParamStdDev = 0.1f
            },
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0.05f,      // Allow growth
                EdgeDeleteRandom = 0.02f
            },
        };
    }

    private SpeciesSpec CreateTopologyWithDensity(int seed, float density)
    {
        var random = new Random(seed);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(6) // Allow up to 6 incoming connections
            .InitializeDense(random, density: density) // KEY PARAMETER
            .Build();
    }

    private List<SweepResult> RunBatchInParallel(Batch batch, int parallelism)
    {
        var results = new List<SweepResult>();
        var options = new ParallelOptions { MaxDegreeOfParallelism = parallelism };

        Parallel.ForEach(batch.Configs, options, config =>
        {
            var result = RunSingleConfig(config);
            lock (results)
            {
                results.Add(result);
                _output.WriteLine($"[{DateTime.Now:HH:mm:ss}] Completed: {config.Name} → {result.Gen150Best:F4}");
            }
        });

        return results;
    }

    private SweepResult RunSingleConfig(Config config)
    {
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config.EvolutionConfig, config.Topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        for (int gen = 1; gen <= 150; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen150Stats = population.GetStatistics();

        return new SweepResult
        {
            ConfigName = config.Name,
            Gen0Best = gen0Stats.BestFitness,
            Gen0Mean = gen0Stats.MeanFitness,
            Gen150Best = gen150Stats.BestFitness,
            Gen150Mean = gen150Stats.MeanFitness,
            Gen150Range = gen150Stats.BestFitness - gen150Stats.WorstFitness,
            Improvement = gen150Stats.BestFitness - gen0Stats.BestFitness,
            MeanImprovement = gen150Stats.MeanFitness - gen0Stats.MeanFitness
        };
    }

    private record Batch
    {
        public string Name { get; init; } = "";
        public Config[] Configs { get; init; } = Array.Empty<Config>();
    }

    private record Config
    {
        public string Name { get; init; } = "";
        public EvolutionConfig EvolutionConfig { get; init; } = new();
        public SpeciesSpec Topology { get; init; } = null!;
    }

    private record SweepResult
    {
        public string ConfigName { get; init; } = "";
        public float Gen0Best { get; init; }
        public float Gen0Mean { get; init; }
        public float Gen150Best { get; init; }
        public float Gen150Mean { get; init; }
        public float Gen150Range { get; init; }
        public float Improvement { get; init; }
        public float MeanImprovement { get; init; }
    }
}
