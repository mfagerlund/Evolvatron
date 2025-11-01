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
/// 30-minute comprehensive parameter sweep testing untested parameters.
/// Tests 10 batches × 5 configs = 50 total configurations across key dimensions.
/// Focuses on parameters that exist in current codebase and were never tested.
/// </summary>
public class ThirtyMinuteSweepTest
{
    private readonly ITestOutputHelper _output;

    public ThirtyMinuteSweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void ThirtyMinuteSweep()
    {
        var batches = new[]
        {
            CreateBatch1_WeightJitterStdDev(),
            CreateBatch2_ParentPoolPercentage(),
            CreateBatch3_PopulationVsTournament(),
            CreateBatch4_EliteCount(),
            CreateBatch5_SpeciesCount(),
            CreateBatch6_FunnelArchitectures(),
            CreateBatch7_DepthVsWidth(),
            CreateBatch8_EdgeMutationRates(),
            CreateBatch9_WeightResetRate(),
            CreateBatch10_BestConfigStability()
        };

        _output.WriteLine("30-Minute Sweep Results");
        _output.WriteLine("========================\n");
        _output.WriteLine("Testing untested parameters with bias mutation bug FIXED\n");

        var allResults = new List<(string BatchName, string ConfigName, float Improvement, float Baseline)>();

        // Run each batch sequentially (configs within batch parallel)
        for (int i = 0; i < batches.Length; i++)
        {
            var batch = batches[i];
            _output.WriteLine($"\nBatch {i + 1}/{batches.Length}: {batch.Name}");
            _output.WriteLine(new string('-', 80));

            var results = RunBatchInParallel(batch);

            // Find winner and baseline
            var winner = results.OrderByDescending(r => r.Improvement).First();
            var baseline = results.FirstOrDefault(r => r.ConfigName.Contains("Current") || r.ConfigName.Contains("Baseline"));
            baseline ??= results.First(); // Fallback

            float percentImprovement = Math.Abs(baseline.Improvement) > 0.001f
                ? ((winner.Improvement - baseline.Improvement) / Math.Abs(baseline.Improvement) * 100f)
                : 0f;

            _output.WriteLine($"\nWinner: {winner.ConfigName}");
            _output.WriteLine($"  Improvement: {winner.Improvement:F4} (Gen0: {winner.Gen0Best:F4} → Gen100: {winner.Gen100Best:F4})");
            _output.WriteLine($"  vs Baseline: {percentImprovement:+0.0}%");
            _output.WriteLine($"  Gen100 fitness range: {winner.Gen100Range:F4}\n");

            // Print all results
            foreach (var r in results.OrderByDescending(r => r.Improvement))
            {
                _output.WriteLine($"  {r.ConfigName,-35} {r.Improvement,9:F4} (Gen100: {r.Gen100Best:F4})");
                allResults.Add((batch.Name, r.ConfigName, r.Improvement, baseline.Improvement));
            }
        }

        // Summary
        _output.WriteLine($"\n{new string('=', 80)}");
        _output.WriteLine("Top 10 Improvements Found:");
        _output.WriteLine(new string('=', 80));

        var topImprovements = allResults
            .Select(r => new {
                Batch = r.BatchName,
                Config = r.ConfigName,
                Improvement = r.Improvement,
                Baseline = r.Baseline,
                Gain = Math.Abs(r.Baseline) > 0.001f ? (r.Improvement - r.Baseline) / Math.Abs(r.Baseline) * 100f : 0f
            })
            .OrderByDescending(r => r.Gain)
            .Take(10)
            .ToList();

        for (int i = 0; i < topImprovements.Count; i++)
        {
            var r = topImprovements[i];
            _output.WriteLine($"{i + 1,2}. {r.Config,-40} {r.Gain:+0.0}% ({r.Batch})");
        }
    }

    private Batch CreateBatch1_WeightJitterStdDev() => new Batch
    {
        Name = "WeightJitter StdDev (NEVER TESTED - highest impact potential)",
        Configs = new[]
        {
            CreateConfig("WJStdDev-0.05", cfg => cfg.MutationRates.WeightJitterStdDev = 0.05f),
            CreateConfig("WJStdDev-0.1", cfg => cfg.MutationRates.WeightJitterStdDev = 0.1f),
            CreateConfig("WJStdDev-0.3-Current", cfg => cfg.MutationRates.WeightJitterStdDev = 0.3f),
            CreateConfig("WJStdDev-0.5", cfg => cfg.MutationRates.WeightJitterStdDev = 0.5f),
            CreateConfig("WJStdDev-1.0", cfg => cfg.MutationRates.WeightJitterStdDev = 1.0f)
        }
    };

    private Batch CreateBatch2_ParentPoolPercentage() => new Batch
    {
        Name = "Parent Pool Percentage (NEVER TESTED - amplifies selection)",
        Configs = new[]
        {
            CreateConfig("ParentPool-1.00-Current", cfg => cfg.ParentPoolPercentage = 1.0f),
            CreateConfig("ParentPool-0.75", cfg => cfg.ParentPoolPercentage = 0.75f),
            CreateConfig("ParentPool-0.50", cfg => cfg.ParentPoolPercentage = 0.50f),
            CreateConfig("ParentPool-0.25", cfg => cfg.ParentPoolPercentage = 0.25f),
            CreateConfig("ParentPool-0.10", cfg => cfg.ParentPoolPercentage = 0.10f)
        }
    };

    private Batch CreateBatch3_PopulationVsTournament() => new Batch
    {
        Name = "Population × Tournament Trade-off",
        Configs = new[]
        {
            CreateConfig("Pop400-T4", cfg => { cfg.SpeciesCount = 4; cfg.IndividualsPerSpecies = 100; cfg.TournamentSize = 4; }),
            CreateConfig("Pop800-T16-Current", cfg => { cfg.SpeciesCount = 8; cfg.IndividualsPerSpecies = 100; cfg.TournamentSize = 16; }),
            CreateConfig("Pop1600-T32", cfg => { cfg.SpeciesCount = 16; cfg.IndividualsPerSpecies = 100; cfg.TournamentSize = 32; }),
            CreateConfig("Pop200-T8", cfg => { cfg.SpeciesCount = 2; cfg.IndividualsPerSpecies = 100; cfg.TournamentSize = 8; }),
            CreateConfig("Pop1200-T24", cfg => { cfg.SpeciesCount = 12; cfg.IndividualsPerSpecies = 100; cfg.TournamentSize = 24; })
        }
    };

    private Batch CreateBatch4_EliteCount() => new Batch
    {
        Name = "Elite Count",
        Configs = new[]
        {
            CreateConfig("Elites-0", cfg => cfg.Elites = 0),
            CreateConfig("Elites-1", cfg => cfg.Elites = 1),
            CreateConfig("Elites-2-Current", cfg => cfg.Elites = 2),
            CreateConfig("Elites-4", cfg => cfg.Elites = 4),
            CreateConfig("Elites-8", cfg => cfg.Elites = 8)
        }
    };

    private Batch CreateBatch5_SpeciesCount() => new Batch
    {
        Name = "Species Count",
        Configs = new[]
        {
            CreateConfig("Species-4", cfg => cfg.SpeciesCount = 4),
            CreateConfig("Species-8-Current", cfg => cfg.SpeciesCount = 8),
            CreateConfig("Species-16", cfg => cfg.SpeciesCount = 16),
            CreateConfig("Species-32", cfg => cfg.SpeciesCount = 32),
            CreateConfig("Species-2", cfg => cfg.SpeciesCount = 2)
        }
    };

    private Batch CreateBatch6_FunnelArchitectures() => new Batch
    {
        Name = "Funnel Architectures (Tanh-only, Dense)",
        Configs = new[]
        {
            CreateArchitectureConfig("Baseline-2-6-6-1", new[] { 6, 6 }),
            CreateArchitectureConfig("Funnel-2-16-8-4-1", new[] { 16, 8, 4 }),
            CreateArchitectureConfig("Funnel-2-12-8-4-1", new[] { 12, 8, 4 }),
            CreateArchitectureConfig("Funnel-2-8-6-4-1", new[] { 8, 6, 4 }),
            CreateArchitectureConfig("Reverse-2-4-8-12-1", new[] { 4, 8, 12 })
        }
    };

    private Batch CreateBatch7_DepthVsWidth() => new Batch
    {
        Name = "Depth vs Width (same ~60-80 edges)",
        Configs = new[]
        {
            CreateArchitectureConfig("Wide-2-12-12-1", new[] { 12, 12 }),
            CreateArchitectureConfig("Baseline-2-6-6-1", new[] { 6, 6 }),
            CreateArchitectureConfig("Deep-2-4-4-4-4-1", new[] { 4, 4, 4, 4 }),
            CreateArchitectureConfig("Medium-2-8-8-1", new[] { 8, 8 }),
            CreateArchitectureConfig("VeryDeep-2-3-3-3-3-3-1", new[] { 3, 3, 3, 3, 3 })
        }
    };

    private Batch CreateBatch8_EdgeMutationRates() => new Batch
    {
        Name = "Edge Topology Mutation Rates",
        Configs = new[]
        {
            CreateConfig("EdgeMutations-Disabled", cfg => { cfg.EdgeMutations.EdgeAdd = 0.0f; cfg.EdgeMutations.EdgeDeleteRandom = 0.0f; }),
            CreateConfig("EdgeMutations-VeryLow", cfg => { cfg.EdgeMutations.EdgeAdd = 0.01f; cfg.EdgeMutations.EdgeDeleteRandom = 0.005f; }),
            CreateConfig("EdgeMutations-Low-Current", cfg => { cfg.EdgeMutations.EdgeAdd = 0.05f; cfg.EdgeMutations.EdgeDeleteRandom = 0.02f; }),
            CreateConfig("EdgeMutations-Medium", cfg => { cfg.EdgeMutations.EdgeAdd = 0.10f; cfg.EdgeMutations.EdgeDeleteRandom = 0.05f; }),
            CreateConfig("EdgeMutations-High", cfg => { cfg.EdgeMutations.EdgeAdd = 0.20f; cfg.EdgeMutations.EdgeDeleteRandom = 0.10f; })
        }
    };

    private Batch CreateBatch9_WeightResetRate() => new Batch
    {
        Name = "Weight Reset Rate",
        Configs = new[]
        {
            CreateConfig("WeightReset-0.01", cfg => cfg.MutationRates.WeightReset = 0.01f),
            CreateConfig("WeightReset-0.05", cfg => cfg.MutationRates.WeightReset = 0.05f),
            CreateConfig("WeightReset-0.10-Current", cfg => cfg.MutationRates.WeightReset = 0.10f),
            CreateConfig("WeightReset-0.20", cfg => cfg.MutationRates.WeightReset = 0.20f),
            CreateConfig("WeightReset-0.50", cfg => cfg.MutationRates.WeightReset = 0.50f)
        }
    };

    private Batch CreateBatch10_BestConfigStability() => new Batch
    {
        Name = "Best Config Stability (different seeds)",
        Configs = new[]
        {
            CreateConfigWithSeed("Seed-42-Current", 42),
            CreateConfigWithSeed("Seed-123", 123),
            CreateConfigWithSeed("Seed-456", 456),
            CreateConfigWithSeed("Seed-789", 789),
            CreateConfigWithSeed("Seed-999", 999)
        }
    };

    private Config CreateConfig(string name, Action<EvolutionConfig> customize)
    {
        var evolutionConfig = CreateBestConfig();
        customize(evolutionConfig);
        return new Config
        {
            Name = name,
            EvolutionConfig = evolutionConfig,
            Topology = CreateBestTopology(42) // Fixed seed for consistency
        };
    }

    private Config CreateArchitectureConfig(string name, int[] hiddenLayers)
    {
        return new Config
        {
            Name = name,
            EvolutionConfig = CreateBestConfig(),
            Topology = CreateTopology(42, hiddenLayers)
        };
    }

    private Config CreateConfigWithSeed(string name, int seed)
    {
        return new Config
        {
            Name = name,
            EvolutionConfig = CreateBestConfig(),
            Topology = CreateBestTopology(seed) // Vary seed for stability test
        };
    }

    private EvolutionConfig CreateBestConfig()
    {
        // Best config from Phase 5: Tanh-only, Dense 2→6→6→1, T=16, WJ=0.95
        // With bias mutation bug NOW FIXED
        return new EvolutionConfig
        {
            SpeciesCount = 8,
            IndividualsPerSpecies = 100,
            Elites = 2,
            TournamentSize = 16, // Critical parameter from Phase 2
            ParentPoolPercentage = 1.0f, // NEVER TESTED - high impact potential
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f, // From Phase 2
                WeightJitterStdDev = 0.3f, // NEVER TESTED - highest impact potential
                WeightReset = 0.10f,
                WeightL1Shrink = 0.1f,
                L1ShrinkFactor = 0.9f,
                ActivationSwap = 0.01f,
                NodeParamMutate = 0.2f,
                NodeParamStdDev = 0.1f
            },
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0.05f,
                EdgeDeleteRandom = 0.02f
            },
        };
    }

    private SpeciesSpec CreateBestTopology(int seed)
    {
        // Best from Phase 4-5: 2→6→6→1, Dense, Tanh-only
        return CreateTopology(seed, new[] { 6, 6 });
    }

    private SpeciesSpec CreateTopology(int seed, int[] hiddenLayers)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2); // Spiral: (x, y)

        foreach (var hiddenSize in hiddenLayers)
        {
            builder.AddHiddenRow(hiddenSize, ActivationType.Tanh); // Tanh-only
        }

        builder.AddOutputRow(1, ActivationType.Tanh) // Binary classification
            .WithMaxInDegree(12)
            .InitializeDense(random, density: 1.0f); // Dense initialization

        return builder.Build();
    }

    private List<SweepResult> RunBatchInParallel(Batch batch)
    {
        var results = new List<SweepResult>();
        var options = new ParallelOptions { MaxDegreeOfParallelism = 5 };

        Parallel.ForEach(batch.Configs, options, config =>
        {
            var result = RunSingleConfig(config);
            lock (results)
            {
                results.Add(result);
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

        // Generation 0
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        // Run 100 generations
        for (int gen = 1; gen <= 100; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen100Stats = population.GetStatistics();

        return new SweepResult
        {
            ConfigName = config.Name,
            Gen0Best = gen0Stats.BestFitness,
            Gen0Mean = gen0Stats.MeanFitness,
            Gen100Best = gen100Stats.BestFitness,
            Gen100Mean = gen100Stats.MeanFitness,
            Gen100Range = gen100Stats.BestFitness - gen100Stats.WorstFitness,
            Improvement = gen100Stats.BestFitness - gen0Stats.BestFitness,
            MeanImprovement = gen100Stats.MeanFitness - gen0Stats.MeanFitness
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
        public float Gen100Best { get; init; }
        public float Gen100Mean { get; init; }
        public float Gen100Range { get; init; }
        public float Improvement { get; init; }
        public float MeanImprovement { get; init; }
    }
}
