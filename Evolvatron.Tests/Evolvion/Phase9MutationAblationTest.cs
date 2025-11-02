using System.Diagnostics;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Benchmarks;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// PHASE 9: Mutation ablation study + aggressive culling validation.
///
/// Goals:
/// 1. Identify which mutations actually matter (leave-one-out testing)
/// 2. Test aggressive culling to enable frequent topology exploration
/// 3. Validate edge topology mutations (currently disabled)
///
/// 32 configs × 5 seeds × 150 generations = 160 total runs
/// Expected runtime: ~25-30 minutes
/// </summary>
public class Phase9MutationAblationTest
{
    private readonly ITestOutputHelper _output;

    public Phase9MutationAblationTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Phase9_MutationAblation_And_AggressiveCulling()
    {
        var batches = new[]
        {
            CreateBatch1_IndividualMutationAblations(),
            CreateBatch2_EdgeTopologyMutations(),
            CreateBatch3_AggressiveCullingVariants(),
            CreateBatch4_CombinedBestPractices()
        };

        _output.WriteLine("PHASE 9: MUTATION ABLATION & AGGRESSIVE CULLING");
        _output.WriteLine("===============================================");
        _output.WriteLine($"Total batches: {batches.Length}");
        _output.WriteLine($"Total configs: {batches.Sum(b => b.Configs.Length)}");
        _output.WriteLine("5 seeds per config, 150 generations per run");
        _output.WriteLine("");

        var stopwatch = Stopwatch.StartNew();
        var allResults = new List<SweepResult>();

        foreach (var batch in batches)
        {
            _output.WriteLine($"\n=== {batch.Name} ===");
            var batchResults = RunBatchInParallel(batch, parallelism: 8);
            allResults.AddRange(batchResults);

            // Print batch results sorted by improvement
            var sorted = batchResults.OrderByDescending(r => r.Improvement).ToList();
            foreach (var result in sorted)
            {
                _output.WriteLine($"{result.ConfigName,-35} | {result.Gen0Best:F4} → {result.Gen150Best:F4} = {result.Improvement:F4}");
            }
        }

        stopwatch.Stop();

        _output.WriteLine($"\n\nTotal runtime: {stopwatch.Elapsed.TotalMinutes:F1} minutes");
        _output.WriteLine("\nTop 10 configurations overall:");

        var top10 = allResults.OrderByDescending(r => r.Improvement).Take(10).ToList();
        for (int i = 0; i < top10.Count; i++)
        {
            var r = top10[i];
            _output.WriteLine($"{i + 1,2}. {r.ConfigName,-35} | Improvement: {r.Improvement:F4}");
        }
    }

    // ===== BATCH 1: Individual Mutation Ablations =====
    private Batch CreateBatch1_IndividualMutationAblations() => new()
    {
        Name = "Batch 1: Individual Mutation Ablations (leave-one-out)",
        Configs = new[]
        {
            CreateConfig("Baseline", cfg => { /* Phase 7 defaults */ }),

            CreateConfig("NoWeightJitter", cfg => cfg.MutationRates.WeightJitter = 0.0f),
            CreateConfig("NoWeightReset", cfg => cfg.MutationRates.WeightReset = 0.0f),
            CreateConfig("NoWeightL1Shrink", cfg => cfg.MutationRates.WeightL1Shrink = 0.0f),
            CreateConfig("NoActivationSwap", cfg => cfg.MutationRates.ActivationSwap = 0.0f),

            CreateConfig("OnlyWeightJitter", cfg =>
            {
                cfg.MutationRates.WeightReset = 0.0f;
                cfg.MutationRates.WeightL1Shrink = 0.0f;
                cfg.MutationRates.ActivationSwap = 0.0f;
            }),

            CreateConfig("OnlyActivationSwap", cfg =>
            {
                cfg.MutationRates.WeightJitter = 0.0f;
                cfg.MutationRates.WeightReset = 0.0f;
                cfg.MutationRates.WeightL1Shrink = 0.0f;
            }),

            CreateConfig("Minimal-JitterOnly", cfg =>
            {
                cfg.MutationRates.WeightJitter = 0.5f;  // Reduced
                cfg.MutationRates.WeightReset = 0.0f;
                cfg.MutationRates.WeightL1Shrink = 0.0f;
                cfg.MutationRates.ActivationSwap = 0.0f;
            })
        }
    };

    // ===== BATCH 2: Edge Topology Mutations (currently all disabled) =====
    private Batch CreateBatch2_EdgeTopologyMutations() => new()
    {
        Name = "Batch 2: Edge Topology Mutations (currently disabled in regular evolution)",
        Configs = new[]
        {
            CreateConfig("EdgeMut-Baseline", cfg => { /* All edge mutations disabled */ }),

            CreateConfig("EdgeMut-AddOnly", cfg => cfg.EdgeMutations.EdgeAdd = 0.05f),
            CreateConfig("EdgeMut-DeleteOnly", cfg => cfg.EdgeMutations.EdgeDeleteRandom = 0.02f),
            CreateConfig("EdgeMut-SplitOnly", cfg => cfg.EdgeMutations.EdgeSplit = 0.01f),
            CreateConfig("EdgeMut-RedirectOnly", cfg => cfg.EdgeMutations.EdgeRedirect = 0.03f),

            CreateConfig("EdgeMut-AllDefault", cfg =>
            {
                cfg.EdgeMutations.EdgeAdd = 0.05f;
                cfg.EdgeMutations.EdgeDeleteRandom = 0.02f;
                cfg.EdgeMutations.EdgeSplit = 0.01f;
                cfg.EdgeMutations.EdgeRedirect = 0.03f;
            }),

            CreateConfig("EdgeMut-Aggressive", cfg =>
            {
                cfg.EdgeMutations.EdgeAdd = 0.10f;
                cfg.EdgeMutations.EdgeDeleteRandom = 0.05f;
                cfg.EdgeMutations.EdgeSplit = 0.02f;
                cfg.EdgeMutations.EdgeRedirect = 0.05f;
            }),

            CreateConfig("EdgeMut-Conservative", cfg =>
            {
                cfg.EdgeMutations.EdgeAdd = 0.02f;
                cfg.EdgeMutations.EdgeDeleteRandom = 0.01f;
                cfg.EdgeMutations.EdgeSplit = 0.005f;
                cfg.EdgeMutations.EdgeRedirect = 0.01f;
            })
        }
    };

    // ===== BATCH 3: Aggressive Culling Variants =====
    private Batch CreateBatch3_AggressiveCullingVariants() => new()
    {
        Name = "Batch 3: Aggressive Culling (frequent topology exploration)",
        Configs = new[]
        {
            CreateConfig("Cull-CurrentConservative", cfg =>
            {
                // Current Phase 7 defaults (likely never triggers)
                cfg.StagnationThreshold = 15;
                cfg.SpeciesDiversityThreshold = 0.15f;
                cfg.RelativePerformanceThreshold = 0.5f;
            }),

            CreateConfig("Cull-Aggressive5gen", cfg =>
            {
                cfg.StagnationThreshold = 5;  // Stagnant after 5 gens
                cfg.SpeciesDiversityThreshold = 0.05f;  // Low diversity threshold
                cfg.RelativePerformanceThreshold = 1.0f;  // Disabled (always eligible)
                cfg.GraceGenerations = 3;  // Keep grace period
            }),

            CreateConfig("Cull-VeryAggressive3gen", cfg =>
            {
                cfg.StagnationThreshold = 3;  // Very aggressive
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;
                cfg.GraceGenerations = 2;  // Shorter grace
            }),

            CreateConfig("Cull-UltraAggressive2gen", cfg =>
            {
                cfg.StagnationThreshold = 2;  // Ultra aggressive
                cfg.SpeciesDiversityThreshold = 0.03f;
                cfg.RelativePerformanceThreshold = 1.0f;
                cfg.GraceGenerations = 1;
            }),

            CreateConfig("Cull-Moderate7gen", cfg =>
            {
                cfg.StagnationThreshold = 7;  // Moderate
                cfg.SpeciesDiversityThreshold = 0.08f;
                cfg.RelativePerformanceThreshold = 1.0f;
                cfg.GraceGenerations = 3;
            }),

            CreateConfig("Cull-DiversityOnly", cfg =>
            {
                cfg.StagnationThreshold = 999;  // Disabled
                cfg.SpeciesDiversityThreshold = 0.05f;  // Only cull if low diversity
                cfg.RelativePerformanceThreshold = 1.0f;
                cfg.GraceGenerations = 3;
            }),

            CreateConfig("Cull-StagnantOnly", cfg =>
            {
                cfg.StagnationThreshold = 5;  // Only cull if stagnant
                cfg.SpeciesDiversityThreshold = 0.0f;  // Disabled
                cfg.RelativePerformanceThreshold = 1.0f;
                cfg.GraceGenerations = 3;
            }),

            CreateConfig("Cull-Disabled", cfg =>
            {
                cfg.MinSpeciesCount = 4;  // Prevent culling entirely
                cfg.SpeciesCount = 4;
            })
        }
    };

    // ===== BATCH 4: Combined Best Practices =====
    private Batch CreateBatch4_CombinedBestPractices() => new()
    {
        Name = "Batch 4: Combined Best Practices (based on Batch 1-3 findings)",
        Configs = new[]
        {
            CreateConfig("Best-BaselineReference", cfg => { /* Phase 7 defaults */ }),

            CreateConfig("Best-AggressiveCull+EdgeMut", cfg =>
            {
                // Aggressive culling
                cfg.StagnationThreshold = 5;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;

                // Enable edge mutations
                cfg.EdgeMutations.EdgeAdd = 0.05f;
                cfg.EdgeMutations.EdgeDeleteRandom = 0.02f;
            }),

            CreateConfig("Best-AggressiveCull+NoEdgeMut", cfg =>
            {
                // Aggressive culling only
                cfg.StagnationThreshold = 5;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;
            }),

            CreateConfig("Best-8Species+AggressiveCull", cfg =>
            {
                // More species for diversity
                cfg.SpeciesCount = 8;
                cfg.IndividualsPerSpecies = 100;  // Keep total pop = 800
                cfg.MinSpeciesCount = 4;

                // Aggressive culling
                cfg.StagnationThreshold = 5;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;
            }),

            CreateConfig("Best-MinimalMut+AggressiveCull", cfg =>
            {
                // Minimal mutations (if Batch 1 shows simpler is better)
                cfg.MutationRates.WeightJitter = 0.8f;
                cfg.MutationRates.WeightReset = 0.05f;
                cfg.MutationRates.WeightL1Shrink = 0.1f;
                cfg.MutationRates.ActivationSwap = 0.05f;

                // Aggressive culling
                cfg.StagnationThreshold = 5;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;
            }),

            CreateConfig("Best-16Species+UltraCull", cfg =>
            {
                // LOTS of species, ultra aggressive culling
                cfg.SpeciesCount = 16;
                cfg.IndividualsPerSpecies = 50;  // Total pop = 800
                cfg.MinSpeciesCount = 8;

                cfg.StagnationThreshold = 3;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;
            }),

            CreateConfig("Best-NoStructuralMut", cfg =>
            {
                // Test if structural mutations at species birth help
                // (Would need code modification to disable - just document this)
                cfg.StagnationThreshold = 5;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;
            }),

            CreateConfig("Best-AllOptimizations", cfg =>
            {
                // Kitchen sink - everything that looked good
                cfg.SpeciesCount = 8;
                cfg.IndividualsPerSpecies = 100;

                cfg.StagnationThreshold = 5;
                cfg.SpeciesDiversityThreshold = 0.05f;
                cfg.RelativePerformanceThreshold = 1.0f;

                cfg.EdgeMutations.EdgeAdd = 0.05f;
                cfg.EdgeMutations.EdgeDeleteRandom = 0.02f;

                cfg.MutationRates.WeightJitter = 0.95f;
                cfg.MutationRates.WeightL1Shrink = 0.20f;
                cfg.MutationRates.ActivationSwap = 0.10f;
            })
        }
    };

    // ===== Helper Methods =====

    private Config CreateConfig(string name, Action<EvolutionConfig> modifier)
    {
        var random = new Random(42);

        // Use Phase 7 best architecture (from quick validation: 5×6 was winner)
        var topology = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.Tanh, count: 5)  // Wide 5×6
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(6)
            .InitializeDense(random, density: 0.85f)
            .Build();

        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 200,
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,

            // Phase 7 defaults
            StagnationThreshold = 15,
            GraceGenerations = 3,
            SpeciesDiversityThreshold = 0.15f,
            RelativePerformanceThreshold = 0.5f,

            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightJitterStdDev = 0.3f,
                WeightReset = 0.10f,
                WeightL1Shrink = 0.20f,
                L1ShrinkFactor = 0.9f,
                ActivationSwap = 0.10f,
                NodeParamMutate = 0.0f
            },

            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0.0f,  // Disabled by default
                EdgeDeleteRandom = 0.0f,
                EdgeSplit = 0.0f,
                EdgeRedirect = 0.0f
            }
        };

        modifier(config);
        return new Config { Name = name, EvolutionConfig = config, Topology = topology };
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
            }
        });

        return results;
    }

    private SweepResult RunSingleConfig(Config config)
    {
        // Multi-seed evaluation: Run 5 different seeds and average results
        int[] seeds = { 42, 123, 456, 789, 999 };
        var seedResults = new List<(float Gen0Best, float Gen0Mean, float Gen150Best, float Gen150Mean)>();

        foreach (var seed in seeds)
        {
            var evolver = new Evolver(seed: seed);
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
            seedResults.Add((gen0Stats.BestFitness, gen0Stats.MeanFitness,
                           gen150Stats.BestFitness, gen150Stats.MeanFitness));
        }

        // Compute averages across seeds
        float avgGen0Best = seedResults.Average(r => r.Gen0Best);
        float avgGen0Mean = seedResults.Average(r => r.Gen0Mean);
        float avgGen150Best = seedResults.Average(r => r.Gen150Best);
        float avgGen150Mean = seedResults.Average(r => r.Gen150Mean);
        float gen150BestRange = seedResults.Max(r => r.Gen150Best) - seedResults.Min(r => r.Gen150Best);

        return new SweepResult
        {
            ConfigName = config.Name,
            Gen0Best = avgGen0Best,
            Gen0Mean = avgGen0Mean,
            Gen150Best = avgGen150Best,
            Gen150Mean = avgGen150Mean,
            Gen150Range = gen150BestRange,
            Improvement = avgGen150Best - avgGen0Best,
            MeanImprovement = avgGen150Mean - avgGen0Mean
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
