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
/// One-hour comprehensive sweep testing ALL untested/minimally-tested parameters.
/// 15 batches × 8 configs = 120 total configurations
/// 8-thread parallelism, 150 generations per config
/// Post-bias-fix validation
/// </summary>
public class OneHourSweepTest
{
    private readonly ITestOutputHelper _output;

    public OneHourSweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Skip = "Long-running test (60 min) - run in separate context")]
    public void OneHourComprehensiveSweep()
    {
        var batches = new[]
        {
            CreateBatch1_StagnationThreshold(),
            CreateBatch2_GraceGenerations(),
            CreateBatch3_SpeciesDiversityThreshold(),
            CreateBatch4_RelativePerformanceThreshold(),
            CreateBatch5_WeightL1ShrinkRate(),
            CreateBatch6_L1ShrinkFactor(),
            CreateBatch7_ActivationSwapRate(),
            CreateBatch8_NodeParamMutation(),
            CreateBatch9_SeedsPerIndividual(),
            CreateBatch10_FitnessAggregation(),
            CreateBatch11_WeightInitialization(),
            CreateBatch12_CombinedCulling(),
            CreateBatch13_MutationCombinations(),
            CreateBatch14_DeepArchitectureVariations(),
            CreateBatch15_ExtremeDeepNetworks()
        };

        _output.WriteLine("ONE-HOUR COMPREHENSIVE SWEEP");
        _output.WriteLine("============================");
        _output.WriteLine($"Total batches: {batches.Length}");
        _output.WriteLine($"Configs per batch: 8");
        _output.WriteLine($"Parallelism: 8 threads");
        _output.WriteLine($"Generations: 150 per config");
        _output.WriteLine($"Expected duration: ~60 minutes\n");

        var allResults = new List<(string BatchName, string ConfigName, float Improvement, float Baseline)>();

        for (int i = 0; i < batches.Length; i++)
        {
            var batch = batches[i];
            _output.WriteLine($"\nBatch {i + 1}/{batches.Length}: {batch.Name}");
            _output.WriteLine(new string('-', 80));

            var startTime = DateTime.Now;
            var results = RunBatchInParallel(batch, parallelism: 8);
            var elapsed = DateTime.Now - startTime;

            var winner = results.OrderByDescending(r => r.Improvement).First();
            var baseline = results.FirstOrDefault(r => r.ConfigName.Contains("Current") || r.ConfigName.Contains("Default"));
            baseline ??= results.First();

            float percentImprovement = Math.Abs(baseline.Improvement) > 0.001f
                ? ((winner.Improvement - baseline.Improvement) / Math.Abs(baseline.Improvement) * 100f)
                : 0f;

            _output.WriteLine($"\nWinner: {winner.ConfigName}");
            _output.WriteLine($"  Gen0: {winner.Gen0Best:F4}  Gen150: {winner.Gen150Best:F4}  Improvement: {winner.Improvement:F4} ({percentImprovement:+0.0}%)");
            _output.WriteLine($"  Batch time: {elapsed.TotalMinutes:F1} minutes\n");

            foreach (var r in results.OrderByDescending(r => r.Improvement))
            {
                _output.WriteLine($"  {r.ConfigName,-40} {r.Improvement,9:F4} (Gen150: {r.Gen150Best:F4})");
                allResults.Add((batch.Name, r.ConfigName, r.Improvement, baseline.Improvement));
            }
        }

        // Summary
        _output.WriteLine($"\n{new string('=', 80)}");
        _output.WriteLine("TOP 20 IMPROVEMENTS FOUND");
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
            .Take(20)
            .ToList();

        for (int i = 0; i < topImprovements.Count; i++)
        {
            var r = topImprovements[i];
            _output.WriteLine($"{i + 1,3}. {r.Config,-45} {r.Gain:+0.1}% ({r.Batch})");
        }

        _output.WriteLine($"\n{new string('=', 80)}");
        _output.WriteLine("RECOMMENDED CONFIGURATION CHANGES");
        _output.WriteLine(new string('=', 80));

        var significantChanges = topImprovements.Where(r => r.Gain > 5.0).ToList();
        if (significantChanges.Any())
        {
            foreach (var change in significantChanges)
            {
                _output.WriteLine($"• {change.Config,-45} +{change.Gain:F1}%");
            }
        }
        else
        {
            _output.WriteLine("No changes with >5% improvement found. Current config is well-optimized!");
        }
    }

    // BATCH 1: Stagnation Threshold
    private Batch CreateBatch1_StagnationThreshold() => new Batch
    {
        Name = "StagnationThreshold (culling trigger)",
        Configs = new[]
        {
            CreateConfig("Stagnation-5", cfg => cfg.StagnationThreshold = 5),
            CreateConfig("Stagnation-10", cfg => cfg.StagnationThreshold = 10),
            CreateConfig("Stagnation-15-Default", cfg => cfg.StagnationThreshold = 15),
            CreateConfig("Stagnation-20", cfg => cfg.StagnationThreshold = 20),
            CreateConfig("Stagnation-30", cfg => cfg.StagnationThreshold = 30),
            CreateConfig("Stagnation-50", cfg => cfg.StagnationThreshold = 50),
            CreateConfig("Stagnation-100", cfg => cfg.StagnationThreshold = 100),
            CreateConfig("Stagnation-Infinite", cfg => cfg.StagnationThreshold = int.MaxValue)
        }
    };

    // BATCH 2: Grace Generations
    private Batch CreateBatch2_GraceGenerations() => new Batch
    {
        Name = "GraceGenerations (new species protection)",
        Configs = new[]
        {
            CreateConfig("Grace-0", cfg => cfg.GraceGenerations = 0),
            CreateConfig("Grace-1", cfg => cfg.GraceGenerations = 1),
            CreateConfig("Grace-3-Default", cfg => cfg.GraceGenerations = 3),
            CreateConfig("Grace-5", cfg => cfg.GraceGenerations = 5),
            CreateConfig("Grace-10", cfg => cfg.GraceGenerations = 10),
            CreateConfig("Grace-20", cfg => cfg.GraceGenerations = 20),
            CreateConfig("Grace-30", cfg => cfg.GraceGenerations = 30),
            CreateConfig("Grace-50", cfg => cfg.GraceGenerations = 50)
        }
    };

    // BATCH 3: Species Diversity Threshold
    private Batch CreateBatch3_SpeciesDiversityThreshold() => new Batch
    {
        Name = "SpeciesDiversityThreshold (fitness variance min)",
        Configs = new[]
        {
            CreateConfig("Diversity-0.01", cfg => cfg.SpeciesDiversityThreshold = 0.01f),
            CreateConfig("Diversity-0.05", cfg => cfg.SpeciesDiversityThreshold = 0.05f),
            CreateConfig("Diversity-0.10", cfg => cfg.SpeciesDiversityThreshold = 0.10f),
            CreateConfig("Diversity-0.15-Default", cfg => cfg.SpeciesDiversityThreshold = 0.15f),
            CreateConfig("Diversity-0.25", cfg => cfg.SpeciesDiversityThreshold = 0.25f),
            CreateConfig("Diversity-0.50", cfg => cfg.SpeciesDiversityThreshold = 0.50f),
            CreateConfig("Diversity-1.00", cfg => cfg.SpeciesDiversityThreshold = 1.00f),
            CreateConfig("Diversity-Infinite", cfg => cfg.SpeciesDiversityThreshold = float.MaxValue)
        }
    };

    // BATCH 4: Relative Performance Threshold
    private Batch CreateBatch4_RelativePerformanceThreshold() => new Batch
    {
        Name = "RelativePerformanceThreshold (performance ratio)",
        Configs = new[]
        {
            CreateConfig("RelPerf-0.10", cfg => cfg.RelativePerformanceThreshold = 0.10f),
            CreateConfig("RelPerf-0.25", cfg => cfg.RelativePerformanceThreshold = 0.25f),
            CreateConfig("RelPerf-0.33", cfg => cfg.RelativePerformanceThreshold = 0.33f),
            CreateConfig("RelPerf-0.50-Default", cfg => cfg.RelativePerformanceThreshold = 0.50f),
            CreateConfig("RelPerf-0.67", cfg => cfg.RelativePerformanceThreshold = 0.67f),
            CreateConfig("RelPerf-0.75", cfg => cfg.RelativePerformanceThreshold = 0.75f),
            CreateConfig("RelPerf-0.90", cfg => cfg.RelativePerformanceThreshold = 0.90f),
            CreateConfig("RelPerf-1.00", cfg => cfg.RelativePerformanceThreshold = 1.00f)
        }
    };

    // BATCH 5: WeightL1Shrink Rate
    private Batch CreateBatch5_WeightL1ShrinkRate() => new Batch
    {
        Name = "WeightL1Shrink Rate",
        Configs = new[]
        {
            CreateConfig("L1Rate-0.00", cfg => cfg.MutationRates.WeightL1Shrink = 0.0f),
            CreateConfig("L1Rate-0.01", cfg => cfg.MutationRates.WeightL1Shrink = 0.01f),
            CreateConfig("L1Rate-0.05", cfg => cfg.MutationRates.WeightL1Shrink = 0.05f),
            CreateConfig("L1Rate-0.10-Default", cfg => cfg.MutationRates.WeightL1Shrink = 0.10f),
            CreateConfig("L1Rate-0.15", cfg => cfg.MutationRates.WeightL1Shrink = 0.15f),
            CreateConfig("L1Rate-0.20", cfg => cfg.MutationRates.WeightL1Shrink = 0.20f),
            CreateConfig("L1Rate-0.30", cfg => cfg.MutationRates.WeightL1Shrink = 0.30f),
            CreateConfig("L1Rate-0.50", cfg => cfg.MutationRates.WeightL1Shrink = 0.50f)
        }
    };

    // BATCH 6: L1 Shrink Factor
    private Batch CreateBatch6_L1ShrinkFactor() => new Batch
    {
        Name = "L1ShrinkFactor (shrinkage amount)",
        Configs = new[]
        {
            CreateConfig("L1Factor-0.50", cfg => cfg.MutationRates.L1ShrinkFactor = 0.50f),
            CreateConfig("L1Factor-0.70", cfg => cfg.MutationRates.L1ShrinkFactor = 0.70f),
            CreateConfig("L1Factor-0.75", cfg => cfg.MutationRates.L1ShrinkFactor = 0.75f),
            CreateConfig("L1Factor-0.80", cfg => cfg.MutationRates.L1ShrinkFactor = 0.80f),
            CreateConfig("L1Factor-0.85", cfg => cfg.MutationRates.L1ShrinkFactor = 0.85f),
            CreateConfig("L1Factor-0.90-Default", cfg => cfg.MutationRates.L1ShrinkFactor = 0.90f),
            CreateConfig("L1Factor-0.95", cfg => cfg.MutationRates.L1ShrinkFactor = 0.95f),
            CreateConfig("L1Factor-0.99", cfg => cfg.MutationRates.L1ShrinkFactor = 0.99f)
        }
    };

    // BATCH 7: Activation Swap
    private Batch CreateBatch7_ActivationSwapRate() => new Batch
    {
        Name = "ActivationSwap Rate",
        Configs = new[]
        {
            CreateConfig("ActSwap-0.000", cfg => cfg.MutationRates.ActivationSwap = 0.000f),
            CreateConfig("ActSwap-0.001", cfg => cfg.MutationRates.ActivationSwap = 0.001f),
            CreateConfig("ActSwap-0.005", cfg => cfg.MutationRates.ActivationSwap = 0.005f),
            CreateConfig("ActSwap-0.010-Default", cfg => cfg.MutationRates.ActivationSwap = 0.010f),
            CreateConfig("ActSwap-0.020", cfg => cfg.MutationRates.ActivationSwap = 0.020f),
            CreateConfig("ActSwap-0.050", cfg => cfg.MutationRates.ActivationSwap = 0.050f),
            CreateConfig("ActSwap-0.100", cfg => cfg.MutationRates.ActivationSwap = 0.100f),
            CreateConfig("ActSwap-0.200", cfg => cfg.MutationRates.ActivationSwap = 0.200f)
        }
    };

    // BATCH 8: Node Param Mutation
    private Batch CreateBatch8_NodeParamMutation() => new Batch
    {
        Name = "NodeParamMutate × NodeParamStdDev",
        Configs = new[]
        {
            CreateConfig("NodeParam-0.0×-", cfg => cfg.MutationRates.NodeParamMutate = 0.0f),
            CreateConfig("NodeParam-0.1×0.05", cfg => { cfg.MutationRates.NodeParamMutate = 0.1f; cfg.MutationRates.NodeParamStdDev = 0.05f; }),
            CreateConfig("NodeParam-0.2×0.10-Default", cfg => { cfg.MutationRates.NodeParamMutate = 0.2f; cfg.MutationRates.NodeParamStdDev = 0.10f; }),
            CreateConfig("NodeParam-0.3×0.15", cfg => { cfg.MutationRates.NodeParamMutate = 0.3f; cfg.MutationRates.NodeParamStdDev = 0.15f; }),
            CreateConfig("NodeParam-0.5×0.20", cfg => { cfg.MutationRates.NodeParamMutate = 0.5f; cfg.MutationRates.NodeParamStdDev = 0.20f; }),
            CreateConfig("NodeParam-0.2×0.05", cfg => { cfg.MutationRates.NodeParamMutate = 0.2f; cfg.MutationRates.NodeParamStdDev = 0.05f; }),
            CreateConfig("NodeParam-0.2×0.20", cfg => { cfg.MutationRates.NodeParamMutate = 0.2f; cfg.MutationRates.NodeParamStdDev = 0.20f; }),
            CreateConfig("NodeParam-0.1×0.10", cfg => { cfg.MutationRates.NodeParamMutate = 0.1f; cfg.MutationRates.NodeParamStdDev = 0.10f; })
        }
    };

    // BATCH 9: Seeds Per Individual
    private Batch CreateBatch9_SeedsPerIndividual() => new Batch
    {
        Name = "SeedsPerIndividual (evaluation seeds)",
        Configs = new[]
        {
            CreateConfig("Seeds-1", cfg => cfg.SeedsPerIndividual = 1),
            CreateConfig("Seeds-2", cfg => cfg.SeedsPerIndividual = 2),
            CreateConfig("Seeds-3", cfg => cfg.SeedsPerIndividual = 3),
            CreateConfig("Seeds-5-Default", cfg => cfg.SeedsPerIndividual = 5),
            CreateConfig("Seeds-7", cfg => cfg.SeedsPerIndividual = 7),
            CreateConfig("Seeds-10", cfg => cfg.SeedsPerIndividual = 10),
            CreateConfig("Seeds-15", cfg => cfg.SeedsPerIndividual = 15),
            CreateConfig("Seeds-20", cfg => cfg.SeedsPerIndividual = 20)
        }
    };

    // BATCH 10: Fitness Aggregation
    private Batch CreateBatch10_FitnessAggregation() => new Batch
    {
        Name = "FitnessAggregation Method",
        Configs = new[]
        {
            CreateConfig("FitAgg-Mean", cfg => cfg.FitnessAggregation = "Mean"),
            CreateConfig("FitAgg-CVaR50-Default", cfg => cfg.FitnessAggregation = "CVaR50"),
            CreateConfig("FitAgg-CVaR25", cfg => cfg.FitnessAggregation = "CVaR25"),
            CreateConfig("FitAgg-CVaR75", cfg => cfg.FitnessAggregation = "CVaR75"),
            CreateConfig("FitAgg-Min", cfg => cfg.FitnessAggregation = "Min"),
            CreateConfig("FitAgg-Max", cfg => cfg.FitnessAggregation = "Max"),
            CreateConfig("FitAgg-TrimmedMean10", cfg => cfg.FitnessAggregation = "TrimmedMean10"),
            CreateConfig("FitAgg-Median", cfg => cfg.FitnessAggregation = "Median")
        }
    };

    // BATCH 11: Weight Initialization
    private Batch CreateBatch11_WeightInitialization() => new Batch
    {
        Name = "WeightInitialization Method",
        Configs = new[]
        {
            CreateConfig("WeightInit-GlorotUniform-Default", cfg => cfg.WeightInitialization = "GlorotUniform"),
            CreateConfig("WeightInit-GlorotNormal", cfg => cfg.WeightInitialization = "GlorotNormal"),
            CreateConfig("WeightInit-HeUniform", cfg => cfg.WeightInitialization = "HeUniform"),
            CreateConfig("WeightInit-HeNormal", cfg => cfg.WeightInitialization = "HeNormal"),
            CreateConfig("WeightInit-XavierUniform", cfg => cfg.WeightInitialization = "XavierUniform"),
            CreateConfig("WeightInit-XavierNormal", cfg => cfg.WeightInitialization = "XavierNormal"),
            CreateConfig("WeightInit-Uniform0.5", cfg => cfg.WeightInitialization = "Uniform0.5"),
            CreateConfig("WeightInit-Uniform1.0", cfg => cfg.WeightInitialization = "Uniform1.0")
        }
    };

    // BATCH 12: Combined Culling
    private Batch CreateBatch12_CombinedCulling() => new Batch
    {
        Name = "Combined Culling Strategies",
        Configs = new[]
        {
            CreateConfig("Culling-Default", cfg => { /* 15/3/0.15/0.5 */ }),
            CreateConfig("Culling-FastCullHighProtect", cfg => { cfg.StagnationThreshold = 10; cfg.GraceGenerations = 5; cfg.SpeciesDiversityThreshold = 0.10f; cfg.RelativePerformanceThreshold = 0.5f; }),
            CreateConfig("Culling-SlowCullLenient", cfg => { cfg.StagnationThreshold = 30; cfg.GraceGenerations = 3; cfg.SpeciesDiversityThreshold = 0.25f; cfg.RelativePerformanceThreshold = 0.75f; }),
            CreateConfig("Culling-BalancedHighProtect", cfg => { cfg.StagnationThreshold = 20; cfg.GraceGenerations = 10; cfg.SpeciesDiversityThreshold = 0.15f; cfg.RelativePerformanceThreshold = 0.5f; }),
            CreateConfig("Culling-StrictAll", cfg => { cfg.StagnationThreshold = 15; cfg.GraceGenerations = 3; cfg.SpeciesDiversityThreshold = 0.10f; cfg.RelativePerformanceThreshold = 0.25f; }),
            CreateConfig("Culling-VeryConservative", cfg => { cfg.StagnationThreshold = 50; cfg.GraceGenerations = 3; cfg.SpeciesDiversityThreshold = 0.50f; cfg.RelativePerformanceThreshold = 0.9f; }),
            CreateConfig("Culling-VeryAggressive", cfg => { cfg.StagnationThreshold = 5; cfg.GraceGenerations = 1; cfg.SpeciesDiversityThreshold = 0.05f; cfg.RelativePerformanceThreshold = 0.25f; }),
            CreateConfig("Culling-Disabled", cfg => { cfg.StagnationThreshold = int.MaxValue; cfg.GraceGenerations = 0; cfg.SpeciesDiversityThreshold = float.MaxValue; cfg.RelativePerformanceThreshold = 1.0f; })
        }
    };

    // BATCH 13: Mutation Combinations
    private Batch CreateBatch13_MutationCombinations() => new Batch
    {
        Name = "Mutation Rate Combinations",
        Configs = new[]
        {
            CreateConfig("Mut-Default", cfg => { /* 0.1/0.01/0.2 */ }),
            CreateConfig("Mut-AllDisabled", cfg => { cfg.MutationRates.WeightL1Shrink = 0.0f; cfg.MutationRates.ActivationSwap = 0.0f; cfg.MutationRates.NodeParamMutate = 0.0f; }),
            CreateConfig("Mut-L1Only", cfg => { cfg.MutationRates.WeightL1Shrink = 0.2f; cfg.MutationRates.ActivationSwap = 0.0f; cfg.MutationRates.NodeParamMutate = 0.0f; }),
            CreateConfig("Mut-ActSwapOnly", cfg => { cfg.MutationRates.WeightL1Shrink = 0.0f; cfg.MutationRates.ActivationSwap = 0.05f; cfg.MutationRates.NodeParamMutate = 0.0f; }),
            CreateConfig("Mut-NodeParamOnly", cfg => { cfg.MutationRates.WeightL1Shrink = 0.0f; cfg.MutationRates.ActivationSwap = 0.0f; cfg.MutationRates.NodeParamMutate = 0.5f; }),
            CreateConfig("Mut-L1DefaultOthersOff", cfg => { cfg.MutationRates.WeightL1Shrink = 0.1f; cfg.MutationRates.ActivationSwap = 0.0f; cfg.MutationRates.NodeParamMutate = 0.0f; }),
            CreateConfig("Mut-AllReduced", cfg => { cfg.MutationRates.WeightL1Shrink = 0.05f; cfg.MutationRates.ActivationSwap = 0.005f; cfg.MutationRates.NodeParamMutate = 0.1f; }),
            CreateConfig("Mut-AllIncreased", cfg => { cfg.MutationRates.WeightL1Shrink = 0.2f; cfg.MutationRates.ActivationSwap = 0.02f; cfg.MutationRates.NodeParamMutate = 0.3f; })
        }
    };

    // BATCH 14: Deep Architecture Variations
    private Batch CreateBatch14_DeepArchitectureVariations() => new Batch
    {
        Name = "Deep Architecture Variations",
        Configs = new[]
        {
            CreateArchitectureConfig("Arch-3x6-Current", new[] { 3, 3, 3, 3, 3, 3 }),
            CreateArchitectureConfig("Arch-4x6", new[] { 4, 4, 4, 4, 4, 4 }),
            CreateArchitectureConfig("Arch-2x6", new[] { 2, 2, 2, 2, 2, 2 }),
            CreateArchitectureConfig("Arch-3x5", new[] { 3, 3, 3, 3, 3 }),
            CreateArchitectureConfig("Arch-3x7", new[] { 3, 3, 3, 3, 3, 3, 3 }),
            CreateArchitectureConfig("Arch-5x6", new[] { 5, 5, 5, 5, 5, 5 }),
            CreateArchitectureConfig("Arch-Bottle", new[] { 3, 4, 4, 4, 3 }),
            CreateArchitectureConfig("Arch-Hourglass", new[] { 4, 3, 3, 3, 4 })
        }
    };

    // BATCH 15: Extreme Deep Networks
    private Batch CreateBatch15_ExtremeDeepNetworks() => new Batch
    {
        Name = "Extreme Deep Networks",
        Configs = new[]
        {
            CreateArchitectureConfig("Deep-3x6-Current", new[] { 3, 3, 3, 3, 3, 3 }),
            CreateArchitectureConfig("Deep-2x8", new[] { 2, 2, 2, 2, 2, 2, 2, 2 }),
            CreateArchitectureConfig("Deep-3x8", new[] { 3, 3, 3, 3, 3, 3, 3, 3 }),
            CreateArchitectureConfig("Deep-2x10", new[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }),
            CreateArchitectureConfig("Deep-3x9", new[] { 3, 3, 3, 3, 3, 3, 3, 3, 3 }),
            CreateArchitectureConfig("Deep-2x12", new[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }),
            CreateArchitectureConfig("Deep-4x8", new[] { 4, 4, 4, 4, 4, 4, 4, 4 }),
            CreateArchitectureConfig("Deep-2x15", Enumerable.Repeat(2, 15).ToArray())
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
            Topology = CreateBestTopology(42)
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

    private EvolutionConfig CreateBestConfig()
    {
        // Best from Phase 6 with updated defaults
        return new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 200,
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
                ActivationSwap = 0.01f,
                NodeParamMutate = 0.2f,
                NodeParamStdDev = 0.1f
            },
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0.05f,
                EdgeDeleteRandom = 0.02f
            },
            WeightInitialization = "GlorotUniform",
            SeedsPerIndividual = 5,
            FitnessAggregation = "CVaR50"
        };
    }

    private SpeciesSpec CreateBestTopology(int seed)
    {
        return CreateTopology(seed, new[] { 3, 3, 3, 3, 3, 3 }); // 6 layers × 3 nodes
    }

    private SpeciesSpec CreateTopology(int seed, int[] hiddenLayers)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2);

        foreach (var hiddenSize in hiddenLayers)
        {
            builder.AddHiddenRow(hiddenSize, ActivationType.Tanh);
        }

        builder.AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random, density: 1.0f);

        return builder.Build();
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
