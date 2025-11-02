using System.Diagnostics;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Benchmarks;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Quick validation sweep to assess impact of bias fix on Phase 7 results.
/// Tests only the most critical parameters that showed strong effects in Phase 7.
///
/// PURPOSE: Determine if full Phase 8 sweep is necessary by checking:
/// - Did the architecture rankings change? (Deep-2x15 still best?)
/// - Did mutation rate rankings change? (NodeParam=0.0 still best?)
/// - Did density rankings change? (0.85 still best?)
///
/// RUNTIME: ~5-10 minutes (24 configs × 150 generations, 8 threads)
/// </summary>
public class QuickValidationSweep
{
    private readonly ITestOutputHelper _output;

    public QuickValidationSweep(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void QuickValidation_PostBiasFix()
    {
        var configs = new[]
        {
            // BATCH 1: Architecture validation (most critical finding from Phase 7)
            ("Arch-Current", CreateConfig("Current 6×3", 6, 3, 1.0f)),
            ("Arch-5x6", CreateConfig("Wide 5×6", 5, 6, 1.0f)),
            ("Arch-Deep2x15", CreateConfig("Deep 15×2", 15, 2, 1.0f)),
            ("Arch-Deep3x9", CreateConfig("Deep 9×3", 9, 3, 1.0f)),

            // BATCH 2: Mutation validation (second most critical)
            ("NodeParam-0.0", CreateConfigWithMutations(0.0f, 0.10f, "NodeParam OFF")),
            ("NodeParam-0.2", CreateConfigWithMutations(0.2f, 0.10f, "NodeParam ON (default)")),
            ("ActSwap-0.0", CreateConfigWithMutations(0.0f, 0.0f, "Tanh-only")),
            ("ActSwap-0.10", CreateConfigWithMutations(0.0f, 0.10f, "Mixed activations")),

            // BATCH 3: Density validation (critical bug fix)
            ("Density-0.5", CreateConfig("Sparse 0.5", 15, 2, 0.5f)),
            ("Density-0.7", CreateConfig("Sparse 0.7", 15, 2, 0.7f)),
            ("Density-0.85", CreateConfig("Sparse 0.85", 15, 2, 0.85f)),
            ("Density-1.0", CreateConfig("Dense 1.0", 15, 2, 1.0f))
        };

        _output.WriteLine("QUICK VALIDATION SWEEP (POST-BIAS-FIX)");
        _output.WriteLine("======================================");
        _output.WriteLine($"Total configs: {configs.Length}");
        _output.WriteLine($"Generations: 150 per config");
        _output.WriteLine($"Parallelism: 8 threads");
        _output.WriteLine("");

        var stopwatch = Stopwatch.StartNew();
        var results = new List<(string Name, string Desc, float Gen0, float Gen150, float Improvement)>();

        // Run all configs in parallel
        var tasks = configs.Select(c => Task.Run(() =>
        {
            var (name, (desc, config, topology, random)) = c;
            return (name, desc, RunEvolution(config, topology, random));
        })).ToArray();

        Task.WaitAll(tasks);

        foreach (var task in tasks)
        {
            var (name, desc, (gen0, gen150, improvement)) = task.Result;
            results.Add((name, desc, gen0, gen150, improvement));
        }

        stopwatch.Stop();

        // Sort results by improvement
        results = results.OrderByDescending(r => r.Improvement).ToList();

        _output.WriteLine("RESULTS (sorted by improvement)");
        _output.WriteLine("================================");
        _output.WriteLine("");

        // Find baseline for each batch
        var archBaseline = results.First(r => r.Name == "Arch-Current").Improvement;
        var nodeParamBaseline = results.First(r => r.Name == "NodeParam-0.2").Improvement;
        var densityBaseline = results.First(r => r.Name == "Density-1.0").Improvement;

        _output.WriteLine("BATCH 1: ARCHITECTURE VALIDATION");
        foreach (var r in results.Where(r => r.Name.StartsWith("Arch-")))
        {
            float pctVsBaseline = (r.Improvement / archBaseline - 1f) * 100f;
            string status = r.Name == "Arch-Deep2x15" ? "🏆" : "";
            _output.WriteLine($"{r.Name,-20} {r.Desc,-20} | {r.Gen0:F4} → {r.Gen150:F4} = {r.Improvement:F4} ({pctVsBaseline:+0.0;-0.0}%) {status}");
        }
        _output.WriteLine("");

        _output.WriteLine("BATCH 2: MUTATION VALIDATION");
        foreach (var r in results.Where(r => r.Name.StartsWith("NodeParam-") || r.Name.StartsWith("ActSwap-")))
        {
            float pctVsBaseline = (r.Improvement / nodeParamBaseline - 1f) * 100f;
            _output.WriteLine($"{r.Name,-20} {r.Desc,-20} | {r.Gen0:F4} → {r.Gen150:F4} = {r.Improvement:F4} ({pctVsBaseline:+0.0;-0.0}%)");
        }
        _output.WriteLine("");

        _output.WriteLine("BATCH 3: DENSITY VALIDATION");
        foreach (var r in results.Where(r => r.Name.StartsWith("Density-")))
        {
            float pctVsBaseline = (r.Improvement / densityBaseline - 1f) * 100f;
            string status = r.Name == "Density-0.85" ? "🏆" : "";
            _output.WriteLine($"{r.Name,-20} {r.Desc,-20} | {r.Gen0:F4} → {r.Gen150:F4} = {r.Improvement:F4} ({pctVsBaseline:+0.0;-0.0}%) {status}");
        }
        _output.WriteLine("");

        _output.WriteLine($"Total runtime: {stopwatch.Elapsed.TotalMinutes:F1} minutes");
        _output.WriteLine("");

        // Compare to Phase 7 findings
        _output.WriteLine("COMPARISON TO PHASE 7");
        _output.WriteLine("=====================");
        _output.WriteLine("");
        _output.WriteLine("Phase 7 claimed (with buggy code):");
        _output.WriteLine("  - Deep-2x15: +63.4% vs 6×3");
        _output.WriteLine("  - NodeParam-0.0: +38.9% vs 0.2");
        _output.WriteLine("  - Density-0.85: +37% vs 1.0 (post-density-fix)");
        _output.WriteLine("");

        var deep2x15Improvement = (results.First(r => r.Name == "Arch-Deep2x15").Improvement / archBaseline - 1f) * 100f;
        var nodeParam0Improvement = (results.First(r => r.Name == "NodeParam-0.0").Improvement / nodeParamBaseline - 1f) * 100f;
        var density085Improvement = (results.First(r => r.Name == "Density-0.85").Improvement / densityBaseline - 1f) * 100f;

        _output.WriteLine("Post-bias-fix validation:");
        _output.WriteLine($"  - Deep-2x15: {deep2x15Improvement:+0.0;-0.0}% vs 6×3");
        _output.WriteLine($"  - NodeParam-0.0: {nodeParam0Improvement:+0.0;-0.0}% vs 0.2");
        _output.WriteLine($"  - Density-0.85: {density085Improvement:+0.0;-0.0}% vs 1.0");
        _output.WriteLine("");

        // Determine if Phase 8 needed
        bool architectureChanged = Math.Abs(deep2x15Improvement - 63.4f) > 10f;
        bool mutationChanged = Math.Abs(nodeParam0Improvement - 38.9f) > 10f;
        bool densityChanged = Math.Abs(density085Improvement - 37f) > 10f;

        if (architectureChanged || mutationChanged || densityChanged)
        {
            _output.WriteLine("⚠️ RECOMMENDATION: Full Phase 8 sweep REQUIRED");
            _output.WriteLine("   Results changed significantly (>10% difference)");
        }
        else
        {
            _output.WriteLine("✅ RECOMMENDATION: Phase 7 findings mostly validated");
            _output.WriteLine("   Results changed minimally (<10% difference)");
            _output.WriteLine("   Phase 8 optional - existing recommendations likely still valid");
        }
    }

    private (float Gen0, float Gen150, float Improvement) RunEvolution(
        EvolutionConfig config,
        SpeciesSpec topology,
        Random random)
    {
        var evolver = new Evolver(random.Next());
        var population = evolver.InitializePopulation(config, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Gen 0
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();
        var gen0Fitness = gen0Stats.BestFitness;

        // Evolve to Gen 150
        for (int gen = 1; gen <= 150; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen150Stats = population.GetStatistics();
        var gen150Fitness = gen150Stats.BestFitness;

        return (gen0Fitness, gen150Fitness, gen150Fitness - gen0Fitness);
    }

    private (string Desc, EvolutionConfig Config, SpeciesSpec Topology, Random Random) CreateConfig(
        string desc, int layers, int nodesPerLayer, float density)
    {
        var random = new Random(42);
        var topology = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(nodesPerLayer, ActivationType.Tanh, count: layers)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(6)
            .InitializeDense(random, density: density)
            .Build();

        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 200,
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightJitterStdDev = 0.3f,
                WeightReset = 0.10f,
                WeightL1Shrink = 0.20f,
                L1ShrinkFactor = 0.9f,
                ActivationSwap = 0.10f,
                NodeParamMutate = 0.0f
            }
        };

        return (desc, config, topology, random);
    }

    private (string Desc, EvolutionConfig Config, SpeciesSpec Topology, Random Random) CreateConfigWithMutations(
        float nodeParamRate, float actSwapRate, string desc)
    {
        var random = new Random(42);
        var topology = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(2, ActivationType.Tanh, count: 15)
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
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightJitterStdDev = 0.3f,
                WeightReset = 0.10f,
                WeightL1Shrink = 0.20f,
                L1ShrinkFactor = 0.9f,
                ActivationSwap = actSwapRate,
                NodeParamMutate = nodeParamRate
            }
        };

        return (desc, config, topology, random);
    }
}
