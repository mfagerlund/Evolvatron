using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Comprehensive hypothesis sweep testing multiple theories in parallel.
/// Based on audit findings:
/// - Biases are NOT mutated (stuck at 0.0 forever)
/// - Edge topology mutations disabled (fixed topology)
/// - SeedsPerIndividual unused (already optimal)
///
/// Tests 15 configurations across 4 hypothesis categories:
/// - Batch A: Depth Experiments (5 configs)
/// - Batch B: Activation Restrictions (4 configs)
/// - Batch C: MaxInDegree Verification (3 configs)
/// - Batch D: Mutation Exploration (3 configs)
/// </summary>
public class HypothesisSweepTest
{
    private readonly ITestOutputHelper _output;

    public HypothesisSweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestAllHypotheses()
    {
        // Define all 15 test configurations
        var configurations = new[]
        {
            // === BATCH A: Depth Experiments (5 configs) ===
            new TestConfig("Dense-3Layer",
                HypothesisCategory.Depth,
                HiddenSizes: new[] { 8, 8, 8 },
                Activations: null, // Use all available
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("Dense-4Layer",
                HypothesisCategory.Depth,
                HiddenSizes: new[] { 6, 6, 6, 6 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("Funnel",
                HypothesisCategory.Depth,
                HiddenSizes: new[] { 12, 8, 4 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("Bottleneck",
                HypothesisCategory.Depth,
                HiddenSizes: new[] { 8, 2, 8 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("Baseline-2Layer",
                HypothesisCategory.Depth,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            // === BATCH B: Activation Restrictions (4 configs) ===
            new TestConfig("ReLU-Only",
                HypothesisCategory.Activation,
                HiddenSizes: new[] { 6, 6 },
                Activations: new[] { new[] { ActivationType.ReLU }, new[] { ActivationType.ReLU } },
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("Tanh-Only",
                HypothesisCategory.Activation,
                HiddenSizes: new[] { 6, 6 },
                Activations: new[] { new[] { ActivationType.Tanh }, new[] { ActivationType.Tanh } },
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("ReLU-Tanh",
                HypothesisCategory.Activation,
                HiddenSizes: new[] { 6, 6 },
                Activations: new[] { new[] { ActivationType.ReLU }, new[] { ActivationType.Tanh } },
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("No-Sigmoid",
                HypothesisCategory.Activation,
                HiddenSizes: new[] { 6, 6 },
                Activations: new[] {
                    new[] { ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU },
                    new[] { ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU }
                },
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            // === BATCH C: MaxInDegree Verification (3 configs) ===
            new TestConfig("MaxIn-6",
                HypothesisCategory.MaxInDegree,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 6,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("MaxIn-8",
                HypothesisCategory.MaxInDegree,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 8,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("MaxIn-12",
                HypothesisCategory.MaxInDegree,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 12,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.20f),

            // === BATCH D: Mutation Exploration (3 configs) ===
            new TestConfig("HighActivSwap",
                HypothesisCategory.Mutation,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.10f,
                NodeParamMutateRate: 0.20f),

            new TestConfig("HighNodeParam",
                HypothesisCategory.Mutation,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.01f,
                NodeParamMutateRate: 0.50f),

            new TestConfig("HighBothMutations",
                HypothesisCategory.Mutation,
                HiddenSizes: new[] { 6, 6 },
                Activations: null,
                MaxInDegree: 100,
                Density: 1.0f,
                ActivationSwapRate: 0.10f,
                NodeParamMutateRate: 0.50f),
        };

        _output.WriteLine($"Running {configurations.Length} hypothesis tests in parallel...\n");

        // Run all configurations in parallel
        var results = configurations
            .AsParallel()
            .WithDegreeOfParallelism(15)
            .Select(config => RunConfiguration(config))
            .ToList();

        // Output individual results
        foreach (var result in results.OrderBy(r => Array.IndexOf(configurations, configurations.First(c => c.Name == r.Name))))
        {
            var config = configurations.First(c => c.Name == result.Name);
            _output.WriteLine($"=== Hypothesis: {result.Name} ===");
            _output.WriteLine($"Category: {config.Category}");
            _output.WriteLine($"Topology: {GetTopologyDescription(config.HiddenSizes, result.TotalNodes, result.EdgeCount, result.ActiveNodeCount)}");
            _output.WriteLine($"Gen 0: {result.Gen0Best:F4} → Gen 99: {result.Gen99Best:F4} (improvement: {result.Improvement:F4})");
            _output.WriteLine("");
        }

        // Summary section
        _output.WriteLine($"\n{new string('=', 80)}");
        _output.WriteLine("SUMMARY");
        _output.WriteLine($"{new string('=', 80)}\n");

        // Top 5 by improvement
        var ranked = results.OrderByDescending(r => r.Improvement).ToList();
        _output.WriteLine("Top 5 by improvement:");
        for (int i = 0; i < Math.Min(5, ranked.Count); i++)
        {
            _output.WriteLine($"{i + 1}. {ranked[i].Name}: {ranked[i].Improvement:F4}");
        }

        // Hypothesis analysis
        _output.WriteLine("\nHypothesis results:");
        _output.WriteLine("- Depth: " + AnalyzeCategory(configurations, results, HypothesisCategory.Depth));
        _output.WriteLine("- Activations: " + AnalyzeCategory(configurations, results, HypothesisCategory.Activation));
        _output.WriteLine("- MaxInDegree: " + AnalyzeCategory(configurations, results, HypothesisCategory.MaxInDegree));
        _output.WriteLine("- Mutations: " + AnalyzeCategory(configurations, results, HypothesisCategory.Mutation));
    }

    private SweepResult RunConfiguration(TestConfig config)
    {
        // Create topology
        var topology = CreateTopology(config.HiddenSizes, config.Activations, config.MaxInDegree, config.Density, seed: 42);

        // Best hyperparameters from previous sweeps
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 8,
            IndividualsPerSpecies = 100,
            Elites = 2,
            TournamentSize = 16,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightReset = 0.10f,
                NodeParamMutate = config.NodeParamMutateRate,
                ActivationSwap = config.ActivationSwapRate
            },
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0.0f,      // Disabled (fixed topology)
                EdgeDeleteRandom = 0.0f
            }
        };

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(evolutionConfig, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Generation 0
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        // Run 99 more generations (total 100)
        for (int gen = 1; gen < 100; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen99Stats = population.GetStatistics();

        // Compute active nodes
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(topology);
        int activeNodeCount = activeNodes.Count(x => x);

        return new SweepResult
        {
            Name = config.Name,
            Gen0Best = gen0Stats.BestFitness,
            Gen99Best = gen99Stats.BestFitness,
            Improvement = gen99Stats.BestFitness - gen0Stats.BestFitness,
            EdgeCount = topology.Edges.Count,
            ActiveNodeCount = activeNodeCount,
            TotalNodes = topology.TotalNodes
        };
    }

    private SpeciesSpec CreateTopology(int[] hiddenSizes, ActivationType[][]? activations, int maxInDegree, float density, int seed)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2);

        // Add hidden layers with specified activations
        for (int i = 0; i < hiddenSizes.Length; i++)
        {
            var layerActivations = activations?[i] ?? new[]
            {
                ActivationType.ReLU,
                ActivationType.Tanh,
                ActivationType.Sigmoid,
                ActivationType.LeakyReLU,
                ActivationType.ELU,
                ActivationType.Softsign
            };

            builder.AddHiddenRow(hiddenSizes[i], layerActivations);
        }

        // Add output
        builder.AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(maxInDegree);

        // Initialize with dense connectivity
        builder.InitializeDense(random, density);

        return builder.Build();
    }

    private string GetTopologyDescription(int[] hiddenSizes, int totalNodes, int edgeCount, int activeNodes)
    {
        var topology = "2→" + string.Join("→", hiddenSizes) + "→1";
        var density = activeNodes > 0 ? (edgeCount * 100 / (activeNodes * activeNodes)) : 0;
        return $"{topology} ({totalNodes} nodes, {edgeCount} edges, {activeNodes} active)";
    }

    private string AnalyzeCategory(TestConfig[] configs, List<SweepResult> results, HypothesisCategory category)
    {
        var categoryConfigs = configs.Where(c => c.Category == category).ToList();
        if (categoryConfigs.Count == 0) return "No tests";

        var categoryResults = results.Where(r => categoryConfigs.Any(c => c.Name == r.Name))
            .OrderByDescending(r => r.Improvement)
            .ToList();

        var best = categoryResults.First();
        var avg = categoryResults.Average(r => r.Improvement);

        return $"{best.Name} best ({best.Improvement:F4}), avg: {avg:F4}";
    }

    private enum HypothesisCategory
    {
        Depth,
        Activation,
        MaxInDegree,
        Mutation
    }

    private record TestConfig(
        string Name,
        HypothesisCategory Category,
        int[] HiddenSizes,
        ActivationType[][]? Activations,
        int MaxInDegree,
        float Density,
        float ActivationSwapRate,
        float NodeParamMutateRate);

    private class SweepResult
    {
        public string Name { get; set; } = "";
        public float Gen0Best { get; set; }
        public float Gen99Best { get; set; }
        public float Improvement { get; set; }
        public int EdgeCount { get; set; }
        public int ActiveNodeCount { get; set; }
        public int TotalNodes { get; set; }
    }
}
