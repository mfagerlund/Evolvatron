using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Test different network architectures and initialization strategies for spiral classification.
/// Based on user request: test variations of bigger networks, denser initialization, and combinations.
/// Run for 20 generations (double the previous 10).
/// </summary>
public class SpiralNetworkArchitectureSweepTest
{
    private readonly ITestOutputHelper _output;

    public SpiralNetworkArchitectureSweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestNetworkArchitectures()
    {
        // Best hyperparameters from previous sweep
        var bestConfig = new EvolutionConfig
        {
            SpeciesCount = 8,
            IndividualsPerSpecies = 100,
            Elites = 2,
            TournamentSize = 16,  // Best from sweep
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,  // Best from sweep
                WeightReset = 0.10f
            },
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0.05f,
                EdgeDeleteRandom = 0.02f
            }
        };

        var architectures = new[]
        {
            // Baseline (from previous best)
            new Architecture("Baseline-Sparse-20gen",
                HiddenLayers: new[] { 8, 8 },
                MaxInDegree: 10,
                Sparse: true,
                Description: "2→8→8→1, sparse init, best hyperparams, 20 gens"),

            // Bigger networks (Option 1)
            new Architecture("Bigger-16x16-Sparse",
                HiddenLayers: new[] { 16, 16 },
                MaxInDegree: 12,
                Sparse: true,
                Description: "2→16→16→1, sparse init"),

            new Architecture("Bigger-16x16x8-Sparse",
                HiddenLayers: new[] { 16, 16, 8 },
                MaxInDegree: 12,
                Sparse: true,
                Description: "2→16→16→8→1, sparse init (3 layers)"),

            new Architecture("Bigger-12x12-Sparse",
                HiddenLayers: new[] { 12, 12 },
                MaxInDegree: 12,
                Sparse: true,
                Description: "2→12→12→1, sparse init (moderate size)"),

            new Architecture("Bigger-20x20-Sparse",
                HiddenLayers: new[] { 20, 20 },
                MaxInDegree: 15,
                Sparse: true,
                Description: "2→20→20→1, sparse init (large)"),

            // Denser initialization (Option 2) - using higher MaxInDegree
            new Architecture("Baseline-HighDegree",
                HiddenLayers: new[] { 8, 8 },
                MaxInDegree: 16,
                Sparse: true,
                Description: "2→8→8→1, high MaxInDegree (16)"),

            new Architecture("Baseline-VeryHighDegree",
                HiddenLayers: new[] { 8, 8 },
                MaxInDegree: 20,
                Sparse: true,
                Description: "2→8→8→1, very high MaxInDegree (20)"),

            new Architecture("Bigger-12x12-HighDegree",
                HiddenLayers: new[] { 12, 12 },
                MaxInDegree: 20,
                Sparse: true,
                Description: "2→12→12→1, high MaxInDegree (20)"),

            // Combinations (Option 3)
            new Architecture("Bigger-16x16-HighDegree",
                HiddenLayers: new[] { 16, 16 },
                MaxInDegree: 24,
                Sparse: true,
                Description: "2→16→16→1, high MaxInDegree (24)"),

            new Architecture("Bigger-16x16x8-HighDegree",
                HiddenLayers: new[] { 16, 16, 8 },
                MaxInDegree: 24,
                Sparse: true,
                Description: "2→16→16→8→1, high MaxInDegree (24)"),

            new Architecture("Bigger-20x10-HighDegree",
                HiddenLayers: new[] { 20, 10 },
                MaxInDegree: 30,
                Sparse: true,
                Description: "2→20→10→1, high MaxInDegree (30)"),

            new Architecture("Deep-10x10x10-Sparse",
                HiddenLayers: new[] { 10, 10, 10 },
                MaxInDegree: 10,
                Sparse: true,
                Description: "2→10→10→10→1, sparse (deep)"),
        };

        var results = new List<ArchitectureResult>();

        foreach (var arch in architectures)
        {
            _output.WriteLine($"\n{'='*80}");
            _output.WriteLine($"Testing: {arch.Name}");
            _output.WriteLine($"  {arch.Description}");
            _output.WriteLine($"  Hidden layers: {string.Join("→", arch.HiddenLayers)}");
            _output.WriteLine($"  Initialization: {(arch.Sparse ? "Sparse" : $"Dense({(arch.DensityOverride ?? 0.5f) * 100:F0}%)")}");
            _output.WriteLine($"{'='*80}\n");

            var result = RunArchitecture(arch, bestConfig);
            results.Add(result);

            _output.WriteLine($"RESULT: Gen0={result.Gen0Best:F4} Gen19={result.Gen19Best:F4} Improvement={result.Improvement:F4} ({result.ImprovementPercent:F1}%)");
            _output.WriteLine($"        EdgeCount: {result.EdgeCount}, ActiveNodes: {result.ActiveNodeCount}/{result.TotalNodeCount}");
            _output.WriteLine($"        FinalRange: {result.Gen19Range:F4}, MeanImprovement: {result.MeanImprovement:F4}\n");
        }

        // Summary
        _output.WriteLine($"\n\n{'='*80}");
        _output.WriteLine("ARCHITECTURE SWEEP SUMMARY - Ranked by Improvement");
        _output.WriteLine($"{'='*80}\n");

        var ranked = results.OrderByDescending(r => r.Improvement).ToList();

        _output.WriteLine($"{"Rank",-5} {"Architecture",-30} {"Gen0",-8} {"Gen19",-8} {"Δ",-9} {"Δ%",-8} {"Edges",-7} {"Nodes",-8}");
        _output.WriteLine(new string('-', 95));

        for (int i = 0; i < ranked.Count; i++)
        {
            var r = ranked[i];
            _output.WriteLine($"{i + 1,-5} {r.Name,-30} {r.Gen0Best,-8:F4} {r.Gen19Best,-8:F4} {r.Improvement,-9:F4} {r.ImprovementPercent,-8:F1} {r.EdgeCount,-7} {r.ActiveNodeCount,-8}");
        }

        _output.WriteLine($"\n{'='*80}");
        _output.WriteLine("KEY COMPARISONS");
        _output.WriteLine($"{'='*80}");

        var baseline = results.First(r => r.Name == "Baseline-Sparse-20gen");
        var best = ranked[0];

        _output.WriteLine($"\nBaseline (2→8→8→1, sparse, 20 gens):");
        _output.WriteLine($"  Improvement: {baseline.Improvement:F4} ({baseline.ImprovementPercent:F1}%)");
        _output.WriteLine($"  Edges: {baseline.EdgeCount}, Active nodes: {baseline.ActiveNodeCount}");

        _output.WriteLine($"\nBest Architecture ({best.Name}):");
        _output.WriteLine($"  Improvement: {best.Improvement:F4} ({best.ImprovementPercent:F1}%)");
        _output.WriteLine($"  Better than baseline: {(best.Improvement / baseline.Improvement - 1) * 100:F0}%");
        _output.WriteLine($"  Edges: {best.EdgeCount}, Active nodes: {best.ActiveNodeCount}");

        // Category analysis
        _output.WriteLine($"\n{'='*80}");
        _output.WriteLine("CATEGORY ANALYSIS");
        _output.WriteLine($"{'='*80}");

        var biggerSparse = results.Where(r => r.Name.Contains("Bigger") && r.Name.Contains("Sparse")).ToList();
        var denseInit = results.Where(r => r.Name.Contains("Dense")).ToList();
        var combinations = results.Where(r => r.Name.Contains("Bigger") && r.Name.Contains("Dense")).ToList();

        if (biggerSparse.Any())
        {
            _output.WriteLine($"\nBigger Networks (Sparse Init):");
            _output.WriteLine($"  Average improvement: {biggerSparse.Average(r => r.Improvement):F4}");
            _output.WriteLine($"  Best: {biggerSparse.OrderByDescending(r => r.Improvement).First().Name}");
        }

        if (denseInit.Any())
        {
            _output.WriteLine($"\nDenser Initialization:");
            _output.WriteLine($"  Average improvement: {denseInit.Average(r => r.Improvement):F4}");
            _output.WriteLine($"  Best: {denseInit.OrderByDescending(r => r.Improvement).First().Name}");
        }

        if (combinations.Any())
        {
            _output.WriteLine($"\nBigger + Dense:");
            _output.WriteLine($"  Average improvement: {combinations.Average(r => r.Improvement):F4}");
            _output.WriteLine($"  Best: {combinations.OrderByDescending(r => r.Improvement).First().Name}");
        }

        // Projected generations to solve
        _output.WriteLine($"\n{'='*80}");
        _output.WriteLine("PROJECTED TIME TO SOLVE (fitness -0.05)");
        _output.WriteLine($"{'='*80}");

        float targetFitness = -0.05f;
        foreach (var r in ranked.Take(5))
        {
            float fitnessGap = r.Gen19Best - targetFitness;
            float improvementPerGen = r.Improvement / 20.0f;
            int projectedGens = improvementPerGen > 0.0001f ? (int)(fitnessGap / improvementPerGen) : 999999;

            _output.WriteLine($"{r.Name,-30}: ~{projectedGens,5} generations ({projectedGens * 0.86 / 60:F1} minutes)");
        }
    }

    private ArchitectureResult RunArchitecture(Architecture arch, EvolutionConfig baseConfig)
    {
        var topology = CreateTopology(arch);

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(baseConfig, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Generation 0
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        // Run 19 more generations (total 20)
        for (int gen = 1; gen < 20; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen19Stats = population.GetStatistics();

        // Count edges and active nodes
        int edgeCount = topology.Edges.Count;
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(topology);
        int activeNodeCount = activeNodes.Count(x => x);
        int totalNodeCount = topology.TotalNodes;

        return new ArchitectureResult
        {
            Name = arch.Name,
            Gen0Best = gen0Stats.BestFitness,
            Gen0Mean = gen0Stats.MeanFitness,
            Gen19Best = gen19Stats.BestFitness,
            Gen19Mean = gen19Stats.MeanFitness,
            Gen19Range = gen19Stats.BestFitness - gen19Stats.WorstFitness,
            Improvement = gen19Stats.BestFitness - gen0Stats.BestFitness,
            MeanImprovement = gen19Stats.MeanFitness - gen0Stats.MeanFitness,
            EdgeCount = edgeCount,
            ActiveNodeCount = activeNodeCount,
            TotalNodeCount = totalNodeCount
        };
    }

    private SpeciesSpec CreateTopology(Architecture arch)
    {
        var random = new Random(42);
        var builder = new SpeciesBuilder()
            .AddInputRow(2);

        // Add hidden layers
        foreach (var layerSize in arch.HiddenLayers)
        {
            builder.AddHiddenRow(layerSize,
                ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid,
                ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign);
        }

        // Add output
        builder.AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(arch.MaxInDegree);

        // Initialize sparse (only method available currently)
        // For "dense" configs, we use higher MaxInDegree to approximate denser connectivity
        builder.InitializeSparse(random);

        return builder.Build();
    }

    private record Architecture(
        string Name,
        int[] HiddenLayers,
        int MaxInDegree,
        bool Sparse,
        string Description,
        float? DensityOverride = null)
    {
        public float? DensityOverride { get; init; } = DensityOverride;
    }

    private class ArchitectureResult
    {
        public string Name { get; set; } = "";
        public float Gen0Best { get; set; }
        public float Gen0Mean { get; set; }
        public float Gen19Best { get; set; }
        public float Gen19Mean { get; set; }
        public float Gen19Range { get; set; }
        public float Improvement { get; set; }
        public float MeanImprovement { get; set; }
        public int EdgeCount { get; set; }
        public int ActiveNodeCount { get; set; }
        public int TotalNodeCount { get; set; }
        public float ImprovementPercent => (Improvement / Math.Abs(Gen0Best)) * 100f;
    }
}
