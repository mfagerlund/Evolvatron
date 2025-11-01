using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Compares different initialization strategies for network topology.
/// Tests sparse vs dense connectivity patterns and their impact on learning.
/// </summary>
public class InitializationComparisonTest
{
    private readonly ITestOutputHelper _output;

    public InitializationComparisonTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void CompareInitializationStrategies()
    {
        const int seed = 42;
        const int maxGenerations = 100;
        const float targetFitness = -0.5f;

        // Define all configurations upfront
        var configs = new[]
        {
            ("Sparse (Baseline)", CreateSparseTopology(hiddenSizes: new[] { 8, 8 }, maxInDegree: 10, seed)),
            ("Dense", CreateDenseTopology(hiddenSizes: new[] { 6, 6 }, maxInDegree: int.MaxValue, seed)),
            ("HighDegree", CreateSparseTopology(hiddenSizes: new[] { 8, 8 }, maxInDegree: 20, seed)),
            ("Dense-Small", CreateDenseTopology(hiddenSizes: new[] { 4, 4 }, maxInDegree: int.MaxValue, seed)),
            ("Dense-Bigger", CreateDenseTopology(hiddenSizes: new[] { 8, 8 }, maxInDegree: int.MaxValue, seed)),
            ("Medium-Sparse", CreateSparseTopology(hiddenSizes: new[] { 6, 6 }, maxInDegree: 10, seed)),
            ("SemiDense-75", CreateDenseTopology(hiddenSizes: new[] { 6, 6 }, maxInDegree: int.MaxValue, seed, density: 0.75f)),
            ("SemiDense-50", CreateDenseTopology(hiddenSizes: new[] { 6, 6 }, maxInDegree: int.MaxValue, seed, density: 0.50f)),
            ("SemiDense-25", CreateDenseTopology(hiddenSizes: new[] { 6, 6 }, maxInDegree: int.MaxValue, seed, density: 0.25f))
        };

        // Run all configurations in parallel
        var results = configs.AsParallel()
            .WithDegreeOfParallelism(9) // Run all 9 in parallel (12 cores available)
            .Select((config, index) => RunConfiguration(
                name: config.Item1,
                topology: config.Item2,
                seed: seed + index, // Different seed per config to avoid lock contention
                maxGenerations: maxGenerations,
                targetFitness: targetFitness))
            .ToList();

        // Print results
        foreach (var result in results)
        {
            PrintResult(result);
        }

        // Summary
        _output.WriteLine("\n" + new string('=', 80));
        _output.WriteLine("SUMMARY");
        _output.WriteLine(new string('=', 80));

        var successful = results.Where(r => r.ReachedTarget).ToList();
        if (successful.Any())
        {
            _output.WriteLine($"\nSuccessful strategies ({successful.Count}/{results.Count}):");
            foreach (var r in successful.OrderBy(r => r.GenerationReached))
            {
                _output.WriteLine($"  {r.Name}: reached {targetFitness:F4} at gen {r.GenerationReached}");
            }
        }
        else
        {
            _output.WriteLine($"\nNo strategy reached target fitness {targetFitness:F4}");
            var best = results.OrderByDescending(r => r.FinalFitness).First();
            _output.WriteLine($"Best: {best.Name} (fitness {best.FinalFitness:F4})");
        }

        _output.WriteLine($"\nImprovement ranking:");
        foreach (var r in results.OrderByDescending(r => r.Improvement))
        {
            _output.WriteLine($"  {r.Name}: {r.Improvement:F4}");
        }
    }

    private InitResult RunConfiguration(string name, SpeciesSpec topology, int seed, int maxGenerations, float targetFitness)
    {
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 8,
            IndividualsPerSpecies = 100,
            Elites = 2,
            TournamentSize = 16,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightReset = 0.10f
            }
        };

        var evolver = new Evolver(seed);
        var population = evolver.InitializePopulation(evolutionConfig, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Initial evaluation
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        int generationReached = -1;
        float finalFitness = gen0Stats.BestFitness;

        // Run evolution
        for (int gen = 1; gen < maxGenerations; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            var stats = population.GetStatistics();
            finalFitness = stats.BestFitness;

            // Check if target reached
            if (finalFitness >= targetFitness && generationReached == -1)
            {
                generationReached = gen;
                break; // Stop early if target reached
            }
        }

        // Compute active nodes
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(topology);
        int activeCount = activeNodes.Count(a => a);
        float activePercent = (activeCount / (float)topology.TotalNodes) * 100f;

        return new InitResult
        {
            Name = name,
            TopologyDescription = GetTopologyDescription(topology),
            TotalNodes = topology.TotalNodes,
            TotalEdges = topology.TotalEdges,
            ActiveNodes = activeCount,
            ActivePercent = activePercent,
            MaxInDegree = topology.MaxInDegree,
            Gen0Fitness = gen0Stats.BestFitness,
            FinalFitness = finalFitness,
            Improvement = finalFitness - gen0Stats.BestFitness,
            ReachedTarget = generationReached != -1,
            GenerationReached = generationReached != -1 ? generationReached : maxGenerations - 1
        };
    }

    private void PrintResult(InitResult result)
    {
        _output.WriteLine($"\n=== Initialization Strategy: {result.Name} ===");
        _output.WriteLine($"Topology: {result.TopologyDescription} ({result.TotalNodes} nodes, MaxInDegree={result.MaxInDegree})");
        _output.WriteLine($"Initial: {result.TotalEdges} edges, {result.ActiveNodes} active nodes ({result.ActivePercent:F1}%)");
        _output.WriteLine($"Gen 0: {result.Gen0Fitness:F4} → Gen {result.GenerationReached}: {result.FinalFitness:F4} (improvement: {result.Improvement:F4})");

        if (result.ReachedTarget)
        {
            _output.WriteLine($"Reached -0.5: Yes (gen {result.GenerationReached}) ✓");
        }
        else
        {
            _output.WriteLine($"Reached -0.5: No (best: {result.FinalFitness:F4} at gen {result.GenerationReached})");
        }
    }

    private string GetTopologyDescription(SpeciesSpec spec)
    {
        return string.Join("→", spec.RowCounts);
    }

    private SpeciesSpec CreateSparseTopology(int[] hiddenSizes, int maxInDegree, int seed)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2)
            .WithMaxInDegree(maxInDegree);

        foreach (var size in hiddenSizes)
        {
            builder.AddHiddenRow(size, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU);
        }

        builder.AddOutputRow(1, ActivationType.Tanh)
            .InitializeSparse(random);

        return builder.Build();
    }

    private SpeciesSpec CreateDenseTopology(int[] hiddenSizes, int maxInDegree, int seed, float density = 1.0f)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2)
            .WithMaxInDegree(maxInDegree);

        foreach (var size in hiddenSizes)
        {
            builder.AddHiddenRow(size, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU);
        }

        builder.AddOutputRow(1, ActivationType.Tanh);

        // Manually connect densely with optional density sampling:
        // Each node connects to a fraction (density) of nodes in previous layers
        // (respecting maxInDegree constraint and ensuring at least 1 edge per node)
        var rowCounts = new List<int> { 2 };
        rowCounts.AddRange(hiddenSizes);
        rowCounts.Add(1);

        int nodeOffset = 0;
        for (int layerIdx = 1; layerIdx < rowCounts.Count; layerIdx++)
        {
            int prevLayerStart = nodeOffset;
            int prevLayerSize = rowCounts[layerIdx - 1];
            nodeOffset += prevLayerSize;

            int currentLayerStart = nodeOffset;
            int currentLayerSize = rowCounts[layerIdx];

            // For each node in current layer, connect to a fraction of nodes in previous layers
            for (int destIdx = 0; destIdx < currentLayerSize; destIdx++)
            {
                int destNode = currentLayerStart + destIdx;
                var candidates = new List<int>();

                // Add all nodes from all previous layers as candidates
                for (int prevLayer = 0; prevLayer < layerIdx; prevLayer++)
                {
                    int layerStart = rowCounts.Take(prevLayer).Sum();
                    int layerSize = rowCounts[prevLayer];
                    for (int i = 0; i < layerSize; i++)
                    {
                        candidates.Add(layerStart + i);
                    }
                }

                // Shuffle candidates for randomness
                for (int i = candidates.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
                }

                // Calculate how many edges to add based on density
                int targetEdges = Math.Max(1, (int)Math.Round(candidates.Count * density));
                int edgesAdded = 0;

                foreach (var srcNode in candidates)
                {
                    if (edgesAdded >= targetEdges || edgesAdded >= maxInDegree)
                        break;

                    builder.AddEdge(srcNode, destNode);
                    edgesAdded++;
                }
            }
        }

        return builder.Build();
    }

    private class InitResult
    {
        public string Name { get; set; } = "";
        public string TopologyDescription { get; set; } = "";
        public int TotalNodes { get; set; }
        public int TotalEdges { get; set; }
        public int ActiveNodes { get; set; }
        public float ActivePercent { get; set; }
        public int MaxInDegree { get; set; }
        public float Gen0Fitness { get; set; }
        public float FinalFitness { get; set; }
        public float Improvement { get; set; }
        public bool ReachedTarget { get; set; }
        public int GenerationReached { get; set; }
    }
}
