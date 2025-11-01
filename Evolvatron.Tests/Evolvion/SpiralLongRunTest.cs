using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Long-running evolution test: Dense 2→6→6→1 topology for 500 generations.
/// Uses best hyperparameters identified in hyperparameter sweep tests.
/// Samples output every 10 generations to minimize verbosity.
/// </summary>
public class SpiralLongRunTest
{
    private readonly ITestOutputHelper _output;

    public SpiralLongRunTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Skip = "Long-running test")]
    public void DenseTwoSixSixOneEvolvesForFiveHundredGenerations()
    {
        const int seed = 42;
        const int maxGenerations = 500;
        const float targetFitness = -0.5f;
        const int sampleInterval = 10;

        // Create Dense 2→6→6→1 topology
        var topology = CreateDenseTopology(hiddenSizes: new[] { 6, 6 }, seed);

        // Best hyperparameters from sweep: Tournament=16, WeightJitter=0.95, Elites=2
        var config = new EvolutionConfig
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

        // Initialize population
        var evolver = new Evolver(seed);
        var population = evolver.InitializePopulation(config, topology);

        // Environment and evaluator
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Compute active nodes
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(topology);
        int activeCount = activeNodes.Count(a => a);
        float activePercent = (activeCount / (float)topology.TotalNodes) * 100f;

        // Print header
        _output.WriteLine("Dense 2→6→6→1 Long Run (500 generations)");
        _output.WriteLine($"Topology: {topology.TotalNodes} nodes, {topology.TotalEdges} edges, {activePercent:F0}% active");
        _output.WriteLine($"Config: Tournament={config.TournamentSize}, WeightJitter={config.MutationRates.WeightJitter}, Elites={config.Elites}");
        _output.WriteLine("");

        // Initial evaluation
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();
        _output.WriteLine($"Gen 0: {gen0Stats.BestFitness:F4}");

        float finalFitness = gen0Stats.BestFitness;
        int generationReached = -1;

        // Evolution loop
        for (int gen = 1; gen < maxGenerations; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            var stats = population.GetStatistics();
            finalFitness = stats.BestFitness;

            // Sample output every 10 generations
            if (gen % sampleInterval == 0)
            {
                _output.WriteLine($"Gen {gen}: {finalFitness:F4}");
            }

            // Check if target reached
            if (finalFitness >= targetFitness && generationReached == -1)
            {
                generationReached = gen;
                _output.WriteLine($"Gen {gen}: {finalFitness:F4} ✓ TARGET REACHED");
                break;
            }
        }

        // Print summary
        _output.WriteLine("");
        if (generationReached != -1)
        {
            float improvement = finalFitness - gen0Stats.BestFitness;
            _output.WriteLine($"Final: Reached {targetFitness:F1} at generation {generationReached}");
            _output.WriteLine($"Total improvement: {improvement:F4}");
        }
        else
        {
            float improvement = finalFitness - gen0Stats.BestFitness;
            _output.WriteLine($"Final: Did not reach {targetFitness:F1} within {maxGenerations} generations");
            _output.WriteLine($"Best fitness: {finalFitness:F4} (improvement: {improvement:F4})");
        }
    }

    private SpeciesSpec CreateDenseTopology(int[] hiddenSizes, int seed)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2)
            .WithMaxInDegree(int.MaxValue);

        foreach (var size in hiddenSizes)
        {
            builder.AddHiddenRow(size, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU);
        }

        builder.AddOutputRow(1, ActivationType.Tanh);

        // Manually connect densely: each node connects to ALL nodes in previous layers
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

            // For each node in current layer, connect to all nodes in previous layers
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

                // Add edges (no maxInDegree limit for dense topology)
                foreach (var srcNode in candidates)
                {
                    builder.AddEdge(srcNode, destNode);
                }
            }
        }

        return builder.Build();
    }
}
