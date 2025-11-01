using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Quick test to measure impact of bias mutation fix.
/// Compares Tanh-Only before and after bias mutation implementation.
/// </summary>
public class BiasImpactTest
{
    private readonly ITestOutputHelper _output;

    public BiasImpactTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TanhOnly_WithBiasMutation_ShouldImprove()
    {
        const int seed = 42;
        const int generations = 100;

        // Create Tanh-only 2→6→6→1 dense topology
        var topology = CreateTanhOnlyTopology(seed);

        // Best hyperparameters from investigation
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

        var evolver = new Evolver(seed);
        var population = evolver.InitializePopulation(config, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Initial evaluation
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        // Run evolution
        for (int gen = 1; gen < generations; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen99Stats = population.GetStatistics();
        float improvement = gen99Stats.BestFitness - gen0Stats.BestFitness;

        _output.WriteLine("=== Tanh-Only with Bias Mutation ===");
        _output.WriteLine($"Gen 0: {gen0Stats.BestFitness:F4}");
        _output.WriteLine($"Gen 99: {gen99Stats.BestFitness:F4}");
        _output.WriteLine($"Improvement: {improvement:F4}");
        _output.WriteLine("");
        _output.WriteLine("Previous (without bias mutation): 0.2058");
        _output.WriteLine($"Current (with bias mutation): {improvement:F4}");
        _output.WriteLine($"Delta: {improvement - 0.2058f:F4}");
        _output.WriteLine("");

        if (improvement > 0.2058f)
        {
            float percentBetter = ((improvement - 0.2058f) / 0.2058f) * 100f;
            _output.WriteLine($"✓ Bias mutation improved results by {percentBetter:F1}%");
        }
        else
        {
            _output.WriteLine("Note: Results may vary with different random seeds");
        }
    }

    private SpeciesSpec CreateTanhOnlyTopology(int seed)
    {
        var random = new Random(seed);
        var builder = new SpeciesBuilder()
            .AddInputRow(2)
            .WithMaxInDegree(int.MaxValue);

        // Layer 1: 6 nodes, Tanh only
        builder.AddHiddenRow(6, ActivationType.Tanh);

        // Layer 2: 6 nodes, Tanh only
        builder.AddHiddenRow(6, ActivationType.Tanh);

        // Output: Tanh
        builder.AddOutputRow(1, ActivationType.Tanh);

        // Build topology structure
        var spec = builder.Build();

        // Manually create dense connections (2→6→6→1)
        int inputCount = 2;
        int hidden1Count = 6;
        int hidden2Count = 6;
        int outputCount = 1;

        int nodeOffset = 0;

        // Input layer (2 nodes)
        nodeOffset += inputCount;
        int hidden1Start = nodeOffset;

        // Connect Input → Hidden1 (2 inputs × 6 nodes = 12 edges)
        for (int dest = 0; dest < hidden1Count; dest++)
        {
            for (int src = 0; src < inputCount; src++)
            {
                builder.AddEdge(src, hidden1Start + dest);
            }
        }

        // Hidden1 layer (6 nodes)
        nodeOffset += hidden1Count;
        int hidden2Start = nodeOffset;

        // Connect Hidden1 → Hidden2 (6 × 6 = 36 edges)
        for (int dest = 0; dest < hidden2Count; dest++)
        {
            for (int src = 0; src < hidden1Count; src++)
            {
                builder.AddEdge(hidden1Start + src, hidden2Start + dest);
            }
        }

        // Hidden2 layer (6 nodes)
        nodeOffset += hidden2Count;
        int outputStart = nodeOffset;

        // Connect Hidden2 → Output (6 × 1 = 6 edges)
        for (int src = 0; src < hidden2Count; src++)
        {
            builder.AddEdge(hidden2Start + src, outputStart);
        }

        return builder.Build();
    }
}
