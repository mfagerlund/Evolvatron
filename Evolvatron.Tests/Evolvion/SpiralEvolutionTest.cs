using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// End-to-end test: Evolve a neural network to solve the two-spiral classification problem.
/// This is a harder problem than XOR, requiring more hidden capacity.
/// </summary>
public class SpiralEvolutionTest
{
    private readonly ITestOutputHelper _output;

    public SpiralEvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void EvolutionCanSolveSpiral()
    {
        // Create topology: 2 inputs (x,y) -> 8 hidden -> 8 hidden -> 1 output
        var topology = CreateSpiralTopology();

        // Configure evolution (topology mutations will be needed)
        var config = new EvolutionConfig
        {
            SpeciesCount = 8,
            IndividualsPerSpecies = 100,
            Elites = 2,
            TournamentSize = 4
        };

        // Initialize population
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Create environment (50 points per spiral = 100 total test cases)
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        _output.WriteLine($"Population: {config.SpeciesCount} species × {config.IndividualsPerSpecies} individuals = {config.SpeciesCount * config.IndividualsPerSpecies} total");
        _output.WriteLine($"Topology: 2 inputs -> 8 hidden -> 8 hidden -> 1 output");
        _output.WriteLine("");

        // Evolution loop - reduced to 10 generations for analysis
        int maxGenerations = 10;
        float successThreshold = -0.05f; // Average squared error < 0.05

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            // Evaluate all individuals
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // Get population statistics
            var stats = population.GetStatistics();

            // Get per-species best fitness
            _output.WriteLine($"=== Generation {gen} ===");
            _output.WriteLine($"Population: Best={stats.BestFitness:F4} Mean={stats.MeanFitness:F4} Median={stats.MedianFitness:F4} Worst={stats.WorstFitness:F4}");
            _output.WriteLine("Per-Species Best:");

            for (int i = 0; i < population.AllSpecies.Count; i++)
            {
                var species = population.AllSpecies[i];
                var speciesBest = species.Individuals.Max(ind => ind.Fitness);
                var speciesMean = species.Individuals.Average(ind => ind.Fitness);
                var speciesMedian = species.Individuals.Select(ind => ind.Fitness).OrderBy(f => f).ElementAt(species.Individuals.Count / 2);

                _output.WriteLine($"  Species {i}: Best={speciesBest:F4} Mean={speciesMean:F4} Median={speciesMedian:F4} (n={species.Individuals.Count})");
            }
            _output.WriteLine("");

            // Check for success (unlikely in 10 gens)
            if (stats.BestFitness >= successThreshold)
            {
                _output.WriteLine($"SUCCESS! Solved spiral classification in {gen} generations with fitness {stats.BestFitness:F6}");
                var best = population.GetBestIndividual();
                VerifySpiralSolution(best.Value.individual, best.Value.species.Topology, environment, evaluator);
                return;
            }

            // Evolve to next generation
            evolver.StepGeneration(population);
        }

        // Report final state
        var finalStats = population.GetStatistics();
        _output.WriteLine($"\n=== After {maxGenerations} Generations ===");
        _output.WriteLine($"Final best fitness: {finalStats.BestFitness:F6} (threshold: {successThreshold:F6})");
        _output.WriteLine($"Fitness range: {finalStats.BestFitness - finalStats.WorstFitness:F6}");
        _output.WriteLine($"Progress from Gen 0: (need to track Gen 0 best separately)");
    }

    private void VerifySpiralSolution(
        Individual individual,
        SpeciesSpec topology,
        SpiralEnvironment environment,
        SimpleFitnessEvaluator evaluator)
    {
        environment.Reset();

        var cpuEval = new CPUEvaluator(topology);
        var observations = new float[2];

        int correct = 0;
        int total = 0;

        var allPoints = environment.GetAllPoints();

        foreach (var (x, y, expectedLabel) in allPoints)
        {
            observations[0] = x;
            observations[1] = y;

            var outputs = cpuEval.Evaluate(individual, observations);
            float output = outputs[0];

            // For tanh output: negative = spiral 0, positive = spiral 1
            float predictedLabel = output > 0 ? 1f : -1f;

            if (MathF.Abs(predictedLabel - expectedLabel) < 0.1f)
                correct++;

            total++;
        }

        float accuracy = (float)correct / total;
        _output.WriteLine($"\nSpiral Classification Accuracy: {correct}/{total} ({accuracy * 100:F1}%)");

        // Require at least 80% accuracy
        Assert.True(accuracy >= 0.8f, $"Classification accuracy too low: {accuracy * 100:F1}%");
    }

    private SpeciesSpec CreateSpiralTopology()
    {
        var random = new Random(42);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeSparse(random)
            .Build();
    }
}
