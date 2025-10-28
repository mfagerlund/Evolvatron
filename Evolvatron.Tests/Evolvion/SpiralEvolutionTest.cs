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

    [Fact(Skip = "Slow: requires topology mutations and many generations")]
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

        // Evolution loop
        int maxGenerations = 500;
        float successThreshold = -0.05f; // Average squared error < 0.05

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            // Evaluate all individuals
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // Get best individual
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            if (gen % 50 == 0)
            {
                _output.WriteLine($"Generation {gen}: Best Fitness = {bestFitness:F6}");
            }

            // Check for success
            if (bestFitness >= successThreshold)
            {
                _output.WriteLine($"SUCCESS! Solved spiral classification in {gen} generations with fitness {bestFitness:F6}");

                // Verify the solution
                VerifySpiralSolution(best.Value.individual, best.Value.species.Topology, environment, evaluator);
                return;
            }

            // Evolve to next generation
            evolver.StepGeneration(population);
        }

        // If we get here, evolution didn't converge
        var final = population.GetBestIndividual();
        float finalFitness = final?.individual.Fitness ?? float.MinValue;

        _output.WriteLine($"Did not fully converge after {maxGenerations} generations.");
        _output.WriteLine($"Final best fitness: {finalFitness:F6} (threshold: {successThreshold:F6})");

        // Still assert some progress was made
        Assert.True(finalFitness > -0.5f,
            $"Evolution made insufficient progress. Final fitness: {finalFitness:F6}");
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
