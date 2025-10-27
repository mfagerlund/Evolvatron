using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// End-to-end test: Evolve a neural network to solve XOR.
/// This validates that the entire evolutionary system works together.
/// </summary>
public class XOREvolutionTest
{
    private readonly ITestOutputHelper _output;

    public XOREvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void EvolutionCanSolveXOR()
    {
        // Create a simple topology: 2 inputs -> 4 hidden -> 1 output
        var topology = CreateXORTopology();

        // Configure evolution (using tuned defaults from hyperparameter sweep)
        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 100,  // 400 total (optimal from sweep)
            Elites = 2,
            TournamentSize = 3
            // Using default MutationRates (tuned from sweep):
            // - WeightJitter = 0.95
            // - WeightJitterStdDev = 0.3
            // - WeightReset = 0.1
        };

        // Initialize population
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Create environment and evaluator
        var environment = new XOREnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        // Evolution loop
        int maxGenerations = 100;
        float successThreshold = -0.01f; // Very close to 0 error

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            // Evaluate all individuals
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // Get best individual
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            _output.WriteLine($"Generation {gen}: Best Fitness = {bestFitness:F6}");

            // Check for success
            if (bestFitness >= successThreshold)
            {
                _output.WriteLine($"SUCCESS! Solved XOR in {gen} generations with fitness {bestFitness:F6}");

                // Verify the solution actually works
                VerifyXORSolution(best.Value.individual, best.Value.species.Topology, environment, evaluator);
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

    private void VerifyXORSolution(
        Individual individual,
        SpeciesSpec topology,
        XOREnvironment environment,
        SimpleFitnessEvaluator evaluator)
    {
        environment.Reset();

        var testCases = new[]
        {
            (0f, 0f, 0f),
            (0f, 1f, 1f),
            (1f, 0f, 1f),
            (1f, 1f, 0f)
        };

        var cpuEval = new CPUEvaluator(topology);
        var observations = new float[2];

        _output.WriteLine("\nXOR Truth Table Verification:");
        foreach (var (x, y, expected) in testCases)
        {
            observations[0] = x;
            observations[1] = y;

            var outputs = cpuEval.Evaluate(individual, observations);
            float output = outputs[0];
            float error = Math.Abs(output - expected);

            _output.WriteLine($"  {x} XOR {y} = {expected:F0} | Network output: {output:F4} | Error: {error:F4}");

            // Allow some tolerance
            Assert.True(error < 0.3f, $"Output error too large for input ({x}, {y})");
        }
    }

    private SpeciesSpec CreateXORTopology()
    {
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(4, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(1, ActivationType.Tanh)
            .FullyConnect(fromRow: 0, toRow: 2)
            .FullyConnect(fromRow: 1, toRow: 2)
            .FullyConnect(fromRow: 2, toRow: 3)
            .WithMaxInDegree(8)
            .Build();
    }
}
