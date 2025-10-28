using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;

namespace Evolvatron.Tests.Evolvion;

public class FollowTheCorridorEvolutionTest
{
    [Fact]
    public void FollowTheCorridor_Evolves_ToCompleteTrack()
    {
        // Arrange
        var topology = CreateCorridorTopology();
        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 100,
            Elites = 4,
            TournamentSize = 4
            // Mutation rates use tuned defaults from hyperparameter sweep
        };

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);
        var environment = new FollowTheCorridorEnvironment(maxSteps: 320);
        var evaluator = new SimpleFitnessEvaluator();

        // Act
        float bestFitness = float.MinValue;
        int maxGenerations = 100; // Limit to 100 generations for test
        int generation = 0;

        for (generation = 0; generation < maxGenerations; generation++)
        {
            // Evaluate population
            evaluator.EvaluatePopulation(population, environment, seed: generation);

            // Get best individual
            var best = population.GetBestIndividual();
            if (best.HasValue)
            {
                bestFitness = best.Value.individual.Fitness;
                Console.WriteLine($"Generation {generation}: Best Fitness = {bestFitness:F3}");

                // Success criteria: reach >50% of checkpoints
                if (bestFitness > 0.5f)
                {
                    Console.WriteLine($"SUCCESS! Reached {bestFitness * 100:F0}% of checkpoints in {generation} generations");
                    break;
                }
            }

            // Evolve to next generation
            evolver.StepGeneration(population);
        }

        // Assert
        Assert.True(bestFitness > 0.1f,
            $"Expected improvement after {generation} generations, but fitness was {bestFitness:F3}");

        Console.WriteLine($"Final result after {generation} generations: {bestFitness * 100:F0}% of track completed");
    }

    private static SpeciesSpec CreateCorridorTopology()
    {
        // Row 0: 9 sensors
        // Row 1: 12 hidden
        // Row 2: 2 outputs (steering, throttle)
        return new SpeciesBuilder()
            .AddInputRow(9)
            .AddHiddenRow(12, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(2, ActivationType.Tanh)
            .FullyConnect(fromRow: 0, toRow: 1)
            .FullyConnect(fromRow: 1, toRow: 2)
            .WithMaxInDegree(12)
            .Build();
    }
}
