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
        // Row 0: 1 bias
        // Row 1: 9 sensors
        // Row 2: 12 hidden
        // Row 3: 2 outputs (steering, throttle)
        var topology = new SpeciesSpec
        {
            RowCounts = new[] { 1, 9, 12, 2 },
            AllowedActivationsPerRow = new uint[]
            {
                0b00000000001, // Row 0 (bias): Identity only
                0b11111111111, // Row 1 (inputs): All activations
                0b11111111111, // Row 2 (hidden): All activations
                0b00000000011  // Row 3 (outputs): Identity and Tanh
            },
            MaxInDegree = 12,
            Edges = new List<(int, int)>()
        };

        // Connect bias + inputs (row 0 + row 1) to hidden layer (row 2)
        for (int src = 0; src < 10; src++) // Nodes 0-9 (bias + 9 sensors)
        {
            for (int dst = 10; dst < 22; dst++) // Nodes 10-21 (12 hidden)
            {
                topology.Edges.Add((src, dst));
            }
        }

        // Connect hidden layer (row 2) to outputs (row 3)
        for (int src = 10; src < 22; src++) // Nodes 10-21 (12 hidden)
        {
            topology.Edges.Add((src, 22)); // Node 22 (steering)
            topology.Edges.Add((src, 23)); // Node 23 (throttle)
        }

        topology.BuildRowPlans();
        return topology;
    }
}
