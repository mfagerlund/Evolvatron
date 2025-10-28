using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// End-to-end test: Evolve a neural network to navigate a winding corridor.
/// This is a simplified version of Colonel.Tests FollowTheCorridor environment.
///
/// Difficulty: Harder than CartPole
/// - 9D state space (vs 4D)
/// - Long episode (320 steps)
/// - Sparse rewards (checkpoints)
/// - Requires coordinated steering + throttle control
/// </summary>
public class SimpleCorridorEvolutionTest
{
    private readonly ITestOutputHelper _output;

    public SimpleCorridorEvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Skip = "Slow test - run explicitly")]
    public void EvolutionCanNavigateCorridor()
    {
        // Create topology: 9 inputs -> 12 hidden -> 2 outputs
        var topology = CreateCorridorTopology();

        // Configure evolution (using tuned defaults)
        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 100,  // 400 total
            Elites = 4,
            TournamentSize = 4
            // Using default MutationRates from XOR/CartPole tuning
        };

        // Initialize population
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Create environment and evaluator
        var environment = new SimpleCorridorEnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        // Evolution loop
        int maxGenerations = 500;
        float successThreshold = 0.8f; // Collect 80%+ of checkpoints

        _output.WriteLine("=== SIMPLE CORRIDOR EVOLUTION ===");
        _output.WriteLine($"Topology: {string.Join("-", topology.RowCounts)}");
        _output.WriteLine($"Population: {config.SpeciesCount} species × {config.IndividualsPerSpecies} = {config.SpeciesCount * config.IndividualsPerSpecies} total");
        _output.WriteLine($"Success threshold: {successThreshold * 100}% of checkpoints");
        _output.WriteLine($"Track: Procedural sine wave, 40 checkpoints");
        _output.WriteLine("");

        float bestEverFitness = float.MinValue;

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            // Evaluate all individuals
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // Get best individual
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            if (bestFitness > bestEverFitness)
                bestEverFitness = bestFitness;

            if (gen % 10 == 0 || bestFitness >= successThreshold)
            {
                _output.WriteLine($"Generation {gen,3}: Best = {bestFitness,6:F3}, Best Ever = {bestEverFitness,6:F3}");
            }

            // Check for success
            if (bestFitness >= successThreshold)
            {
                _output.WriteLine("");
                _output.WriteLine($"SUCCESS! Navigated corridor in {gen} generations with fitness {bestFitness:F3}");
                _output.WriteLine("");

                // Verify solution on multiple seeds
                VerifyCorridorSolution(best.Value.individual, best.Value.species.Topology, environment, evaluator);
                return;
            }

            // Evolve to next generation
            evolver.StepGeneration(population);
        }

        // If we get here, evolution didn't fully converge
        var final = population.GetBestIndividual();
        float finalFitness = final?.individual.Fitness ?? float.MinValue;

        _output.WriteLine("");
        _output.WriteLine($"Did not fully converge after {maxGenerations} generations.");
        _output.WriteLine($"Final best fitness: {finalFitness:F3} (threshold: {successThreshold:F3})");
        _output.WriteLine($"Best ever fitness: {bestEverFitness:F3}");

        // Assert some progress was made
        Assert.True(finalFitness > 0.2f,
            $"Evolution made insufficient progress. Final fitness: {finalFitness:F3}");
    }

    private void VerifyCorridorSolution(
        Individual individual,
        SpeciesSpec topology,
        SimpleCorridorEnvironment environment,
        SimpleFitnessEvaluator evaluator)
    {
        _output.WriteLine("=== VERIFICATION ACROSS MULTIPLE SEEDS ===");

        var seeds = new[] { 100, 101, 102, 103, 104 };
        var rewards = new float[seeds.Length];

        for (int i = 0; i < seeds.Length; i++)
        {
            float reward = evaluator.Evaluate(individual, topology, environment, seed: seeds[i]);
            rewards[i] = reward;
            _output.WriteLine($"  Seed {seeds[i]}: Reward = {reward,6:F3} ({reward * 100:F0}% checkpoints)");
        }

        float meanReward = rewards.Average();
        float minReward = rewards.Min();
        float maxReward = rewards.Max();

        _output.WriteLine("");
        _output.WriteLine($"Mean reward: {meanReward:F3} ({meanReward * 100:F0}%)");
        _output.WriteLine($"Min reward:  {minReward:F3} ({minReward * 100:F0}%)");
        _output.WriteLine($"Max reward:  {maxReward:F3} ({maxReward * 100:F0}%)");

        // Solution should show some generalization
        Assert.True(meanReward > 0.3f, $"Mean reward too low: {meanReward:F3}");
    }

    private SpeciesSpec CreateCorridorTopology()
    {
        var random = new Random(42);
        // 9 inputs (9 distance sensors)
        // 12 hidden neurons (more capacity than CartPole due to higher dimensionality)
        // 2 outputs (steering, throttle)
        return new SpeciesBuilder()
            .AddInputRow(9)
            .AddHiddenRow(12, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeSparse(random)
            .Build();
    }
}
