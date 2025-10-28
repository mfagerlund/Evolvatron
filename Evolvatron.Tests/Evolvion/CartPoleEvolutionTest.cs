using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// End-to-end test: Evolve a neural network to balance CartPole.
/// This is a significantly harder problem than XOR.
/// </summary>
public class CartPoleEvolutionTest
{
    private readonly ITestOutputHelper _output;

    public CartPoleEvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void EvolutionCanSolveCartPole()
    {
        // Create topology: 4 inputs -> 8 hidden -> 1 output
        var topology = CreateCartPoleTopology();

        // Configure evolution (using tuned defaults from XOR sweep)
        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 100,  // 400 total
            Elites = 4,
            TournamentSize = 4
            // Using default MutationRates (tuned from XOR sweep)
        };

        // Initialize population
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Create environment and evaluator
        var environment = new CartPoleEnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        // Evolution loop
        int maxGenerations = 200;
        float successThreshold = 500f; // Survive 500+ steps consistently

        _output.WriteLine("=== CARTPOLE EVOLUTION ===");
        _output.WriteLine($"Topology: {string.Join("-", topology.RowCounts)}");
        _output.WriteLine($"Population: {config.SpeciesCount} species × {config.IndividualsPerSpecies} = {config.SpeciesCount * config.IndividualsPerSpecies} total");
        _output.WriteLine($"Success threshold: {successThreshold} total reward");
        _output.WriteLine("");

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            // Evaluate all individuals
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // Get best individual
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            if (gen % 10 == 0 || bestFitness >= successThreshold)
            {
                _output.WriteLine($"Generation {gen,3}: Best Fitness = {bestFitness,7:F1}");
            }

            // Check for success
            if (bestFitness >= successThreshold)
            {
                _output.WriteLine("");
                _output.WriteLine($"SUCCESS! Solved CartPole in {gen} generations with fitness {bestFitness:F1}");
                _output.WriteLine("");

                // Verify solution on multiple seeds
                VerifyCartPoleSolution(best.Value.individual, best.Value.species.Topology, environment, evaluator);
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
        _output.WriteLine($"Final best fitness: {finalFitness:F1} (threshold: {successThreshold:F1})");

        // Assert some progress was made
        Assert.True(finalFitness > 100f,
            $"Evolution made insufficient progress. Final fitness: {finalFitness:F1}");
    }

    private void VerifyCartPoleSolution(
        Individual individual,
        SpeciesSpec topology,
        CartPoleEnvironment environment,
        SimpleFitnessEvaluator evaluator)
    {
        _output.WriteLine("=== VERIFICATION ACROSS MULTIPLE SEEDS ===");

        var cpuEval = new CPUEvaluator(topology);
        var seeds = new[] { 100, 101, 102, 103, 104 };
        var rewards = new float[seeds.Length];

        for (int i = 0; i < seeds.Length; i++)
        {
            float reward = evaluator.Evaluate(individual, topology, environment, seed: seeds[i]);
            rewards[i] = reward;
            _output.WriteLine($"  Seed {seeds[i]}: Reward = {reward,7:F1}");
        }

        float meanReward = rewards.Average();
        float minReward = rewards.Min();
        float maxReward = rewards.Max();

        _output.WriteLine("");
        _output.WriteLine($"Mean reward: {meanReward:F1}");
        _output.WriteLine($"Min reward:  {minReward:F1}");
        _output.WriteLine($"Max reward:  {maxReward:F1}");

        // Solution should be reasonably robust
        Assert.True(meanReward > 300f, $"Mean reward too low: {meanReward:F1}");
        Assert.True(minReward > 150f, $"Min reward too low (not robust): {minReward:F1}");
    }

    private SpeciesSpec CreateCartPoleTopology()
    {
        var random = new Random(42);
        // 4 inputs (cart_pos, cart_vel, pole_angle, pole_angular_vel)
        // 8 hidden neurons (more capacity than XOR)
        // 1 output (force)
        return new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(8, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(8)
            .InitializeSparse(random)
            .Build();
    }
}
