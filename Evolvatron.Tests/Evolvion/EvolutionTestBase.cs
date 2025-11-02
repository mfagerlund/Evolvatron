namespace Evolvatron.Tests.Evolvion;

using Xunit;
using Xunit.Abstractions;
using Evolvatron.Evolvion;

public abstract class EvolutionTestBase
{
    protected readonly ITestOutputHelper _output;

    protected EvolutionTestBase(ITestOutputHelper output)
    {
        _output = output;
    }

    protected EvolutionResult RunEvolutionTest<TEnv>(
        TEnv environment,
        SpeciesSpec topology,
        EvolutionConfig config,
        float successThreshold,
        int maxGenerations,
        int seed = 42) where TEnv : IEnvironment
    {
        var evolver = new Evolver(seed);
        var population = evolver.InitializePopulation(config, topology);
        var evaluator = new SimpleFitnessEvaluator();

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            _output.WriteLine($"Gen {gen}: Best={bestFitness:F4}");

            if (bestFitness >= successThreshold)
            {
                _output.WriteLine($"SOLVED at generation {gen}!");
                return new EvolutionResult
                {
                    Solved = true,
                    Generations = gen,
                    BestIndividual = best.Value.individual,
                    BestFitness = bestFitness
                };
            }

            evolver.StepGeneration(population);
        }

        var final = population.GetBestIndividual();
        float finalFitness = final?.individual.Fitness ?? float.MinValue;
        _output.WriteLine($"Did not solve. Best fitness: {finalFitness:F4}");
        return new EvolutionResult
        {
            Solved = false,
            BestIndividual = final?.individual ?? default,
            BestFitness = finalFitness
        };
    }

    protected VerificationResult VerifyAcrossSeeds<TEnv>(
        Individual individual,
        SpeciesSpec topology,
        TEnv environment,
        int[] seeds) where TEnv : IEnvironment
    {
        var evaluator = new SimpleFitnessEvaluator();
        var rewards = new float[seeds.Length];

        _output.WriteLine($"Verifying solution across {seeds.Length} seeds:");
        for (int i = 0; i < seeds.Length; i++)
        {
            rewards[i] = evaluator.Evaluate(individual, topology, environment, seed: seeds[i]);
            _output.WriteLine($"  Seed {seeds[i]}: Reward = {rewards[i]:F3}");
        }

        var result = new VerificationResult
        {
            MeanReward = rewards.Average(),
            MinReward = rewards.Min(),
            MaxReward = rewards.Max()
        };

        _output.WriteLine($"Mean: {result.MeanReward:F3}, Min: {result.MinReward:F3}, Max: {result.MaxReward:F3}");
        return result;
    }
}

public struct EvolutionResult
{
    public bool Solved;
    public int Generations;
    public Individual BestIndividual;
    public float BestFitness;
}

public struct VerificationResult
{
    public float MeanReward;
    public float MinReward;
    public float MaxReward;
}
