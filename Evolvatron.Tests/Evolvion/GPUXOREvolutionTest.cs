using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class GPUXOREvolutionTest
{
    private readonly ITestOutputHelper _output;

    public GPUXOREvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void GPU_CanSolveXOR()
    {
        var env = new XOREnvironment();

        var topology = new SpeciesBuilder()
            .AddInputRow(env.InputCount)
            .AddHiddenRow(6, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid)
            .AddOutputRow(env.OutputCount, ActivationType.Tanh)
            .WithMaxInDegree(8)
            .InitializeSparse(new Random(42))
            .Build();

        var config = new EvolutionConfig
        {
            SpeciesCount = 3,
            IndividualsPerSpecies = 30,
            Elites = 2,
            TournamentSize = 3
        };

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        using var gpuEval = new GPUFitnessEvaluator(maxIndividuals: 1000, maxNodes: 100, maxEdges: 500);

        int maxGenerations = 100;
        float successThreshold = -0.01f;

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            gpuEval.EvaluatePopulation(population, env, episodesPerIndividual: 1, seed: 42);

            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            if (gen % 10 == 0)
            {
                _output.WriteLine($"Gen {gen}: Best Fitness = {bestFitness:F6}");
            }

            if (bestFitness >= successThreshold)
            {
                _output.WriteLine($"\nSUCCESS! Solved XOR in {gen} generations with fitness {bestFitness:F6}");
                VerifyXORSolution(best.Value.individual, best.Value.species.Topology);
                return;
            }

            evolver.StepGeneration(population);
        }

        var final = population.GetBestIndividual();
        float finalFitness = final?.individual.Fitness ?? float.MinValue;

        _output.WriteLine($"\nDid not fully converge after {maxGenerations} generations.");
        _output.WriteLine($"Final best fitness: {finalFitness:F6} (threshold: {successThreshold:F6})");

        Assert.True(finalFitness > -0.5f,
            $"Evolution made insufficient progress. Final fitness: {finalFitness:F6}");
    }

    private void VerifyXORSolution(Individual individual, SpeciesSpec topology)
    {
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

            Assert.True(error < 0.3f, $"Output error too large for input ({x}, {y})");
        }
    }
}
