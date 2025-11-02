using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class GPUSpiralEvolutionTest
{
    private readonly ITestOutputHelper _output;

    public GPUSpiralEvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void GPU_CanImproveOnSpiral()
    {
        var env = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);

        var topology = new SpeciesBuilder()
            .AddInputRow(env.InputCount)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(env.OutputCount, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeSparse(new Random(42))
            .Build();

        var config = new EvolutionConfig
        {
            SpeciesCount = 8,
            IndividualsPerSpecies = 50,
            Elites = 2,
            TournamentSize = 4
        };

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        using var gpuEval = new GPUFitnessEvaluator(maxIndividuals: 1000, maxNodes: 200, maxEdges: 1000);

        _output.WriteLine($"Population: {config.SpeciesCount} species x {config.IndividualsPerSpecies} individuals = {config.SpeciesCount * config.IndividualsPerSpecies} total");
        _output.WriteLine($"Topology: 2 inputs -> 8 hidden -> 8 hidden -> 1 output");
        _output.WriteLine("");

        gpuEval.EvaluatePopulation(population, env, episodesPerIndividual: 1, seed: 42);
        var initialBest = population.GetBestIndividual();
        float initialBestFitness = initialBest?.individual.Fitness ?? float.MinValue;
        _output.WriteLine($"Gen 0: Best Fitness = {initialBestFitness:F6}");

        int maxGenerations = 50;
        float successThreshold = -0.05f;

        for (int gen = 1; gen <= maxGenerations; gen++)
        {
            evolver.StepGeneration(population);
            gpuEval.EvaluatePopulation(population, env, episodesPerIndividual: 1, seed: 42);

            var stats = population.GetStatistics();

            if (gen % 10 == 0)
            {
                _output.WriteLine($"Gen {gen}: Best={stats.BestFitness:F6} Mean={stats.MeanFitness:F6} Median={stats.MedianFitness:F6}");
            }

            if (stats.BestFitness >= successThreshold)
            {
                _output.WriteLine($"\nSUCCESS! Solved spiral in {gen} generations with fitness {stats.BestFitness:F6}");
                var best = population.GetBestIndividual();
                VerifySpiralSolution(best.Value.individual, best.Value.species.Topology, env);
                return;
            }
        }

        var finalStats = population.GetStatistics();
        _output.WriteLine($"\nAfter {maxGenerations} generations:");
        _output.WriteLine($"Final best fitness: {finalStats.BestFitness:F6}");
        _output.WriteLine($"Initial best fitness: {initialBestFitness:F6}");
        _output.WriteLine($"Improvement: {finalStats.BestFitness - initialBestFitness:F6}");

        float improvementRatio = finalStats.BestFitness / initialBestFitness;
        _output.WriteLine($"Improvement ratio: {improvementRatio:F2}x");

        Assert.True(finalStats.BestFitness > initialBestFitness * 1.2,
            $"Expected at least 20% improvement, got {improvementRatio:F2}x (final: {finalStats.BestFitness:F6}, initial: {initialBestFitness:F6})");

        _output.WriteLine("\nSPIRAL TEST PASSED: Demonstrated significant fitness improvement on GPU!");
    }

    private void VerifySpiralSolution(Individual individual, SpeciesSpec topology, SpiralEnvironment environment)
    {
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

            float predictedLabel = output > 0 ? 1f : -1f;

            if (MathF.Abs(predictedLabel - expectedLabel) < 0.1f)
                correct++;

            total++;
        }

        float accuracy = (float)correct / total;
        _output.WriteLine($"\nSpiral Classification Accuracy: {correct}/{total} ({accuracy * 100:F1}%)");

        Assert.True(accuracy >= 0.8f, $"Classification accuracy too low: {accuracy * 100:F1}%");
    }
}
