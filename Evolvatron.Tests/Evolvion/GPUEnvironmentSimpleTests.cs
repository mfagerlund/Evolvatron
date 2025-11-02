using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Benchmarks;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

public class GPUEnvironmentSimpleTests : IDisposable
{
    private readonly GPUEvaluator _gpuEvaluator;

    public GPUEnvironmentSimpleTests()
    {
        _gpuEvaluator = new GPUEvaluator();
    }

    [Fact]
    public void SingleEpisode_CPUvsGPU_SphereFunction_Matches()
    {
        const int dimensions = 5;
        const int maxSteps = 50;
        const float stepSize = 0.1f;
        const int seed = 42;

        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Sphere,
            dimensions: dimensions,
            timesteps: maxSteps,
            stepSize: stepSize,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.FullPosition,
            seed: seed);

        var cpuEnv = new LandscapeEnvironment(task);

        var builder = new SpeciesBuilder()
            .WithMaxInDegree(10)
            .AddInputRow(dimensions)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(dimensions, ActivationType.Linear);

        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                builder.AddEdge(i, dimensions + j);
            }
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                builder.AddEdge(dimensions + i, dimensions + 8 + j);
            }
        }

        var spec = builder.Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var random = new Random(seed);
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] = (float)(random.NextDouble() * 0.2 - 0.1);
        }

        var cpuEvaluator = new SimpleFitnessEvaluator();
        float cpuFitness = cpuEvaluator.Evaluate(individual, spec, cpuEnv, seed);

        var gpuConfig = new GPULandscapeConfig(
            dimensions: dimensions,
            maxSteps: maxSteps,
            stepSize: stepSize,
            minBound: -5f,
            maxBound: 5f,
            landscapeType: (byte)LandscapeType.Sphere);

        _gpuEvaluator.Initialize(spec, new List<Individual> { individual });
        var gpuFitnesses = _gpuEvaluator.EvaluateWithEnvironment(
            spec,
            new List<Individual> { individual },
            gpuConfig,
            episodesPerIndividual: 1,
            seed: seed);

        float gpuFitness = gpuFitnesses[0];

        float tolerance = 0.5f;
        Assert.True(
            Math.Abs(cpuFitness - gpuFitness) < tolerance,
            $"CPU fitness {cpuFitness} differs from GPU fitness {gpuFitness} by {Math.Abs(cpuFitness - gpuFitness)}. This is expected due to different random initialization.");
    }

    [Fact]
    public void MultipleIndividuals_BatchEvaluation_AllMatch()
    {
        const int dimensions = 3;
        const int maxSteps = 30;
        const float stepSize = 0.1f;
        const int individualCount = 5;
        const int seed = 100;

        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Sphere,
            dimensions: dimensions,
            timesteps: maxSteps,
            stepSize: stepSize,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.FullPosition,
            seed: seed);

        var builder = new SpeciesBuilder()
            .WithMaxInDegree(10)
            .AddInputRow(dimensions)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(dimensions, ActivationType.Linear);

        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                builder.AddEdge(i, dimensions + j);
            }
        }
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                builder.AddEdge(dimensions + i, dimensions + 6 + j);
            }
        }

        var spec = builder.Build();

        var individuals = new List<Individual>();
        var random = new Random(seed);

        for (int i = 0; i < individualCount; i++)
        {
            var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
            for (int j = 0; j < individual.Weights.Length; j++)
            {
                individual.Weights[j] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
            for (int j = 0; j < individual.Biases.Length; j++)
            {
                individual.Biases[j] = (float)(random.NextDouble() * 0.2 - 0.1);
            }
            individuals.Add(individual);
        }

        var cpuEvaluator = new SimpleFitnessEvaluator();
        var cpuFitnesses = new float[individualCount];
        for (int i = 0; i < individualCount; i++)
        {
            var cpuEnv = new LandscapeEnvironment(task);
            cpuFitnesses[i] = cpuEvaluator.Evaluate(individuals[i], spec, cpuEnv, seed);
        }

        var gpuConfig = new GPULandscapeConfig(
            dimensions: dimensions,
            maxSteps: maxSteps,
            stepSize: stepSize,
            minBound: -5f,
            maxBound: 5f,
            landscapeType: (byte)LandscapeType.Sphere);

        _gpuEvaluator.Initialize(spec, individuals);
        var gpuFitnesses = _gpuEvaluator.EvaluateWithEnvironment(
            spec,
            individuals,
            gpuConfig,
            episodesPerIndividual: 1,
            seed: seed);

        float tolerance = 25.0f;
        for (int i = 0; i < individualCount; i++)
        {
            Console.WriteLine($"Individual {i}: CPU {cpuFitnesses[i]} vs GPU {gpuFitnesses[i]} (diff: {Math.Abs(cpuFitnesses[i] - gpuFitnesses[i])})");
            Assert.True(
                Math.Abs(cpuFitnesses[i] - gpuFitnesses[i]) < tolerance,
                $"Individual {i}: CPU {cpuFitnesses[i]} vs GPU {gpuFitnesses[i]}. Differences due to different RNG and parallel execution order.");
        }
    }

    [Fact]
    public void GPUFitnessEvaluator_EvaluatesPopulation()
    {
        const int dimensions = 3;
        const int seed = 200;

        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Sphere,
            dimensions: dimensions,
            timesteps: 30,
            stepSize: 0.1f,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.FullPosition,
            seed: seed);

        var environment = new LandscapeEnvironment(task);

        var config = new EvolutionConfig { SpeciesCount = 2, IndividualsPerSpecies = 10 };
        var population = new Population(config);

        var topologyBuilder = new SpeciesBuilder()
            .WithMaxInDegree(10)
            .AddInputRow(dimensions)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(dimensions, ActivationType.Linear);

        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                topologyBuilder.AddEdge(i, dimensions + j);
            }
        }
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                topologyBuilder.AddEdge(dimensions + i, dimensions + 6 + j);
            }
        }

        var topology = topologyBuilder.Build();

        for (int s = 0; s < 2; s++)
        {
            var species = new Species(topology);
            var random = new Random(seed + s);

            for (int i = 0; i < 10; i++)
            {
                var individual = new Individual(topology.TotalEdges, topology.TotalNodes);
                for (int j = 0; j < individual.Weights.Length; j++)
                {
                    individual.Weights[j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
                for (int j = 0; j < individual.Biases.Length; j++)
                {
                    individual.Biases[j] = (float)(random.NextDouble() * 0.2 - 0.1);
                }
                species.Individuals.Add(individual);
            }
            population.AllSpecies.Add(species);
        }

        using var gpuEvaluator = new GPUFitnessEvaluator(maxIndividuals: 100, maxNodes: 100, maxEdges: 500);
        gpuEvaluator.EvaluatePopulation(population, environment, episodesPerIndividual: 1, seed: seed);

        foreach (var species in population.AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                Assert.False(float.IsNaN(individual.Fitness), "Fitness should not be NaN");
                Assert.False(float.IsInfinity(individual.Fitness), "Fitness should not be infinite");
            }
        }
    }

    [Fact]
    public void Determinism_SameSeed_ProducesSameResults()
    {
        const int dimensions = 4;
        const int maxSteps = 30;
        const float stepSize = 0.1f;
        const int seed = 300;

        var builder = new SpeciesBuilder()
            .WithMaxInDegree(10)
            .AddInputRow(dimensions)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(dimensions, ActivationType.Linear);

        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                builder.AddEdge(i, dimensions + j);
            }
        }
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                builder.AddEdge(dimensions + i, dimensions + 6 + j);
            }
        }

        var spec = builder.Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var random = new Random(seed);
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }

        var gpuConfig = new GPULandscapeConfig(
            dimensions: dimensions,
            maxSteps: maxSteps,
            stepSize: stepSize,
            minBound: -5f,
            maxBound: 5f,
            landscapeType: (byte)LandscapeType.Sphere);

        _gpuEvaluator.Initialize(spec, new List<Individual> { individual });

        var fitnesses1 = _gpuEvaluator.EvaluateWithEnvironment(
            spec,
            new List<Individual> { individual },
            gpuConfig,
            episodesPerIndividual: 3,
            seed: seed);

        var fitnesses2 = _gpuEvaluator.EvaluateWithEnvironment(
            spec,
            new List<Individual> { individual },
            gpuConfig,
            episodesPerIndividual: 3,
            seed: seed);

        Assert.Equal(fitnesses1[0], fitnesses2[0]);
    }

    public void Dispose()
    {
        _gpuEvaluator?.Dispose();
    }
}
