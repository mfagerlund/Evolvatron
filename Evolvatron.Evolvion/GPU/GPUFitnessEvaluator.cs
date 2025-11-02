using Evolvatron.Evolvion.Benchmarks;
using Evolvatron.Evolvion.Environments;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated fitness evaluator for Evolvion populations.
/// Replaces SimpleFitnessEvaluator with massive parallel evaluation on GPU.
/// </summary>
public class GPUFitnessEvaluator : IDisposable
{
    private readonly GPUEvaluator _gpuEvaluator;
    private readonly int _maxIndividuals;
    private readonly int _maxNodes;
    private readonly int _maxEdges;

    public GPUFitnessEvaluator(int maxIndividuals = 1000, int maxNodes = 100, int maxEdges = 500)
    {
        _maxIndividuals = maxIndividuals;
        _maxNodes = maxNodes;
        _maxEdges = maxEdges;
        _gpuEvaluator = new GPUEvaluator();
    }

    /// <summary>
    /// Evaluates all individuals in a population using GPU acceleration.
    /// Currently only supports LandscapeEnvironment.
    /// </summary>
    public void EvaluatePopulation(
        Population population,
        IEnvironment environment,
        int episodesPerIndividual = 5,
        int seed = 0)
    {
        if (environment is LandscapeEnvironment landscapeEnv)
        {
            EvaluatePopulationWithLandscape(population, landscapeEnv, episodesPerIndividual, seed);
        }
        else if (environment is XOREnvironment xorEnv)
        {
            EvaluatePopulationWithXOR(population, xorEnv, episodesPerIndividual, seed);
        }
        else if (environment is SpiralEnvironment spiralEnv)
        {
            EvaluatePopulationWithSpiral(population, spiralEnv, seed);
        }
        else
        {
            throw new ArgumentException(
                $"GPUFitnessEvaluator does not support environment type: {environment.GetType().Name}");
        }
    }

    private void EvaluatePopulationWithLandscape(
        Population population,
        LandscapeEnvironment landscapeEnv,
        int episodesPerIndividual,
        int seed)
    {
        var landscapeConfig = CreateLandscapeConfig(landscapeEnv);

        foreach (var species in population.AllSpecies)
        {
            if (species.Individuals.Count == 0)
                continue;

            if (species.Individuals.Count > _maxIndividuals)
            {
                throw new InvalidOperationException(
                    $"Species has {species.Individuals.Count} individuals but GPU evaluator max is {_maxIndividuals}");
            }

            _gpuEvaluator.Initialize(species.Topology, species.Individuals);

            var fitnessValues = _gpuEvaluator.EvaluateWithEnvironment(
                species.Topology,
                species.Individuals,
                landscapeConfig,
                episodesPerIndividual,
                seed);

            for (int i = 0; i < species.Individuals.Count; i++)
            {
                var individual = species.Individuals[i];
                individual.Fitness = fitnessValues[i];
                species.Individuals[i] = individual;
            }
        }
    }

    private void EvaluatePopulationWithXOR(
        Population population,
        XOREnvironment xorEnv,
        int episodesPerIndividual,
        int seed)
    {
        foreach (var species in population.AllSpecies)
        {
            if (species.Individuals.Count == 0)
                continue;

            if (species.Individuals.Count > _maxIndividuals)
            {
                throw new InvalidOperationException(
                    $"Species has {species.Individuals.Count} individuals but GPU evaluator max is {_maxIndividuals}");
            }

            _gpuEvaluator.Initialize(species.Topology, species.Individuals);

            var fitnessValues = _gpuEvaluator.EvaluateWithXOR(
                species.Topology,
                species.Individuals,
                episodesPerIndividual,
                seed);

            for (int i = 0; i < species.Individuals.Count; i++)
            {
                var individual = species.Individuals[i];
                individual.Fitness = fitnessValues[i];
                species.Individuals[i] = individual;
            }
        }
    }

    private void EvaluatePopulationWithSpiral(
        Population population,
        SpiralEnvironment spiralEnv,
        int seed)
    {
        var allPoints = spiralEnv.GetAllPoints();
        int pointsPerSpiral = allPoints.Count / 2;

        foreach (var species in population.AllSpecies)
        {
            if (species.Individuals.Count == 0)
                continue;

            if (species.Individuals.Count > _maxIndividuals)
            {
                throw new InvalidOperationException(
                    $"Species has {species.Individuals.Count} individuals but GPU evaluator max is {_maxIndividuals}");
            }

            _gpuEvaluator.Initialize(species.Topology, species.Individuals);

            var fitnessValues = _gpuEvaluator.EvaluateWithSpiral(
                species.Topology,
                species.Individuals,
                pointsPerSpiral,
                0.0f,
                seed);

            for (int i = 0; i < species.Individuals.Count; i++)
            {
                var individual = species.Individuals[i];
                individual.Fitness = fitnessValues[i];
                species.Individuals[i] = individual;
            }
        }
    }

    /// <summary>
    /// Creates GPU landscape config from LandscapeEnvironment.
    /// </summary>
    private GPULandscapeConfig CreateLandscapeConfig(LandscapeEnvironment env)
    {
        var landscapeType = DetermineLandscapeType(env);

        return new GPULandscapeConfig(
            dimensions: env.OutputCount,
            maxSteps: env.MaxSteps,
            stepSize: GetStepSize(env),
            minBound: GetMinBound(env),
            maxBound: GetMaxBound(env),
            landscapeType: (byte)landscapeType);
    }

    private LandscapeType DetermineLandscapeType(LandscapeEnvironment env)
    {
        var task = GetLandscapeTask(env);
        var landscapeFunc = task.GetLandscapeFunction();

        var testPoint = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float sphereValue = OptimizationLandscapes.Sphere(testPoint);
        float rosenbrockValue = OptimizationLandscapes.Rosenbrock(testPoint);

        float actualValue = landscapeFunc(testPoint);

        if (Math.Abs(actualValue - sphereValue) < 1e-4f)
            return LandscapeType.Sphere;
        if (Math.Abs(actualValue - rosenbrockValue) < 1e-4f)
            return LandscapeType.Rosenbrock;

        return LandscapeType.Sphere;
    }

    private LandscapeNavigationTask GetLandscapeTask(LandscapeEnvironment env)
    {
        var field = typeof(LandscapeEnvironment).GetField(
            "_task",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (field == null)
            throw new InvalidOperationException("Unable to access _task field from LandscapeEnvironment");

        return (LandscapeNavigationTask)field.GetValue(env)!;
    }

    private float GetStepSize(LandscapeEnvironment env)
    {
        var task = GetLandscapeTask(env);
        return task.GetStepSize();
    }

    private float GetMinBound(LandscapeEnvironment env)
    {
        var task = GetLandscapeTask(env);
        return task.GetMinBound();
    }

    private float GetMaxBound(LandscapeEnvironment env)
    {
        var task = GetLandscapeTask(env);
        return task.GetMaxBound();
    }

    public void Dispose()
    {
        _gpuEvaluator?.Dispose();
    }
}
