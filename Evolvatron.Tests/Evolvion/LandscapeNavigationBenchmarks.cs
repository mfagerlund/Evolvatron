using Evolvatron.Evolvion.Benchmarks;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

[Trait("Category", "Benchmark")]
public class LandscapeNavigationBenchmarks
{
    private readonly ITestOutputHelper _output;

    public LandscapeNavigationBenchmarks(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Sphere_5D_Easy()
    {
        RunBenchmark(
            "Sphere-5D-Easy",
            OptimizationLandscapes.Sphere,
            dimensions: 5,
            timesteps: 50,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.FullPosition);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Rosenbrock_5D_Easy()
    {
        RunBenchmark(
            "Rosenbrock-5D-Easy",
            OptimizationLandscapes.Rosenbrock,
            dimensions: 5,
            timesteps: 100,
            minBound: -2f,
            maxBound: 2f,
            observationType: ObservationType.FullPosition);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Rastrigin_8D_Medium()
    {
        RunBenchmark(
            "Rastrigin-8D-Medium",
            OptimizationLandscapes.Rastrigin,
            dimensions: 8,
            timesteps: 100,
            minBound: -5.12f,
            maxBound: 5.12f,
            observationType: ObservationType.FullPosition);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Ackley_8D_Medium()
    {
        RunBenchmark(
            "Ackley-8D-Medium",
            OptimizationLandscapes.Ackley,
            dimensions: 8,
            timesteps: 100,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.FullPosition);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Rosenbrock_10D_GradientOnly()
    {
        RunBenchmark(
            "Rosenbrock-10D-GradientOnly",
            OptimizationLandscapes.Rosenbrock,
            dimensions: 10,
            timesteps: 150,
            minBound: -2f,
            maxBound: 2f,
            observationType: ObservationType.GradientOnly);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Rastrigin_15D_Hard()
    {
        RunBenchmark(
            "Rastrigin-15D-Hard",
            OptimizationLandscapes.Rastrigin,
            dimensions: 15,
            timesteps: 200,
            minBound: -5.12f,
            maxBound: 5.12f,
            observationType: ObservationType.FullPosition);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Schwefel_12D_Hard()
    {
        RunBenchmark(
            "Schwefel-12D-Hard",
            OptimizationLandscapes.Schwefel,
            dimensions: 12,
            timesteps: 200,
            minBound: -500f,
            maxBound: 500f,
            observationType: ObservationType.FullPosition);
    }

    [Fact(Skip = "Benchmark - run explicitly")]
    public void Ackley_15D_PartialObs()
    {
        RunBenchmark(
            "Ackley-15D-PartialObs",
            OptimizationLandscapes.Ackley,
            dimensions: 15,
            timesteps: 250,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.PartialObservability);
    }

    private void RunBenchmark(
        string name,
        LandscapeNavigationTask.LandscapeFunction landscape,
        int dimensions,
        int timesteps,
        float minBound,
        float maxBound,
        ObservationType observationType)
    {
        var task = new LandscapeNavigationTask(
            landscape,
            dimensions,
            timesteps,
            stepSize: 0.1f,
            minBound: minBound,
            maxBound: maxBound,
            observationType: observationType,
            seed: 42);

        var randomPolicy = CreateRandomPolicy(dimensions);

        var fitness = task.Evaluate(randomPolicy);

        _output.WriteLine($"Benchmark: {name}");
        _output.WriteLine($"  Dimensions: {dimensions}");
        _output.WriteLine($"  Timesteps: {timesteps}");
        _output.WriteLine($"  Observation Type: {observationType}");
        _output.WriteLine($"  Random Policy Fitness: {fitness:F4}");
        _output.WriteLine("");
    }

    private Func<float[], float[]> CreateRandomPolicy(int dimensions)
    {
        var random = new Random(123);
        return inputs =>
        {
            var outputs = new float[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                outputs[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
            return outputs;
        };
    }

    [Fact]
    public void VerifyLandscapeFunctions()
    {
        var origin = new float[] { 0f, 0f };

        float sphereAtOrigin = OptimizationLandscapes.Sphere(origin);
        Assert.Equal(0f, sphereAtOrigin, precision: 4);

        var rosenbrockOptimum = new float[] { 1f, 1f };
        float rosenbrockAtOptimum = OptimizationLandscapes.Rosenbrock(rosenbrockOptimum);
        Assert.Equal(0f, rosenbrockAtOptimum, precision: 4);

        float rastriginAtOrigin = OptimizationLandscapes.Rastrigin(origin);
        Assert.Equal(0f, rastriginAtOrigin, precision: 4);

        _output.WriteLine("Landscape function verification passed");
    }
}
