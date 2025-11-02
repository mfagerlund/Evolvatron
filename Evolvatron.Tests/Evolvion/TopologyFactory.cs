namespace Evolvatron.Tests.Evolvion;

using Evolvatron.Evolvion;

public static class TopologyFactory
{
    private static readonly ActivationType[] StandardActivations = new[]
    {
        ActivationType.Linear,
        ActivationType.Tanh,
        ActivationType.ReLU,
        ActivationType.Sigmoid,
        ActivationType.LeakyReLU,
        ActivationType.ELU,
        ActivationType.Softsign,
        ActivationType.Softplus,
        ActivationType.Sin,
        ActivationType.Gaussian,
        ActivationType.GELU
    };

    public static SpeciesSpec CreateXOR(int seed = 42, int hiddenSize = 4)
    {
        var random = new Random(seed);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(8)
            .InitializeSparse(random)
            .Build();
    }

    public static SpeciesSpec CreateSpiral(int seed = 42)
    {
        var random = new Random(seed);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, StandardActivations)
            .AddHiddenRow(8, StandardActivations)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)
            .Build();
    }

    public static SpeciesSpec CreateCartPole(int seed = 42, int hiddenSize = 8)
    {
        var random = new Random(seed);
        return new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)
            .Build();
    }

    public static SpeciesSpec CreateLandscape(int dimensions, int seed = 42, int hiddenSize = 8)
    {
        var random = new Random(seed);
        return new SpeciesBuilder()
            .AddInputRow(dimensions)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddOutputRow(dimensions, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)
            .Build();
    }

    public static SpeciesSpec CreateCorridor(int seed = 42, int hiddenSize = 12)
    {
        var random = new Random(seed);
        return new SpeciesBuilder()
            .AddInputRow(5)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddHiddenRow(hiddenSize, StandardActivations)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)
            .Build();
    }
}
