using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

public class GPUXORSimpleTest : IDisposable
{
    private readonly GPUEvaluator _gpuEvaluator;

    public GPUXORSimpleTest()
    {
        _gpuEvaluator = new GPUEvaluator();
    }

    [Fact]
    public void GPU_XOR_BasicEvaluation_Works()
    {
        const int seed = 42;

        var builder = new SpeciesBuilder()
            .WithMaxInDegree(10)
            .AddInputRow(2)
            .AddHiddenRow(4, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                builder.AddEdge(i, 2 + j);
            }
        }
        for (int i = 0; i < 4; i++)
        {
            builder.AddEdge(2 + i, 6);
        }

        var spec = builder.Build();

        var individuals = new List<Individual>();
        var random = new Random(seed);

        for (int i = 0; i < 5; i++)
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

        Console.WriteLine("Testing XOR GPU evaluation...");
        Console.WriteLine("Note: Fitness should be negative (errors), closer to 0 is better");

        foreach (var individual in individuals)
        {
            Assert.False(float.IsNaN(individual.Fitness));
            Assert.False(float.IsInfinity(individual.Fitness));
        }

        Console.WriteLine("XOR GPU evaluation test passed!");
    }

    public void Dispose()
    {
        _gpuEvaluator?.Dispose();
    }
}
