using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

public class GPUEvaluatorTests : IDisposable
{
    private readonly GPUEvaluator _gpuEvaluator;

    public GPUEvaluatorTests()
    {
        _gpuEvaluator = new GPUEvaluator();
    }

    public void Dispose()
    {
        _gpuEvaluator?.Dispose();
    }

    [Fact]
    public void GPUEvaluator_SimpleNetwork_MatchesCPU()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Linear)
            .AddEdge(0, 4)
            .AddEdge(0, 5)
            .AddEdge(1, 4)
            .AddEdge(1, 6)
            .AddEdge(2, 5)
            .AddEdge(2, 7)
            .AddEdge(3, 6)
            .AddEdge(3, 7)
            .AddEdge(4, 12)
            .AddEdge(5, 12)
            .AddEdge(6, 13)
            .AddEdge(7, 13)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var random = new Random(42);
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] = (float)(random.NextDouble() * 0.2 - 0.1);
        }

        var cpuEvaluator = new CPUEvaluator(spec);
        _gpuEvaluator.Initialize(spec, new List<Individual> { individual });

        var inputs = new float[] { 0.5f, -0.3f, 0.8f, -0.2f };

        var cpuOutput = cpuEvaluator.Evaluate(individual, inputs);
        var gpuOutput = _gpuEvaluator.Evaluate(individual, inputs);

        Assert.Equal(cpuOutput.Length, gpuOutput.Length);
        for (int i = 0; i < cpuOutput.Length; i++)
        {
            Assert.Equal(cpuOutput[i], gpuOutput[i], precision: 6);
        }
    }

    [Fact]
    public void GPUEvaluator_BatchEvaluation_MatchesCPU()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Linear)
            .AddEdge(0, 4)
            .AddEdge(0, 5)
            .AddEdge(1, 5)
            .AddEdge(1, 6)
            .AddEdge(2, 6)
            .AddEdge(2, 7)
            .AddEdge(3, 7)
            .AddEdge(3, 4)
            .AddEdge(4, 12)
            .AddEdge(5, 12)
            .AddEdge(6, 13)
            .AddEdge(7, 13)
            .AddEdge(5, 13)
            .AddEdge(6, 12)
            .Build();

        var individuals = new List<Individual>();
        var random = new Random(123);

        for (int ind = 0; ind < 10; ind++)
        {
            var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
            for (int i = 0; i < individual.Weights.Length; i++)
            {
                individual.Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
            for (int i = 0; i < individual.Biases.Length; i++)
            {
                individual.Biases[i] = (float)(random.NextDouble() * 0.2 - 0.1);
            }
            individuals.Add(individual);
        }

        var cpuEvaluator = new CPUEvaluator(spec);
        _gpuEvaluator.Initialize(spec, individuals);

        var inputs = new float[10, 4];
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                inputs[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

        var gpuOutputs = _gpuEvaluator.EvaluateBatch(inputs);

        for (int i = 0; i < 10; i++)
        {
            var inputArray = new float[4];
            for (int j = 0; j < 4; j++)
            {
                inputArray[j] = inputs[i, j];
            }

            var cpuOutput = cpuEvaluator.Evaluate(individuals[i], inputArray);

            for (int j = 0; j < cpuOutput.Length; j++)
            {
                Assert.Equal(cpuOutput[j], gpuOutputs[i, j], precision: 6);
            }
        }
    }

    [Fact]
    public void GPUEvaluator_AllActivationTypes_MatchesCPU()
    {
        var allActivations = new[]
        {
            ActivationType.Linear,
            ActivationType.Tanh,
            ActivationType.Sigmoid,
            ActivationType.ReLU,
            ActivationType.LeakyReLU,
            ActivationType.ELU,
            ActivationType.Softsign,
            ActivationType.Softplus,
            ActivationType.Sin,
            ActivationType.Gaussian,
            ActivationType.GELU
        };

        var spec = new SpeciesBuilder()
            .WithMaxInDegree(15)
            .AddInputRow(3)
            .AddHiddenRow(11, allActivations)
            .AddOutputRow(2, ActivationType.Linear)
            .AddEdge(0, 3)
            .AddEdge(1, 4)
            .AddEdge(2, 5)
            .AddEdge(0, 6)
            .AddEdge(1, 7)
            .AddEdge(2, 8)
            .AddEdge(0, 9)
            .AddEdge(1, 10)
            .AddEdge(2, 11)
            .AddEdge(0, 12)
            .AddEdge(1, 13)
            .AddEdge(3, 14)
            .AddEdge(4, 14)
            .AddEdge(5, 14)
            .AddEdge(6, 14)
            .AddEdge(7, 15)
            .AddEdge(8, 15)
            .AddEdge(9, 15)
            .AddEdge(10, 15)
            .AddEdge(11, 15)
            .AddEdge(12, 15)
            .AddEdge(13, 15)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var random = new Random(999);

        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] = (float)(random.NextDouble() * 0.2 - 0.1);
        }

        for (int i = 0; i < 11; i++)
        {
            individual.Activations[3 + i] = allActivations[i];
            if (allActivations[i] == ActivationType.LeakyReLU)
            {
                individual.NodeParams[(3 + i) * 4] = 0.01f;
            }
            else if (allActivations[i] == ActivationType.ELU)
            {
                individual.NodeParams[(3 + i) * 4] = 1.0f;
            }
        }

        var cpuEvaluator = new CPUEvaluator(spec);
        _gpuEvaluator.Initialize(spec, new List<Individual> { individual });

        var inputs = new float[] { 0.5f, -0.5f, 1.0f };

        var cpuOutput = cpuEvaluator.Evaluate(individual, inputs);
        var gpuOutput = _gpuEvaluator.Evaluate(individual, inputs);

        Assert.Equal(cpuOutput.Length, gpuOutput.Length);
        for (int i = 0; i < cpuOutput.Length; i++)
        {
            Assert.Equal(cpuOutput[i], gpuOutput[i], precision: 6);
        }
    }

    [Fact]
    public void GPUEvaluator_DeeperNetwork_MatchesCPU()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddHiddenRow(4, ActivationType.Sigmoid)
            .AddOutputRow(2, ActivationType.Linear)
            .AddEdge(0, 4)
            .AddEdge(0, 5)
            .AddEdge(1, 6)
            .AddEdge(1, 7)
            .AddEdge(2, 8)
            .AddEdge(2, 9)
            .AddEdge(3, 10)
            .AddEdge(3, 11)
            .AddEdge(4, 12)
            .AddEdge(5, 13)
            .AddEdge(6, 14)
            .AddEdge(7, 15)
            .AddEdge(8, 16)
            .AddEdge(9, 17)
            .AddEdge(12, 18)
            .AddEdge(13, 19)
            .AddEdge(14, 20)
            .AddEdge(15, 21)
            .AddEdge(18, 22)
            .AddEdge(19, 22)
            .AddEdge(20, 23)
            .AddEdge(21, 23)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var random = new Random(555);

        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] = (float)(random.NextDouble() * 0.2 - 0.1);
        }

        var cpuEvaluator = new CPUEvaluator(spec);
        _gpuEvaluator.Initialize(spec, new List<Individual> { individual });

        var inputs = new float[] { 0.3f, -0.7f, 0.9f, -0.1f };

        var cpuOutput = cpuEvaluator.Evaluate(individual, inputs);
        var gpuOutput = _gpuEvaluator.Evaluate(individual, inputs);

        Assert.Equal(cpuOutput.Length, gpuOutput.Length);
        for (int i = 0; i < cpuOutput.Length; i++)
        {
            Assert.Equal(cpuOutput[i], gpuOutput[i], precision: 6);
        }
    }
}
