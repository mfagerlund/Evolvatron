using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class GPUPerformanceBenchmark : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly GPUEvaluator _gpuEvaluator;

    public GPUPerformanceBenchmark(ITestOutputHelper output)
    {
        _output = output;
        _gpuEvaluator = new GPUEvaluator(maxIndividuals: 3000);
    }

    public void Dispose()
    {
        _gpuEvaluator?.Dispose();
    }

    [Fact]
    public void BenchmarkCPU_vs_GPU_NeuralEvaluation()
    {
        int individualCount = 2500;
        int batchSize = 1000;
        int iterations = 1000;

        var builder = new SpeciesBuilder()
            .AddInputRow(5)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(5, ActivationType.Tanh);

        for (int i = 0; i < 5; i++)
        {
            for (int j = 5; j < 21; j++)
            {
                builder = builder.AddEdge(i, j);
            }
        }
        for (int i = 5; i < 21; i++)
        {
            for (int j = 21; j < 29; j++)
            {
                builder = builder.AddEdge(i, j);
            }
        }
        for (int i = 21; i < 29; i++)
        {
            for (int j = 29; j < 34; j++)
            {
                builder = builder.AddEdge(i, j);
            }
        }

        var spec = builder.WithMaxInDegree(20).Build();

        var individuals = new List<Individual>();
        var random = new Random(42);

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

        var inputs = new float[batchSize, 5];
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                inputs[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

        var cpuEvaluator = new CPUEvaluator(spec);

        var cpuWatch = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            for (int i = 0; i < batchSize; i++)
            {
                var input = new float[5];
                for (int j = 0; j < 5; j++)
                {
                    input[j] = inputs[i, j];
                }
                var _ = cpuEvaluator.Evaluate(individuals[i % individualCount], input);
            }
        }
        cpuWatch.Stop();

        _gpuEvaluator.Initialize(spec, individuals);

        var gpuWatch = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            var _ = _gpuEvaluator.EvaluateBatch(inputs);
        }
        gpuWatch.Stop();

        long cpuEvaluations = (long)iterations * batchSize;
        long gpuEvaluations = (long)iterations * batchSize;

        _output.WriteLine($"\n=== GPU Performance Benchmark ===");
        _output.WriteLine($"Configuration:");
        _output.WriteLine($"  Network: 5 inputs -> 16 hidden -> 8 hidden -> 5 outputs");
        _output.WriteLine($"  Total nodes: {spec.TotalNodes}");
        _output.WriteLine($"  Total edges: {spec.TotalEdges}");
        _output.WriteLine($"  Individuals: {individualCount}");
        _output.WriteLine($"  Batch size: {batchSize}");
        _output.WriteLine($"  Iterations: {iterations}");
        _output.WriteLine($"  Total evaluations: {cpuEvaluations}");
        _output.WriteLine($"");
        _output.WriteLine($"Results:");
        _output.WriteLine($"  CPU time: {cpuWatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"  GPU time: {gpuWatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"  Speedup: {(double)cpuWatch.ElapsedMilliseconds/gpuWatch.ElapsedMilliseconds:F2}x");
        _output.WriteLine($"");
        _output.WriteLine($"Throughput:");
        _output.WriteLine($"  CPU: {cpuEvaluations * 1000.0 / cpuWatch.ElapsedMilliseconds:F0} evals/sec");
        _output.WriteLine($"  GPU: {gpuEvaluations * 1000.0 / gpuWatch.ElapsedMilliseconds:F0} evals/sec");
    }
}
