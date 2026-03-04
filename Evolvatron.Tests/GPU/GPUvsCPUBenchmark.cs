using Evolvatron.Core;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.GPU;

public class GPUvsCPUBenchmark
{
    private readonly ITestOutputHelper _output;

    public GPUvsCPUBenchmark(ITestOutputHelper output) => _output = output;

    [Fact]
    public void BenchmarkGPUvsCPU()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var config = new EvolutionConfig
        {
            SpeciesCount = 5,
            IndividualsPerSpecies = 400,
        };

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Collect all individuals
        var allIndividuals = new List<Individual>();
        foreach (var species in population.AllSpecies)
            allIndividuals.AddRange(species.Individuals);

        int count = allIndividuals.Count;
        _output.WriteLine($"Population: {count} individuals");
        _output.WriteLine($"MaxSteps: 600 (5s at 120Hz)");
        _output.WriteLine($"3-body rocket with 2 joints\n");

        // --- GPU benchmark ---
        using var gpuEval = new GPURocketLandingEvaluator(maxIndividuals: count + 100);
        _output.WriteLine($"GPU: {gpuEval.Accelerator.Name}");

        // Warmup
        gpuEval.EvaluatePopulation(topology, allIndividuals, seed: 0, maxSteps: 600);

        var sw = Stopwatch.StartNew();
        const int gpuRuns = 5;
        for (int i = 0; i < gpuRuns; i++)
            gpuEval.EvaluatePopulation(topology, allIndividuals, seed: i, maxSteps: 600);
        sw.Stop();
        double gpuMs = sw.Elapsed.TotalMilliseconds / gpuRuns;
        _output.WriteLine($"GPU: {gpuMs:F1}ms per eval ({count} rockets x 600 steps)");

        // --- CPU benchmark (Parallel.For) ---
        sw.Restart();
        const int cpuRuns = 1; // CPU is slow, just 1 run
        for (int run = 0; run < cpuRuns; run++)
        {
            var results = new float[count];
            Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                i =>
                {
                    var env = new RocketEnvironment();
                    env.MaxSteps = 600;
                    var evaluator = new CPUEvaluator(topology);
                    var obs = new float[8];

                    env.Reset(i);
                    while (!env.IsTerminal())
                    {
                        env.GetObservations(obs);
                        var acts = evaluator.Evaluate(allIndividuals[i], obs);
                        env.Step(acts);
                    }
                    results[i] = env.GetFinalFitness();
                });
        }
        sw.Stop();
        double cpuMs = sw.Elapsed.TotalMilliseconds / cpuRuns;
        _output.WriteLine($"CPU: {cpuMs:F1}ms per eval ({count} rockets x 600 steps, {Environment.ProcessorCount} threads)");

        _output.WriteLine($"\nSpeedup: {cpuMs / gpuMs:F1}x");
    }
}
