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

    [Fact]
    public void BenchmarkGPUScaling()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        int[] scales = { 2000, 5000, 8000, 10000 };
        int maxScale = scales.Max();

        // Build a pool of individuals large enough for the biggest scale
        var evolver = new Evolver(seed: 42);
        var bigConfig = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = maxScale };
        var bigPop = evolver.InitializePopulation(bigConfig, topology);
        var pool = new List<Individual>();
        foreach (var species in bigPop.AllSpecies)
            pool.AddRange(species.Individuals);

        using var gpuEval = new GPURocketLandingEvaluator(maxIndividuals: maxScale + 100);
        _output.WriteLine($"GPU: {gpuEval.Accelerator.Name}");
        _output.WriteLine($"CPU threads: {Environment.ProcessorCount}");
        _output.WriteLine($"MaxSteps: 600 (5s at 120Hz)\n");
        _output.WriteLine($"   Pop |   GPU (ms) |   CPU (ms) |  Speedup");
        _output.WriteLine(new string('-', 50));

        // Warmup GPU with smallest scale
        var warmupList = pool.Take(scales[0]).ToList();
        gpuEval.EvaluatePopulation(topology, warmupList, seed: 99, maxSteps: 600);

        foreach (int n in scales)
        {
            var individuals = pool.Take(n).ToList();

            // GPU: 3 runs, take median
            var gpuTimes = new List<double>();
            for (int r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                gpuEval.EvaluatePopulation(topology, individuals, seed: r, maxSteps: 600);
                sw.Stop();
                gpuTimes.Add(sw.Elapsed.TotalMilliseconds);
            }
            gpuTimes.Sort();
            double gpuMs = gpuTimes[1]; // median

            // CPU: 1 run (expensive)
            var sw2 = Stopwatch.StartNew();
            var results = new float[n];
            Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
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
                        var acts = evaluator.Evaluate(individuals[i], obs);
                        env.Step(acts);
                    }
                    results[i] = env.GetFinalFitness();
                });
            sw2.Stop();
            double cpuMs = sw2.Elapsed.TotalMilliseconds;

            double speedup = cpuMs / gpuMs;
            _output.WriteLine($"{n,6} | {gpuMs,10:F1} | {cpuMs,10:F1} | {speedup,7:F1}x");
        }
    }

    [Fact]
    public void BenchmarkGPUSaturation()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        int[] scales = { 5000, 10000, 15000, 20000, 30000, 40000, 50000 };
        int maxScale = scales.Max();

        var evolver = new Evolver(seed: 42);
        var bigConfig = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = maxScale };
        var bigPop = evolver.InitializePopulation(bigConfig, topology);
        var pool = new List<Individual>();
        foreach (var species in bigPop.AllSpecies)
            pool.AddRange(species.Individuals);

        using var gpuEval = new GPURocketLandingEvaluator(maxIndividuals: maxScale + 100);
        _output.WriteLine($"GPU: {gpuEval.Accelerator.Name}");
        _output.WriteLine($"MaxSteps: 600 (5s at 120Hz)");
        _output.WriteLine($"Finding GPU saturation point...\n");
        _output.WriteLine($"    Pop |   GPU (ms) |  ms/1K ind | evals/sec");
        _output.WriteLine(new string('-', 55));

        foreach (int n in scales)
        {
            var individuals = pool.Take(n).ToList();

            // Warmup at this scale (triggers buffer reallocation, not measured)
            gpuEval.EvaluatePopulation(topology, individuals, seed: 99, maxSteps: 600);

            // 3 runs, take median
            var gpuTimes = new List<double>();
            for (int r = 0; r < 3; r++)
            {
                var sw = Stopwatch.StartNew();
                gpuEval.EvaluatePopulation(topology, individuals, seed: r, maxSteps: 600);
                sw.Stop();
                gpuTimes.Add(sw.Elapsed.TotalMilliseconds);
            }
            gpuTimes.Sort();
            double gpuMs = gpuTimes[1];
            double msPerThousand = gpuMs / (n / 1000.0);
            double evalsPerSec = n / (gpuMs / 1000.0);

            _output.WriteLine($"{n,7} | {gpuMs,10:F1} | {msPerThousand,10:F2} | {evalsPerSec,9:F0}");
        }
    }
}
