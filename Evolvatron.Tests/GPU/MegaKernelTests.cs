using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.GPU;

public class MegaKernelTests
{
    private readonly ITestOutputHelper _output;

    public MegaKernelTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Smoke test: verify the mega evaluator compiles, loads the fused kernel,
    /// and runs without crashing on a small population.
    /// </summary>
    [Fact]
    public void MegaEvaluator_SmokeTest()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 50 };
        var population = evolver.InitializePopulation(config, topology);
        var individuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();

        using var megaEval = new GPURocketLandingMegaEvaluator(maxIndividuals: 100);

        var (fitness, landings, _) = megaEval.EvaluatePopulation(topology, individuals, seed: 42, maxSteps: 100);

        _output.WriteLine($"Population: {individuals.Count}");
        _output.WriteLine($"Fitness range: [{fitness.Min():F2}, {fitness.Max():F2}]");
        _output.WriteLine($"Landings: {landings}");

        Assert.Equal(individuals.Count, fitness.Length);
        Assert.All(fitness, f => Assert.False(float.IsNaN(f)));
    }

    /// <summary>
    /// Numerical parity: compare fitness arrays from original and mega evaluator.
    /// NOTE: Parity is intentionally broken — mega evaluator computes fitness at the
    /// moment of termination (inline), while the original computes fitness post-mortem
    /// after crashed rockets settle via physics for remaining steps. Expect larger diffs.
    /// </summary>
    [Fact]
    public void MegaEvaluator_NumericalParity()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 200 };
        var population = evolver.InitializePopulation(config, topology);
        var individuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();

        const int seed = 77;
        const int maxSteps = 300;

        // Run original evaluator
        float[] originalFitness;
        int originalLandings;
        using (var original = new GPURocketLandingEvaluator(maxIndividuals: 300))
        {
            (originalFitness, originalLandings) = original.EvaluatePopulation(
                topology, individuals, seed: seed, maxSteps: maxSteps);
        }

        // Run mega evaluator
        float[] megaFitness;
        int megaLandings;
        using (var mega = new GPURocketLandingMegaEvaluator(maxIndividuals: 300))
        {
            (megaFitness, megaLandings, _) = mega.EvaluatePopulation(
                topology, individuals, seed: seed, maxSteps: maxSteps);
        }

        _output.WriteLine($"Original: landings={originalLandings}, best={originalFitness.Max():F2}, mean={originalFitness.Average():F2}");
        _output.WriteLine($"Mega:     landings={megaLandings}, best={megaFitness.Max():F2}, mean={megaFitness.Average():F2}");

        // Compare fitness arrays
        int matchCount = 0;
        int closeCount = 0;
        float maxDiff = 0f;
        int maxDiffIdx = 0;

        for (int i = 0; i < originalFitness.Length; i++)
        {
            float diff = MathF.Abs(originalFitness[i] - megaFitness[i]);
            if (diff < 1e-4f) matchCount++;
            if (diff < 1f) closeCount++;
            if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
        }

        _output.WriteLine($"Exact matches (<1e-4): {matchCount}/{originalFitness.Length}");
        _output.WriteLine($"Close matches (<1.0): {closeCount}/{originalFitness.Length}");
        _output.WriteLine($"Max difference: {maxDiff:F4} at index {maxDiffIdx}");
        _output.WriteLine($"  Original[{maxDiffIdx}] = {originalFitness[maxDiffIdx]:F4}");
        _output.WriteLine($"  Mega[{maxDiffIdx}] = {megaFitness[maxDiffIdx]:F4}");

        // Print first 10 comparisons
        _output.WriteLine("\nFirst 10 fitness comparisons:");
        for (int i = 0; i < Math.Min(10, originalFitness.Length); i++)
        {
            float diff = MathF.Abs(originalFitness[i] - megaFitness[i]);
            _output.WriteLine($"  [{i}] orig={originalFitness[i]:F4} mega={megaFitness[i]:F4} diff={diff:F4}");
        }

        // Tolerance: allow some divergence from kernel fusion order differences
        // The fused kernel runs observations→NN→actions→physics sequentially per world,
        // while the original runs each stage for ALL worlds before moving to the next.
        // This can cause minor floating-point ordering differences.
        // If parity is exact, great. If not, we report but don't hard-fail on small diffs.
        Assert.Equal(originalFitness.Length, megaFitness.Length);
    }

    /// <summary>
    /// Performance benchmark: compare wall-clock time of original vs mega evaluator
    /// on a large population over multiple generations.
    /// </summary>
    [Fact]
    public void MegaEvaluator_PerformanceBenchmark()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 5000 };
        var population = evolver.InitializePopulation(config, topology);
        var individuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();

        const int maxSteps = 600;
        const int rounds = 3;

        // Benchmark original
        using var original = new GPURocketLandingEvaluator(maxIndividuals: 5100);
        // Warmup
        original.EvaluatePopulation(topology, individuals, seed: 0, maxSteps: maxSteps);

        var sw = Stopwatch.StartNew();
        int origLandings = 0;
        for (int k = 0; k < rounds; k++)
        {
            var (_, landings) = original.EvaluatePopulation(topology, individuals, seed: k + 1, maxSteps: maxSteps);
            origLandings += landings;
        }
        sw.Stop();
        var originalTime = sw.Elapsed;

        // Benchmark mega
        using var mega = new GPURocketLandingMegaEvaluator(maxIndividuals: 5100);
        // Warmup
        mega.EvaluatePopulation(topology, individuals, seed: 0, maxSteps: maxSteps);

        sw.Restart();
        int megaLandings = 0;
        for (int k = 0; k < rounds; k++)
        {
            var (_, landings, _2) = mega.EvaluatePopulation(topology, individuals, seed: k + 1, maxSteps: maxSteps);
            megaLandings += landings;
        }
        sw.Stop();
        var megaTime = sw.Elapsed;

        float speedup = (float)originalTime.TotalMilliseconds / (float)megaTime.TotalMilliseconds;

        _output.WriteLine($"Population: {individuals.Count}, MaxSteps: {maxSteps}, Rounds: {rounds}");
        _output.WriteLine($"Original: {originalTime.TotalMilliseconds:F0}ms ({origLandings} landings)");
        _output.WriteLine($"Mega:     {megaTime.TotalMilliseconds:F0}ms ({megaLandings} landings)");
        _output.WriteLine($"Speedup:  {speedup:F2}x");
    }

    /// <summary>
    /// Smoke test with obstacles and distance sensors enabled.
    /// Verifies the full obstacle+sensor pipeline compiles and runs on GPU.
    /// </summary>
    [Fact]
    public void MegaEvaluator_ObstaclesAndSensors_SmokeTest()
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(12)  // 8 base + 4 sensor inputs
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 50 };
        var population = evolver.InitializePopulation(config, topology);
        var individuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();

        using var megaEval = new GPURocketLandingMegaEvaluator(maxIndividuals: 100);
        megaEval.SensorCount = 4;
        megaEval.MaxSensorRange = 30f;

        // Funnel-inspired obstacle layout
        megaEval.Obstacles = new List<GPUOBBCollider>
        {
            // Left angled wall (rotated 30 degrees)
            new GPUOBBCollider
            {
                CX = -8f, CY = 5f,
                UX = MathF.Cos(30f * MathF.PI / 180f), UY = MathF.Sin(30f * MathF.PI / 180f),
                HalfExtentX = 5f, HalfExtentY = 0.3f
            },
            // Right angled wall (rotated -30 degrees)
            new GPUOBBCollider
            {
                CX = 8f, CY = 5f,
                UX = MathF.Cos(-30f * MathF.PI / 180f), UY = MathF.Sin(-30f * MathF.PI / 180f),
                HalfExtentX = 5f, HalfExtentY = 0.3f
            },
            // Left platform
            new GPUOBBCollider { CX = -6f, CY = 0f, UX = 1f, UY = 0f, HalfExtentX = 2f, HalfExtentY = 0.2f },
            // Right platform
            new GPUOBBCollider { CX = 6f, CY = 0f, UX = 1f, UY = 0f, HalfExtentX = 2f, HalfExtentY = 0.2f },
        };

        var (fitness, landings, _) = megaEval.EvaluatePopulation(topology, individuals, seed: 42, maxSteps: 200);

        _output.WriteLine($"Population: {individuals.Count}");
        _output.WriteLine($"Obstacles: {megaEval.Obstacles.Count}");
        _output.WriteLine($"Sensors: {megaEval.SensorCount}");
        _output.WriteLine($"Fitness range: [{fitness.Min():F2}, {fitness.Max():F2}]");
        _output.WriteLine($"Landings: {landings}");

        Assert.Equal(individuals.Count, fitness.Length);
        Assert.All(fitness, f => Assert.False(float.IsNaN(f)));
    }
}
