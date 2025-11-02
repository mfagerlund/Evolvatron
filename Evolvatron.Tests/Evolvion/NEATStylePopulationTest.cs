using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;
using System.Text;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Validation test for NEAT-style population structure refactor.
///
/// Goal: Prove that 20 species × 40 individuals with OR-based culling enables
/// topology exploration through frequent species turnover.
///
/// Expected behavior:
/// - Species count fluctuates between 8-20
/// - TotalSpeciesCreated grows to 30-50+ over 150 generations
/// - Culling events occur every 5-10 generations
/// - Final fitness is competitive with baseline (4×200 with culling disabled)
/// </summary>
public class NEATStylePopulationTest
{
    private readonly ITestOutputHelper _output;

    public NEATStylePopulationTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void NEATStyle_EnablesTopologyExploration()
    {
        const int seeds = 3;
        const int generations = 150;

        _output.WriteLine("=== NEAT-STYLE POPULATION VALIDATION ===");
        _output.WriteLine($"Configuration: 20 species × 40 individuals = 800 total");
        _output.WriteLine($"Culling: OR-based (grace=3, stagnation=6, diversity=0.08, performance=0.7)");
        _output.WriteLine($"MinSpeciesCount: 8 (allows culling)");
        _output.WriteLine($"Benchmark: Spiral classification (2 inputs → 8 hidden → 8 hidden → 1 output)");
        _output.WriteLine($"Seeds: {seeds}, Generations: {generations}");
        _output.WriteLine("");

        var neatResults = new List<RunMetrics>();

        for (int seed = 0; seed < seeds; seed++)
        {
            _output.WriteLine($"\n=== NEAT-STYLE RUN (Seed {seed}) ===");
            var metrics = RunNEATStyle(seed, generations);
            neatResults.Add(metrics);

            _output.WriteLine($"\nSeed {seed} Summary:");
            _output.WriteLine($"  TotalSpeciesCreated: {metrics.TotalSpeciesCreated}");
            _output.WriteLine($"  CullingEvents: {metrics.CullingEvents}");
            _output.WriteLine($"  Final Species Count: {metrics.FinalSpeciesCount}");
            _output.WriteLine($"  Gen0 Best: {metrics.Gen0BestFitness:F4}");
            _output.WriteLine($"  Gen150 Best: {metrics.Gen150BestFitness:F4}");
            _output.WriteLine($"  Improvement: {metrics.Gen150BestFitness - metrics.Gen0BestFitness:F4}");
        }

        // Aggregate NEAT-style results
        _output.WriteLine("\n\n=== NEAT-STYLE AGGREGATE RESULTS ===");
        double avgSpeciesCreated = neatResults.Average(r => r.TotalSpeciesCreated);
        double avgCullingEvents = neatResults.Average(r => r.CullingEvents);
        double avgGen150Fitness = neatResults.Average(r => r.Gen150BestFitness);

        _output.WriteLine($"Average TotalSpeciesCreated: {avgSpeciesCreated:F1} (expect >= 30)");
        _output.WriteLine($"Average CullingEvents: {avgCullingEvents:F1}");
        _output.WriteLine($"Average Gen150 Best Fitness: {avgGen150Fitness:F4}");

        // Run baseline comparison (4 species × 200, culling disabled via MinSpeciesCount=4)
        _output.WriteLine("\n\n=== BASELINE COMPARISON (4×200, Culling Disabled) ===");
        var baselineResults = new List<RunMetrics>();

        for (int seed = 0; seed < seeds; seed++)
        {
            _output.WriteLine($"\n=== BASELINE RUN (Seed {seed}) ===");
            var metrics = RunBaseline(seed, generations);
            baselineResults.Add(metrics);

            _output.WriteLine($"\nSeed {seed} Summary:");
            _output.WriteLine($"  TotalSpeciesCreated: {metrics.TotalSpeciesCreated}");
            _output.WriteLine($"  CullingEvents: {metrics.CullingEvents}");
            _output.WriteLine($"  Final Species Count: {metrics.FinalSpeciesCount}");
            _output.WriteLine($"  Gen0 Best: {metrics.Gen0BestFitness:F4}");
            _output.WriteLine($"  Gen150 Best: {metrics.Gen150BestFitness:F4}");
            _output.WriteLine($"  Improvement: {metrics.Gen150BestFitness - metrics.Gen0BestFitness:F4}");
        }

        double avgBaselineFitness = baselineResults.Average(r => r.Gen150BestFitness);
        _output.WriteLine($"\nAverage Baseline Gen150 Best Fitness: {avgBaselineFitness:F4}");

        // Final comparison
        _output.WriteLine("\n\n=== FINAL COMPARISON ===");
        _output.WriteLine($"NEAT-style Gen150 fitness: {avgGen150Fitness:F4}");
        _output.WriteLine($"Baseline Gen150 fitness:   {avgBaselineFitness:F4}");
        _output.WriteLine($"Difference: {avgGen150Fitness - avgBaselineFitness:F4} ({(avgGen150Fitness / avgBaselineFitness - 1.0) * 100:F1}%)");

        // Assertions
        _output.WriteLine("\n=== VALIDATION ===");

        // 1. Topology exploration is working (proof: species turnover)
        Assert.True(avgSpeciesCreated >= 30,
            $"NEAT-style should create >= 30 species over 150 gens, but only created {avgSpeciesCreated:F1}");
        _output.WriteLine($"✓ Topology exploration working: {avgSpeciesCreated:F1} species created");

        // 2. Culling is actually happening
        Assert.True(avgCullingEvents > 0,
            $"NEAT-style should have culling events, but had {avgCullingEvents:F1}");
        _output.WriteLine($"✓ Culling is active: {avgCullingEvents:F1} events on average");

        // 3. Fitness should be competitive (within 20% of baseline)
        double fitnessRatio = avgGen150Fitness / avgBaselineFitness;
        Assert.True(fitnessRatio >= 0.80,
            $"NEAT-style fitness should be >= 80% of baseline, but was {fitnessRatio * 100:F1}%");
        _output.WriteLine($"✓ Fitness is competitive: {fitnessRatio * 100:F1}% of baseline");

        _output.WriteLine("\n=== ALL VALIDATION CHECKS PASSED ===");
    }

    private RunMetrics RunNEATStyle(int seed, int generations)
    {
        var config = new EvolutionConfig
        {
            // NEAT-style configuration (new defaults)
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            MinSpeciesCount = 8,
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,

            // NEAT-style culling thresholds (use new defaults)
            GraceGenerations = 1,  // Allow culling to start at generation 2+
            StagnationThreshold = 6,
            SpeciesDiversityThreshold = 0.08f,
            RelativePerformanceThreshold = 0.7f
        };

        var evolver = new Evolver(seed);
        var random = new Random(seed);
        var topology = CreateSpiralTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        var metrics = new RunMetrics();
        var cullingEvents = new List<CullingEvent>();

        for (int gen = 0; gen < generations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            var stats = population.GetStatistics();
            if (gen == 0)
                metrics.Gen0BestFitness = stats.BestFitness;

            // Track TotalSpeciesCreated to detect culling (count stays constant due to immediate replacement)
            int speciesCreatedBefore = population.TotalSpeciesCreated;

            evolver.StepGeneration(population);

            int speciesCreatedAfter = population.TotalSpeciesCreated;

            // Detect culling events: TotalSpeciesCreated increases when new species are born
            if (speciesCreatedAfter > speciesCreatedBefore)
            {
                metrics.CullingEvents++;
                cullingEvents.Add(new CullingEvent
                {
                    Generation = gen,
                    SpeciesBefore = speciesCreatedBefore,
                    SpeciesAfter = speciesCreatedAfter
                });
            }

            // Log every 25 generations
            if (gen % 25 == 0 || gen == generations - 1)
            {
                _output.WriteLine($"  Gen {gen,3}: Species={population.AllSpecies.Count,2} " +
                    $"TotalCreated={population.TotalSpeciesCreated,2} " +
                    $"Best={stats.BestFitness:F4} Mean={stats.MeanFitness:F4}");
            }
        }

        // Final fitness
        evaluator.EvaluatePopulation(population, environment, seed: generations);
        var finalStats = population.GetStatistics();
        metrics.Gen150BestFitness = finalStats.BestFitness;
        metrics.TotalSpeciesCreated = population.TotalSpeciesCreated;
        metrics.FinalSpeciesCount = population.AllSpecies.Count;

        // Log culling events
        if (cullingEvents.Count > 0)
        {
            _output.WriteLine($"\n  Culling Events ({cullingEvents.Count} total):");
            foreach (var evt in cullingEvents.Take(10))
            {
                _output.WriteLine($"    Gen {evt.Generation}: {evt.SpeciesBefore} → {evt.SpeciesAfter} species");
            }
            if (cullingEvents.Count > 10)
                _output.WriteLine($"    ... and {cullingEvents.Count - 10} more");
        }

        return metrics;
    }

    private RunMetrics RunBaseline(int seed, int generations)
    {
        var config = new EvolutionConfig
        {
            // Baseline: 4 species × 200 individuals (Phase 6 champion)
            SpeciesCount = 4,
            IndividualsPerSpecies = 200,
            MinSpeciesCount = 4,  // SAME as SpeciesCount -> blocks ALL culling
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,

            // Culling thresholds (unused since MinSpeciesCount = SpeciesCount)
            GraceGenerations = 3,
            StagnationThreshold = 15,
            SpeciesDiversityThreshold = 0.15f,
            RelativePerformanceThreshold = 0.5f
        };

        var evolver = new Evolver(seed);
        var random = new Random(seed);
        var topology = CreateSpiralTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        var metrics = new RunMetrics();

        for (int gen = 0; gen < generations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            var stats = population.GetStatistics();
            if (gen == 0)
                metrics.Gen0BestFitness = stats.BestFitness;

            evolver.StepGeneration(population);

            // Log every 25 generations
            if (gen % 25 == 0 || gen == generations - 1)
            {
                _output.WriteLine($"  Gen {gen,3}: Best={stats.BestFitness:F4} Mean={stats.MeanFitness:F4}");
            }
        }

        // Final fitness
        evaluator.EvaluatePopulation(population, environment, seed: generations);
        var finalStats = population.GetStatistics();
        metrics.Gen150BestFitness = finalStats.BestFitness;
        metrics.TotalSpeciesCreated = population.TotalSpeciesCreated;
        metrics.FinalSpeciesCount = population.AllSpecies.Count;
        metrics.CullingEvents = 0; // Culling disabled

        return metrics;
    }

    private SpeciesSpec CreateSpiralTopology(Random random)
    {
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)  // Use dense initialization with 30% connectivity
            .Build();
    }

    private class RunMetrics
    {
        public int TotalSpeciesCreated;
        public int CullingEvents;
        public int FinalSpeciesCount;
        public float Gen0BestFitness;
        public float Gen150BestFitness;
    }

    private class CullingEvent
    {
        public int Generation;
        public int SpeciesBefore;
        public int SpeciesAfter;
    }
}
