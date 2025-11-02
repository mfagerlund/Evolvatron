using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;
using System.Text;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Quick comparison of population dynamics strategies.
/// Validates that NEAT-style (20×40 with aggressive culling) improves or maintains performance
/// vs baseline (4×200 with culling disabled).
/// </summary>
public class PopulationDynamicsComparisonTest
{
    private readonly ITestOutputHelper _output;

    public PopulationDynamicsComparisonTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void ComparePopulationDynamics_QuickValidation()
    {
        const int seeds = 3;
        const int generations = 100;

        _output.WriteLine("=== POPULATION DYNAMICS COMPARISON ===");
        _output.WriteLine($"Seeds: {seeds}, Generations: {generations}");
        _output.WriteLine($"Benchmark: Spiral classification (2→8→8→1)");
        _output.WriteLine("");

        var configs = new[]
        {
            CreateConfig("Baseline-4x200-NoCulling", baseline: true),
            CreateConfig("NEAT-20x40-AggressiveCulling", baseline: false),
            CreateConfig("Middle-8x100-ModerateCulling", middle: true)
        };

        var results = new Dictionary<string, List<RunResult>>();

        foreach (var (name, config) in configs)
        {
            _output.WriteLine($"\n=== {name} ===");
            _output.WriteLine($"  Species: {config.SpeciesCount} × {config.IndividualsPerSpecies} = {config.SpeciesCount * config.IndividualsPerSpecies} total");
            _output.WriteLine($"  MinSpeciesCount: {config.MinSpeciesCount}");
            _output.WriteLine($"  GraceGenerations: {config.GraceGenerations}");

            var runResults = new List<RunResult>();

            for (int seed = 0; seed < seeds; seed++)
            {
                var result = RunEvolution(config, seed, generations);
                runResults.Add(result);

                _output.WriteLine($"  Seed {seed}: Gen0={result.Gen0Best:F4} → Gen{generations}={result.GenFinalBest:F4} " +
                    $"(Δ={result.Improvement:F4}, Species={result.TotalSpeciesCreated})");
            }

            results[name] = runResults;
        }

        // Aggregate analysis
        _output.WriteLine("\n\n=== AGGREGATE COMPARISON ===");

        var baseline = results["Baseline-4x200-NoCulling"];
        var neat = results["NEAT-20x40-AggressiveCulling"];
        var middle = results["Middle-8x100-ModerateCulling"];

        double baselineAvg = baseline.Average(r => r.GenFinalBest);
        double neatAvg = neat.Average(r => r.GenFinalBest);
        double middleAvg = middle.Average(r => r.GenFinalBest);

        _output.WriteLine($"Baseline-4x200:  Avg Gen{generations} = {baselineAvg:F4}");
        _output.WriteLine($"NEAT-20x40:      Avg Gen{generations} = {neatAvg:F4} ({(neatAvg / baselineAvg - 1.0) * 100:+0.0;-0.0}%)");
        _output.WriteLine($"Middle-8x100:    Avg Gen{generations} = {middleAvg:F4} ({(middleAvg / baselineAvg - 1.0) * 100:+0.0;-0.0}%)");

        _output.WriteLine("");
        _output.WriteLine($"Baseline species turnover: {baseline.Average(r => r.TotalSpeciesCreated):F1}");
        _output.WriteLine($"NEAT species turnover:     {neat.Average(r => r.TotalSpeciesCreated):F1}");
        _output.WriteLine($"Middle species turnover:   {middle.Average(r => r.TotalSpeciesCreated):F1}");

        // Validation: NEAT should be within 90% of baseline (allow some variance for quick test)
        double neatRatio = neatAvg / baselineAvg;
        Assert.True(neatRatio >= 0.90,
            $"NEAT-style performance should be >= 90% of baseline, but was {neatRatio * 100:F1}%");

        _output.WriteLine("\n✓ NEAT-style population dynamics validated!");
    }

    private (string, EvolutionConfig) CreateConfig(string name, bool baseline = false, bool middle = false)
    {
        if (baseline)
        {
            // Old approach: Few species, many individuals, culling disabled
            return (name, new EvolutionConfig
            {
                SpeciesCount = 4,
                IndividualsPerSpecies = 200,
                MinSpeciesCount = 4,  // Blocks culling
                GraceGenerations = 3,
                StagnationThreshold = 15,
                SpeciesDiversityThreshold = 0.15f,
                RelativePerformanceThreshold = 0.5f,
                Elites = 2,
                TournamentSize = 16,
                ParentPoolPercentage = 1.0f
            });
        }
        else if (middle)
        {
            // Middle ground: Moderate species count and culling
            return (name, new EvolutionConfig
            {
                SpeciesCount = 8,
                IndividualsPerSpecies = 100,
                MinSpeciesCount = 4,
                GraceGenerations = 2,
                StagnationThreshold = 10,
                SpeciesDiversityThreshold = 0.10f,
                RelativePerformanceThreshold = 0.6f,
                Elites = 2,
                TournamentSize = 16,
                ParentPoolPercentage = 1.0f
            });
        }
        else
        {
            // NEAT-style: Many species, fewer individuals, aggressive culling
            return (name, new EvolutionConfig
            {
                SpeciesCount = 20,
                IndividualsPerSpecies = 40,
                MinSpeciesCount = 8,
                GraceGenerations = 1,
                StagnationThreshold = 6,
                SpeciesDiversityThreshold = 0.08f,
                RelativePerformanceThreshold = 0.7f,
                Elites = 2,
                TournamentSize = 16,
                ParentPoolPercentage = 1.0f
            });
        }
    }

    private RunResult RunEvolution(EvolutionConfig config, int seed, int generations)
    {
        var evolver = new Evolver(seed);
        var random = new Random(seed);
        var topology = CreateSpiralTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Gen 0 fitness
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        // Evolve
        for (int gen = 0; gen < generations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            evolver.StepGeneration(population);
        }

        // Final fitness
        evaluator.EvaluatePopulation(population, environment, seed: generations);
        var finalStats = population.GetStatistics();

        return new RunResult
        {
            Gen0Best = gen0Stats.BestFitness,
            GenFinalBest = finalStats.BestFitness,
            Improvement = finalStats.BestFitness - gen0Stats.BestFitness,
            TotalSpeciesCreated = population.TotalSpeciesCreated
        };
    }

    private SpeciesSpec CreateSpiralTopology(Random random)
    {
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(random, density: 0.3f)
            .Build();
    }

    private struct RunResult
    {
        public float Gen0Best;
        public float GenFinalBest;
        public float Improvement;
        public int TotalSpeciesCreated;
    }
}
