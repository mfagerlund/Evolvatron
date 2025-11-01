using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Hyperparameter sweep for spiral classification to find better settings.
/// Runs 16 different configurations for 10 generations each.
/// </summary>
public class SpiralHyperparameterSweepTest
{
    private readonly ITestOutputHelper _output;

    public SpiralHyperparameterSweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void SweepHyperparameters()
    {
        // Test configurations: vary key parameters
        var configurations = new[]
        {
            // Baseline (current settings)
            new Config("Baseline", species: 8, individuals: 100, elites: 2, tournamentSize: 4,
                weightJitter: 0.9f, weightReset: 0.05f, edgeAdd: 0.05f, edgeDelete: 0.02f),

            // Larger population
            new Config("Large Pop", species: 8, individuals: 200, elites: 2, tournamentSize: 4,
                weightJitter: 0.9f, weightReset: 0.05f, edgeAdd: 0.05f, edgeDelete: 0.02f),

            // Stronger tournament selection
            new Config("Strong Select", species: 8, individuals: 100, elites: 4, tournamentSize: 8,
                weightJitter: 0.9f, weightReset: 0.05f, edgeAdd: 0.05f, edgeDelete: 0.02f),

            // More elites
            new Config("More Elites", species: 8, individuals: 100, elites: 10, tournamentSize: 4,
                weightJitter: 0.9f, weightReset: 0.05f, edgeAdd: 0.05f, edgeDelete: 0.02f),

            // Higher mutation rates
            new Config("High Mutation", species: 8, individuals: 100, elites: 2, tournamentSize: 4,
                weightJitter: 0.95f, weightReset: 0.15f, edgeAdd: 0.15f, edgeDelete: 0.08f),

            // Lower mutation rates
            new Config("Low Mutation", species: 8, individuals: 100, elites: 2, tournamentSize: 4,
                weightJitter: 0.5f, weightReset: 0.01f, edgeAdd: 0.01f, edgeDelete: 0.005f),

            // Fewer species, more individuals
            new Config("Few Species", species: 4, individuals: 200, elites: 4, tournamentSize: 4,
                weightJitter: 0.9f, weightReset: 0.05f, edgeAdd: 0.05f, edgeDelete: 0.02f),

            // More species, fewer individuals
            new Config("Many Species", species: 16, individuals: 50, elites: 2, tournamentSize: 4,
                weightJitter: 0.9f, weightReset: 0.05f, edgeAdd: 0.05f, edgeDelete: 0.02f),

            // Aggressive exploration
            new Config("Aggressive", species: 8, individuals: 100, elites: 1, tournamentSize: 8,
                weightJitter: 0.95f, weightReset: 0.20f, edgeAdd: 0.20f, edgeDelete: 0.10f),

            // Conservative exploitation
            new Config("Conservative", species: 8, individuals: 100, elites: 20, tournamentSize: 2,
                weightJitter: 0.3f, weightReset: 0.01f, edgeAdd: 0.01f, edgeDelete: 0.005f),

            // Large tournament + high mutation
            new Config("LargeTournament", species: 8, individuals: 100, elites: 2, tournamentSize: 16,
                weightJitter: 0.95f, weightReset: 0.10f, edgeAdd: 0.10f, edgeDelete: 0.05f),

            // Small tournament + low mutation
            new Config("SmallTournament", species: 8, individuals: 100, elites: 2, tournamentSize: 2,
                weightJitter: 0.5f, weightReset: 0.02f, edgeAdd: 0.02f, edgeDelete: 0.01f),

            // Edge mutation focused
            new Config("Edge Focus", species: 8, individuals: 100, elites: 2, tournamentSize: 4,
                weightJitter: 0.5f, weightReset: 0.02f, edgeAdd: 0.25f, edgeDelete: 0.15f),

            // Weight mutation focused
            new Config("Weight Focus", species: 8, individuals: 100, elites: 2, tournamentSize: 4,
                weightJitter: 0.99f, weightReset: 0.20f, edgeAdd: 0.01f, edgeDelete: 0.005f),

            // Balanced extreme
            new Config("Balanced High", species: 12, individuals: 80, elites: 5, tournamentSize: 6,
                weightJitter: 0.9f, weightReset: 0.10f, edgeAdd: 0.10f, edgeDelete: 0.05f),

            // Minimal everything
            new Config("Minimal", species: 4, individuals: 50, elites: 1, tournamentSize: 2,
                weightJitter: 0.3f, weightReset: 0.01f, edgeAdd: 0.01f, edgeDelete: 0.005f),
        };

        var results = new List<SweepResult>();

        foreach (var config in configurations)
        {
            _output.WriteLine($"\n{'='*80}");
            _output.WriteLine($"Testing: {config.Name}");
            _output.WriteLine($"  Population: {config.SpeciesCount} species × {config.IndividualsPerSpecies} = {config.SpeciesCount * config.IndividualsPerSpecies}");
            _output.WriteLine($"  Selection: {config.Elites} elites, tournament size {config.TournamentSize}");
            _output.WriteLine($"  Mutations: WeightJitter={config.WeightJitter:F2}, Reset={config.WeightReset:F2}, EdgeAdd={config.EdgeAdd:F2}, EdgeDel={config.EdgeDelete:F2}");
            _output.WriteLine($"{'='*80}\n");

            var result = RunConfiguration(config);
            results.Add(result);

            _output.WriteLine($"RESULT: Gen0={result.Gen0Best:F4} Gen9={result.Gen9Best:F4} Improvement={result.Improvement:F4} ({result.ImprovementPercent:F1}%)");
            _output.WriteLine($"        FitnessRange: {result.Gen9Range:F4}, MeanImprovement: {result.MeanImprovement:F4}\n");
        }

        // Summary
        _output.WriteLine($"\n\n{'='*80}");
        _output.WriteLine("SWEEP SUMMARY - Ranked by Best Improvement");
        _output.WriteLine($"{'='*80}\n");

        var ranked = results.OrderByDescending(r => r.Improvement).ToList();

        _output.WriteLine($"{"Rank",-5} {"Configuration",-20} {"Gen0",-8} {"Gen9",-8} {"Improve",-9} {"Improve%",-9} {"Gen9Range",-10} {"MeanΔ",-8}");
        _output.WriteLine(new string('-', 80));

        for (int i = 0; i < ranked.Count; i++)
        {
            var r = ranked[i];
            _output.WriteLine($"{i + 1,-5} {r.ConfigName,-20} {r.Gen0Best,-8:F4} {r.Gen9Best,-8:F4} {r.Improvement,-9:F4} {r.ImprovementPercent,-9:F1} {r.Gen9Range,-10:F4} {r.MeanImprovement,-8:F4}");
        }

        _output.WriteLine($"\n{'='*80}");
        _output.WriteLine("KEY INSIGHTS:");
        _output.WriteLine($"{'='*80}");

        var best = ranked[0];
        var worst = ranked[^1];

        _output.WriteLine($"\nBest Configuration: {best.ConfigName}");
        _output.WriteLine($"  Improved from {best.Gen0Best:F4} to {best.Gen9Best:F4} (+{best.ImprovementPercent:F1}%)");
        _output.WriteLine($"  Final fitness range: {best.Gen9Range:F4}");

        _output.WriteLine($"\nWorst Configuration: {worst.ConfigName}");
        _output.WriteLine($"  Improved from {worst.Gen0Best:F4} to {worst.Gen9Best:F4} (+{worst.ImprovementPercent:F1}%)");

        var avgImprovement = results.Average(r => r.Improvement);
        _output.WriteLine($"\nAverage improvement across all configs: {avgImprovement:F4} ({avgImprovement / 0.0063 * 100:F0}% of baseline)");

        // Correlation analysis
        _output.WriteLine($"\nParameter Correlations with Improvement:");
        AnalyzeCorrelations(configurations, results);
    }

    private SweepResult RunConfiguration(Config config)
    {
        var topology = CreateSpiralTopology();

        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = config.SpeciesCount,
            IndividualsPerSpecies = config.IndividualsPerSpecies,
            Elites = config.Elites,
            TournamentSize = config.TournamentSize,
            MutationRates = new MutationRates
            {
                WeightJitter = config.WeightJitter,
                WeightReset = config.WeightReset
            },
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = config.EdgeAdd,
                EdgeDeleteRandom = config.EdgeDelete
            }
        };

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(evolutionConfig, topology);
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        // Generation 0
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var gen0Stats = population.GetStatistics();

        // Run 9 more generations
        for (int gen = 1; gen < 10; gen++)
        {
            evolver.StepGeneration(population);
            evaluator.EvaluatePopulation(population, environment, seed: gen);
        }

        var gen9Stats = population.GetStatistics();

        return new SweepResult
        {
            ConfigName = config.Name,
            Gen0Best = gen0Stats.BestFitness,
            Gen0Mean = gen0Stats.MeanFitness,
            Gen9Best = gen9Stats.BestFitness,
            Gen9Mean = gen9Stats.MeanFitness,
            Gen9Range = gen9Stats.BestFitness - gen9Stats.WorstFitness,
            Improvement = gen9Stats.BestFitness - gen0Stats.BestFitness,
            MeanImprovement = gen9Stats.MeanFitness - gen0Stats.MeanFitness
        };
    }

    private void AnalyzeCorrelations(Config[] configs, List<SweepResult> results)
    {
        // Simple correlation analysis
        var improvements = results.Select(r => (double)r.Improvement).ToArray();

        var popSizes = configs.Select(c => (double)(c.SpeciesCount * c.IndividualsPerSpecies)).ToArray();
        var elites = configs.Select(c => (double)c.Elites).ToArray();
        var tournaments = configs.Select(c => (double)c.TournamentSize).ToArray();
        var weightJitter = configs.Select(c => (double)c.WeightJitter).ToArray();
        var weightReset = configs.Select(c => (double)c.WeightReset).ToArray();
        var edgeAdd = configs.Select(c => (double)c.EdgeAdd).ToArray();

        _output.WriteLine($"  Population size:  {Correlation(popSizes, improvements):F3}");
        _output.WriteLine($"  Elites:           {Correlation(elites, improvements):F3}");
        _output.WriteLine($"  Tournament size:  {Correlation(tournaments, improvements):F3}");
        _output.WriteLine($"  Weight jitter:    {Correlation(weightJitter, improvements):F3}");
        _output.WriteLine($"  Weight reset:     {Correlation(weightReset, improvements):F3}");
        _output.WriteLine($"  Edge add:         {Correlation(edgeAdd, improvements):F3}");
    }

    private double Correlation(double[] x, double[] y)
    {
        var n = x.Length;
        var meanX = x.Average();
        var meanY = y.Average();

        var numerator = 0.0;
        var sumSqX = 0.0;
        var sumSqY = 0.0;

        for (int i = 0; i < n; i++)
        {
            var dx = x[i] - meanX;
            var dy = y[i] - meanY;
            numerator += dx * dy;
            sumSqX += dx * dx;
            sumSqY += dy * dy;
        }

        return numerator / Math.Sqrt(sumSqX * sumSqY);
    }

    private SpeciesSpec CreateSpiralTopology()
    {
        var random = new Random(42);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeSparse(random)
            .Build();
    }

    private record Config(
        string Name,
        int species,
        int individuals,
        int elites,
        int tournamentSize,
        float weightJitter,
        float weightReset,
        float edgeAdd,
        float edgeDelete)
    {
        public int SpeciesCount => species;
        public int IndividualsPerSpecies => individuals;
        public int Elites => elites;
        public int TournamentSize => tournamentSize;
        public float WeightJitter => weightJitter;
        public float WeightReset => weightReset;
        public float EdgeAdd => edgeAdd;
        public float EdgeDelete => edgeDelete;
    }

    private class SweepResult
    {
        public string ConfigName { get; set; } = "";
        public float Gen0Best { get; set; }
        public float Gen0Mean { get; set; }
        public float Gen9Best { get; set; }
        public float Gen9Mean { get; set; }
        public float Gen9Range { get; set; }
        public float Improvement { get; set; }
        public float MeanImprovement { get; set; }
        public float ImprovementPercent => (Improvement / Math.Abs(Gen0Best)) * 100f;
    }
}
