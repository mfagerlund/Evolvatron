using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Diagnostic test to identify why species culling NEVER triggers.
///
/// Hypothesis: The 4-condition AND logic is too restrictive:
/// 1. Age > GraceGenerations
/// 2. GenerationsSinceImprovement >= StagnationThreshold
/// 3. BestFitnessEver < RelativePerformanceThreshold * globalBest
/// 4. FitnessVariance < SpeciesDiversityThreshold
///
/// ALL must be true -> culling almost never happens.
/// </summary>
public class CullingDiagnosticTest
{
    private readonly ITestOutputHelper _output;

    public CullingDiagnosticTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DiagnoseCullingFailure_TrackAllConditions()
    {
        const int seed = 42;
        const int generations = 50;

        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 200,
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,
            MinSpeciesCount = 2,  // Allow culling down to 2

            // EXTREMELY AGGRESSIVE culling settings
            GraceGenerations = 1,  // Only 1 generation grace
            StagnationThreshold = 2,  // Stagnant after 2 gens
            SpeciesDiversityThreshold = 100.0f,  // Effectively disabled (always low diversity)
            RelativePerformanceThreshold = 10.0f  // Effectively disabled (always below threshold)
        };

        var evolver = new Evolver(seed);
        var random = new Random(seed);
        var topology = CreateXORTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new XOREnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        int totalCullingChecks = 0;
        int totalCullingEvents = 0;

        for (int gen = 0; gen < generations; gen++)
        {
            // Evaluate fitness
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // BEFORE stepping generation, log culling status
            var report = SpeciesCuller.GetCullingReport(population, config);

            int eligibleCount = report.Count(kvp => kvp.Value.IsEligible);

            _output.WriteLine($"\n=== Generation {gen} ===");
            _output.WriteLine($"Species count: {population.AllSpecies.Count}");
            _output.WriteLine($"Eligible for culling: {eligibleCount}/{population.AllSpecies.Count}");

            foreach (var (species, status) in report)
            {
                int speciesIdx = population.AllSpecies.IndexOf(species);
                _output.WriteLine($"\nSpecies {speciesIdx} (Age={species.Age}):");
                _output.WriteLine($"  BestFitnessEver: {species.Stats.BestFitnessEver:F4}");
                _output.WriteLine($"  GenerationsSinceImprovement: {species.Stats.GenerationsSinceImprovement}");
                _output.WriteLine($"  FitnessVariance: {species.Stats.FitnessVariance:F4}");
                _output.WriteLine($"  PastGracePeriod: {status.PastGracePeriod} (Age {species.Age} > Grace {config.GraceGenerations})");
                _output.WriteLine($"  IsStagnant: {status.IsStagnant} (GensSinceImprove {species.Stats.GenerationsSinceImprovement} >= Threshold {config.StagnationThreshold})");
                _output.WriteLine($"  BelowPerformanceThreshold: {status.BelowPerformanceThreshold} (RelPerf {status.RelativePerformance:F4} < Threshold {config.RelativePerformanceThreshold})");
                _output.WriteLine($"  HasLowDiversity: {status.HasLowDiversity} (Variance {species.Stats.FitnessVariance:F4} < Threshold {config.SpeciesDiversityThreshold})");
                _output.WriteLine($"  >>> ELIGIBLE: {status.IsEligible}");
            }

            totalCullingChecks++;

            // Step generation (includes culling)
            int speciesCountBefore = population.AllSpecies.Count;
            evolver.StepGeneration(population);
            int speciesCountAfter = population.AllSpecies.Count;

            if (speciesCountAfter < speciesCountBefore)
            {
                totalCullingEvents++;
                _output.WriteLine($"\n>>> CULLING OCCURRED! {speciesCountBefore} -> {speciesCountAfter} species");
            }
        }

        _output.WriteLine($"\n\n=== SUMMARY ===");
        _output.WriteLine($"Total generations: {generations}");
        _output.WriteLine($"Total culling events: {totalCullingEvents}");
        _output.WriteLine($"Culling rate: {totalCullingEvents * 100.0 / totalCullingChecks:F1}%");

        // With EXTREMELY aggressive settings, we should see culling
        if (totalCullingEvents == 0)
        {
            _output.WriteLine("\n!!! CULLING NEVER TRIGGERED !!!");
            _output.WriteLine("This proves the bug exists.");
        }
    }

    [Fact]
    public void IdentifyWhichConditionBlocksCulling()
    {
        const int seed = 42;
        const int generations = 20;

        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 200,
            Elites = 2,
            TournamentSize = 16,
            MinSpeciesCount = 2,

            // ULTRA AGGRESSIVE
            GraceGenerations = 0,  // No grace period
            StagnationThreshold = 2,
            SpeciesDiversityThreshold = 1000.0f,  // Disabled
            RelativePerformanceThreshold = 1000.0f  // Disabled
        };

        var evolver = new Evolver(seed);
        var random = new Random(seed);
        var topology = CreateXORTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new XOREnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        var failureReasons = new Dictionary<string, int>
        {
            ["PastGracePeriod_FAIL"] = 0,
            ["IsStagnant_FAIL"] = 0,
            ["BelowPerformanceThreshold_FAIL"] = 0,
            ["HasLowDiversity_FAIL"] = 0,
            ["ALL_PASS"] = 0
        };

        for (int gen = 0; gen < generations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            var report = SpeciesCuller.GetCullingReport(population, config);

            foreach (var (species, status) in report)
            {
                // Track which condition fails first
                if (!status.PastGracePeriod)
                    failureReasons["PastGracePeriod_FAIL"]++;
                else if (!status.IsStagnant)
                    failureReasons["IsStagnant_FAIL"]++;
                else if (!status.BelowPerformanceThreshold)
                    failureReasons["BelowPerformanceThreshold_FAIL"]++;
                else if (!status.HasLowDiversity)
                    failureReasons["HasLowDiversity_FAIL"]++;
                else
                    failureReasons["ALL_PASS"]++;
            }

            evolver.StepGeneration(population);
        }

        _output.WriteLine("=== CONDITION FAILURE ANALYSIS ===");
        _output.WriteLine($"Total species-generations checked: {failureReasons.Values.Sum()}");
        _output.WriteLine("");

        foreach (var (reason, count) in failureReasons.OrderByDescending(kvp => kvp.Value))
        {
            double pct = count * 100.0 / failureReasons.Values.Sum();
            _output.WriteLine($"{reason,-35}: {count,4} ({pct:F1}%)");
        }

        // If ALL_PASS is 0, culling logic is broken
        Assert.True(failureReasons["ALL_PASS"] > 0,
            "Even with disabled thresholds, NO species passed all conditions!");
    }

    [Fact]
    public void TestMinSpeciesCountBlocking()
    {
        const int seed = 42;

        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 100,
            MinSpeciesCount = 4,  // SAME as SpeciesCount - blocks ALL culling

            GraceGenerations = 0,
            StagnationThreshold = 1,
            SpeciesDiversityThreshold = 1000.0f,
            RelativePerformanceThreshold = 1000.0f
        };

        var evolver = new Evolver(seed);
        var random = new Random(seed);
        var topology = CreateXORTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new XOREnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        int cullingEvents = 0;

        for (int gen = 0; gen < 20; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            int before = population.AllSpecies.Count;
            evolver.StepGeneration(population);
            int after = population.AllSpecies.Count;

            if (after < before)
                cullingEvents++;
        }

        _output.WriteLine($"Culling events with MinSpeciesCount={config.MinSpeciesCount}: {cullingEvents}");
        Assert.Equal(0, cullingEvents); // Should be blocked by MinSpeciesCount check
    }

    private SpeciesSpec CreateXORTopology(Random random)
    {
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(4, ActivationType.Tanh, ActivationType.ReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(8)
            .InitializeSparse(random)
            .Build();
    }
}
