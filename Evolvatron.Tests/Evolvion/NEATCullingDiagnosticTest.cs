using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Diagnostic test to understand why NEAT-style OR-based culling isn't triggering.
/// </summary>
public class NEATCullingDiagnosticTest
{
    private readonly ITestOutputHelper _output;

    public NEATCullingDiagnosticTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DiagnoseNEATCulling_TrackEligibility()
    {
        const int seed = 42;
        const int generations = 30;

        var config = new EvolutionConfig
        {
            // NEAT-style configuration
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            MinSpeciesCount = 8,
            Elites = 2,
            TournamentSize = 16,

            // NEAT-style culling thresholds
            GraceGenerations = 3,
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

        for (int gen = 0; gen < generations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // BEFORE stepping generation, analyze culling eligibility
            var report = SpeciesCuller.GetCullingReport(population, config);

            int eligibleCount = report.Count(kvp => kvp.Value.IsEligible);

            _output.WriteLine($"\n=== Generation {gen} ===");
            _output.WriteLine($"Species count: {population.AllSpecies.Count}");
            _output.WriteLine($"TotalSpeciesCreated: {population.TotalSpeciesCreated}");
            _output.WriteLine($"Eligible for culling: {eligibleCount}/{population.AllSpecies.Count}");

            if (gen >= config.GraceGenerations)
            {
                // Show detailed status for all species past grace period
                _output.WriteLine("\nDetailed Status:");
                foreach (var (species, status) in report.OrderBy(kvp => kvp.Value.RelativePerformance))
                {
                    int speciesIdx = population.AllSpecies.IndexOf(species);
                    _output.WriteLine($"  Species {speciesIdx,2} (Age={species.Age,2}): " +
                        $"RelPerf={status.RelativePerformance:F3} " +
                        $"Stag={status.IsStagnant,5} " +
                        $"BelowPerf={status.BelowPerformanceThreshold,5} " +
                        $"LowDiv={status.HasLowDiversity,5} " +
                        $"-> Eligible={status.IsEligible,5}");
                }
            }

            int speciesCountBefore = population.AllSpecies.Count;
            evolver.StepGeneration(population);
            int speciesCountAfter = population.AllSpecies.Count;

            if (speciesCountAfter < speciesCountBefore)
            {
                _output.WriteLine($"\n>>> CULLING OCCURRED! {speciesCountBefore} -> {speciesCountAfter} species");
            }
            else if (eligibleCount >= 2)
            {
                _output.WriteLine($">>> WARNING: {eligibleCount} species eligible, but NO culling occurred!");
            }
        }

        _output.WriteLine($"\n\n=== SUMMARY ===");
        _output.WriteLine($"Final TotalSpeciesCreated: {population.TotalSpeciesCreated}");
        _output.WriteLine($"New species created during evolution: {population.TotalSpeciesCreated - config.SpeciesCount}");
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
}
