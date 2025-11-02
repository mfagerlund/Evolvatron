using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Diagnostic to figure out why eligible.Count < 2 blocks culling.
/// Prints detailed eligibility breakdown for first 20 generations.
/// </summary>
public class CullingEligibilityDiagnostic
{
    private readonly ITestOutputHelper _output;

    public CullingEligibilityDiagnostic(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DiagnoseEligibilityFailure()
    {
        var config = new EvolutionConfig
        {
            // NEAT-style configuration
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            MinSpeciesCount = 8,
            Elites = 2,
            TournamentSize = 16,
            ParentPoolPercentage = 1.0f,

            // NEAT-style culling thresholds
            GraceGenerations = 0,  // No grace period - immediate eligibility for testing
            StagnationThreshold = 6,
            SpeciesDiversityThreshold = 0.08f,
            RelativePerformanceThreshold = 0.7f
        };

        var evolver = new Evolver(seed: 42);
        var random = new Random(42);
        var topology = CreateSpiralTopology(random);
        var population = evolver.InitializePopulation(config, topology);

        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        _output.WriteLine("=== CULLING ELIGIBILITY DIAGNOSTIC ===");
        _output.WriteLine($"Configuration: MinSpeciesCount={config.MinSpeciesCount}, SpeciesCount={config.SpeciesCount}");
        _output.WriteLine($"Grace={config.GraceGenerations}, Stagnation={config.StagnationThreshold}, " +
            $"Diversity={config.SpeciesDiversityThreshold}, Performance={config.RelativePerformanceThreshold}");
        _output.WriteLine("");

        for (int gen = 0; gen < 20; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            StagnationTracker.UpdateAllSpecies(population);

            _output.WriteLine($"\n=== Generation {gen} ===");
            _output.WriteLine($"Species count: {population.AllSpecies.Count}");

            // Check gate 1: population-level check
            bool passesGate1 = population.AllSpecies.Count > config.MinSpeciesCount;
            _output.WriteLine($"Gate 1 (AllSpecies.Count > MinSpeciesCount): {passesGate1} ({population.AllSpecies.Count} > {config.MinSpeciesCount})");

            if (!passesGate1)
            {
                _output.WriteLine("  ❌ BLOCKED at Gate 1 (Evolver.cs:39)");
                evolver.StepGeneration(population);
                continue;
            }

            // Find best species
            Species? bestSpecies = null;
            float bestFitness = float.MinValue;
            foreach (var species in population.AllSpecies)
            {
                foreach (var individual in species.Individuals)
                {
                    if (individual.Fitness > bestFitness)
                    {
                        bestFitness = individual.Fitness;
                        bestSpecies = species;
                    }
                }
            }

            // Find eligible species
            var eligible = SpeciesCuller.FindEligibleForCulling(population, config);
            _output.WriteLine($"Eligible species (before removing best): {eligible.Count}");

            // Remove best
            if (bestSpecies != null)
                eligible.Remove(bestSpecies);

            _output.WriteLine($"Eligible species (after removing best): {eligible.Count}");

            // Check gate 2: eligible count
            bool passesGate2 = eligible.Count >= 2;
            _output.WriteLine($"Gate 2 (eligible.Count >= 2): {passesGate2} ({eligible.Count} >= 2)");

            if (!passesGate2)
            {
                _output.WriteLine("  ❌ BLOCKED at Gate 2 (SpeciesCuller.cs:55)");
            }
            else
            {
                _output.WriteLine("  ✅ PASSES ALL GATES - culling should occur!");
            }

            // Print eligibility breakdown for each species
            var report = SpeciesCuller.GetCullingReport(population, config);
            int idx = 0;
            foreach (var (species, status) in report)
            {
                string isBest = (species == bestSpecies) ? " [BEST]" : "";
                string isEligible = status.IsEligible ? " [ELIGIBLE]" : "";

                _output.WriteLine($"  Species {idx}{isBest}{isEligible}:");
                _output.WriteLine($"    Age={species.Age} PastGrace={status.PastGracePeriod}");
                _output.WriteLine($"    Stagnant={status.IsStagnant} LowDiversity={status.HasLowDiversity} BelowPerf={status.BelowPerformanceThreshold}");
                _output.WriteLine($"    RelativePerf={status.RelativePerformance:F4} (threshold={config.RelativePerformanceThreshold})");
                _output.WriteLine($"    BestFitness={species.Stats.BestFitnessEver:F4} Variance={status.FitnessVariance:F6}");

                idx++;
            }

            evolver.StepGeneration(population);
        }
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
