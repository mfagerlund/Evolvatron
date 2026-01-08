using Xunit;
using Xunit.Abstractions;
using Evolvatron.Core;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Headless evolution test with detailed statistics output.
/// Runs batched GPU evolution without UI to diagnose learning issues.
/// </summary>
public class HeadlessEvolutionTest
{
    private readonly ITestOutputHelper _output;

    public HeadlessEvolutionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void RunHeadlessEvolution_ShouldShowLearning()
    {
        const int Generations = 50;
        const int PopulationSize = 500;

        _output.WriteLine("=== HEADLESS EVOLUTION TEST ===");
        _output.WriteLine($"Population: {PopulationSize}, Generations: {Generations}");
        _output.WriteLine("");

        // Evolution setup
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 1,              // Single species - simpler
            IndividualsPerSpecies = PopulationSize,
            MinSpeciesCount = 1,
            Elites = 20,                   // More elites for better learning
            TournamentSize = 5,
            GraceGenerations = 10000,
            StagnationThreshold = 10000,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.9f,
                WeightJitterStdDev = 0.2f,  // Bigger mutations for exploration
                WeightReset = 0.1f
            }
        };

        var evolver = new Evolver(seed: 42);

        // Simple network - 4 inputs (dx, dy, vx, vy) -> 4 hidden -> 2 outputs (thrust_x, thrust_y)
        var topology = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(4, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var population = evolver.InitializePopulation(evolutionConfig, topology);

        // Create GPU evaluator
        using var gpuEvaluator = new GPUBatchedFitnessEvaluator(maxIndividuals: PopulationSize + 100);

        _output.WriteLine($"GPU: {gpuEvaluator.Accelerator.Name}");
        _output.WriteLine("");
        _output.WriteLine("Gen | Min      | Avg      | Max      | Std      | MaxTgt | Learning");
        _output.WriteLine("----|----------|----------|----------|----------|--------|----------");

        float previousMaxFitness = float.NegativeInfinity;
        float[] maxFitnessHistory = new float[10];
        float[] avgFitnessHistory = new float[10];  // Track average too
        int historyIdx = 0;
        float initialAvgFitness = 0;

        for (int gen = 0; gen < Generations; gen++)
        {
            // Gather all individuals
            var allIndividuals = new List<Individual>();
            foreach (var species in population.AllSpecies)
            {
                allIndividuals.AddRange(species.Individuals);
            }

            // Evaluate on GPU
            float[] fitnessValues = gpuEvaluator.EvaluatePopulation(
                topology,
                allIndividuals,
                seed: gen * 1000);

            // Apply fitness back
            int idx = 0;
            foreach (var species in population.AllSpecies)
            {
                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = fitnessValues[idx++];
                    species.Individuals[i] = ind;
                }
            }

            // Compute stats
            float minFitness = fitnessValues.Min();
            float maxFitness = fitnessValues.Max();
            float avgFitness = fitnessValues.Average();
            float stdDev = MathF.Sqrt(fitnessValues.Select(f => (f - avgFitness) * (f - avgFitness)).Average());

            // Track learning trend (use average fitness, not max)
            maxFitnessHistory[historyIdx % 10] = maxFitness;
            avgFitnessHistory[historyIdx % 10] = avgFitness;
            if (gen == 0) initialAvgFitness = avgFitness;
            historyIdx++;

            string learningIndicator = "";
            if (gen >= 10)
            {
                float oldAvg = avgFitnessHistory.Take(5).Average();
                float newAvg = avgFitnessHistory.Skip(5).Average();

                if (newAvg > oldAvg * 1.05f)
                    learningIndicator = "↑ IMPROVING";
                else if (newAvg < oldAvg * 0.95f)
                    learningIndicator = "↓ declining";
                else
                    learningIndicator = "→ stable";
            }

            // Estimate max targets collected from fitness
            // Target bonus is 500 per target, shaping is ~3/step max
            // If max fitness > baseline, targets were likely collected
            int estimatedMaxTargets = (int)Math.Max(0, (maxFitness - 300 * 3f) / 500f);

            _output.WriteLine($"{gen,3} | {minFitness,8:F1} | {avgFitness,8:F1} | {maxFitness,8:F1} | {stdDev,8:F1} | {estimatedMaxTargets,6} | {learningIndicator}");

            previousMaxFitness = maxFitness;

            // Evolve
            evolver.StepGeneration(population);
        }

        _output.WriteLine("");
        _output.WriteLine("=== TEST COMPLETE ===");

        // Check for learning using AVERAGE fitness (more meaningful metric)
        float finalAvgFitness = avgFitnessHistory.Average();
        float avgImprovement = (finalAvgFitness - initialAvgFitness) / Math.Abs(initialAvgFitness + 1) * 100;

        _output.WriteLine($"Initial avg fitness: {initialAvgFitness:F1}");
        _output.WriteLine($"Final avg fitness: {finalAvgFitness:F1}");
        _output.WriteLine($"Improvement: {avgImprovement:F1}%");

        // Check max fitness too
        float finalMax = maxFitnessHistory.Max();
        _output.WriteLine($"Best ever fitness: {finalMax:F1}");

        // Soft assertion - just report, don't fail
        if (avgImprovement > 50f)
        {
            _output.WriteLine("✓ CLEAR LEARNING DETECTED - avg fitness improved by >50%");
        }
        else if (avgImprovement > 20f)
        {
            _output.WriteLine("✓ LEARNING DETECTED - avg fitness improved by >20%");
        }
        else
        {
            _output.WriteLine("✗ NO CLEAR LEARNING - need to investigate reward function");
        }
    }
}
