using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Hyperparameter sweep to find optimal evolution settings for XOR.
/// Tests various population sizes, mutation rates, and adaptive schedules.
/// </summary>
public class HyperparameterSweepTest
{
    private readonly ITestOutputHelper _output;

    public HyperparameterSweepTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact(Skip = "Slow test - run explicitly")]
    public void HyperparameterSweep_FindOptimalSettings()
    {
        var results = new List<SweepResult>();

        // Population size sweep
        var popSizes = new[] { 20, 40, 60, 100 };

        // Mutation rate sweeps
        var weightJitterRates = new[] { 0.8f, 0.9f, 0.95f, 0.99f };
        var weightJitterStdDevs = new[] { 0.1f, 0.2f, 0.3f, 0.5f };
        var weightResetRates = new[] { 0.05f, 0.1f, 0.15f, 0.2f };

        _output.WriteLine("=== HYPERPARAMETER SWEEP FOR XOR ===\n");
        _output.WriteLine("Testing combinations of:");
        _output.WriteLine($"  Population sizes: {string.Join(", ", popSizes)}");
        _output.WriteLine($"  WeightJitter rates: {string.Join(", ", weightJitterRates)}");
        _output.WriteLine($"  WeightJitterStdDev: {string.Join(", ", weightJitterStdDevs)}");
        _output.WriteLine($"  WeightReset rates: {string.Join(", ", weightResetRates)}");
        _output.WriteLine("");

        int runCount = 0;
        int totalRuns = popSizes.Length * weightJitterRates.Length * weightJitterStdDevs.Length * weightResetRates.Length;

        // Grid search over key hyperparameters
        foreach (var popSize in popSizes)
        {
            foreach (var jitterRate in weightJitterRates)
            {
                foreach (var jitterStdDev in weightJitterStdDevs)
                {
                    foreach (var resetRate in weightResetRates)
                    {
                        runCount++;

                        var config = new EvolutionConfig
                        {
                            SpeciesCount = 4,
                            IndividualsPerSpecies = popSize,
                            Elites = 2,
                            TournamentSize = 3,
                            MutationRates = new MutationRates
                            {
                                WeightJitter = jitterRate,
                                WeightJitterStdDev = jitterStdDev,
                                WeightReset = resetRate,
                                ActivationSwap = 0.05f,
                                WeightL1Shrink = 0.05f
                            }
                        };

                        // Run 3 trials per config for statistical reliability
                        var trials = new int[3];
                        for (int trial = 0; trial < 3; trial++)
                        {
                            trials[trial] = RunXOREvolution(config, seed: 42 + trial);
                        }

                        float meanGens = (float)trials.Average();
                        float stdGens = MathF.Sqrt((float)trials.Select(x => (x - meanGens) * (x - meanGens)).Average());

                        results.Add(new SweepResult
                        {
                            PopulationSize = popSize * 4, // 4 species
                            WeightJitter = jitterRate,
                            WeightJitterStdDev = jitterStdDev,
                            WeightReset = resetRate,
                            MeanGenerations = meanGens,
                            StdGenerations = stdGens,
                            Trials = trials
                        });

                        if (runCount % 20 == 0)
                        {
                            _output.WriteLine($"Progress: {runCount}/{totalRuns} runs complete...");
                        }
                    }
                }
            }
        }

        // Analyze results
        _output.WriteLine("\n=== TOP 10 CONFIGURATIONS ===\n");
        var top10 = results.OrderBy(r => r.MeanGenerations).Take(10).ToList();

        for (int i = 0; i < top10.Count; i++)
        {
            var r = top10[i];
            _output.WriteLine($"#{i + 1}: {r.MeanGenerations:F1} ± {r.StdGenerations:F1} generations");
            _output.WriteLine($"     PopSize={r.PopulationSize}, Jitter={r.WeightJitter:F2}, JitterStd={r.WeightJitterStdDev:F2}, Reset={r.WeightReset:F2}");
            _output.WriteLine($"     Trials: [{string.Join(", ", r.Trials)}]");
            _output.WriteLine("");
        }

        // Statistical analysis
        _output.WriteLine("\n=== FACTOR ANALYSIS ===\n");
        AnalyzeFactorImpact(results, "PopulationSize", r => r.PopulationSize);
        AnalyzeFactorImpact(results, "WeightJitter", r => r.WeightJitter);
        AnalyzeFactorImpact(results, "WeightJitterStdDev", r => r.WeightJitterStdDev);
        AnalyzeFactorImpact(results, "WeightReset", r => r.WeightReset);

        // Best config should converge in < 15 generations
        var best = results.OrderBy(r => r.MeanGenerations).First();
        Assert.True(best.MeanGenerations < 15f,
            $"Best config should converge in < 15 generations, got {best.MeanGenerations:F1}");
    }

    [Fact]
    public void AdaptiveMutationSchedule_ConvergesFaster()
    {
        _output.WriteLine("=== ADAPTIVE MUTATION SCHEDULE TEST ===\n");

        // Baseline: constant mutation rates
        var baselineConfig = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 50,
            Elites = 2,
            TournamentSize = 3,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.95f,
                WeightJitterStdDev = 0.3f,
                WeightReset = 0.1f,
                ActivationSwap = 0.05f
            }
        };

        var baselineTrials = new int[5];
        for (int i = 0; i < 5; i++)
        {
            baselineTrials[i] = RunXOREvolution(baselineConfig, seed: 100 + i);
        }
        float baselineMean = (float)baselineTrials.Average();

        _output.WriteLine($"Baseline (constant rates): {baselineMean:F1} ± {CalculateStdDev(baselineTrials):F1} generations");
        _output.WriteLine($"  Trials: [{string.Join(", ", baselineTrials)}]");
        _output.WriteLine("");

        // Adaptive: decay mutation rates over time
        var adaptiveTrials = new int[5];
        for (int i = 0; i < 5; i++)
        {
            adaptiveTrials[i] = RunXOREvolutionWithAdaptiveSchedule(seed: 100 + i);
        }
        float adaptiveMean = (float)adaptiveTrials.Average();

        _output.WriteLine($"Adaptive schedule: {adaptiveMean:F1} ± {CalculateStdDev(adaptiveTrials):F1} generations");
        _output.WriteLine($"  Trials: [{string.Join(", ", adaptiveTrials)}]");
        _output.WriteLine("");

        _output.WriteLine($"Improvement: {baselineMean - adaptiveMean:F1} generations ({100 * (baselineMean - adaptiveMean) / baselineMean:F1}% faster)");
    }

    /// <summary>
    /// Run XOR evolution with constant mutation rates.
    /// Returns number of generations to solve.
    /// </summary>
    private int RunXOREvolution(EvolutionConfig config, int seed)
    {
        var topology = CreateXORTopology();
        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);
        var environment = new XOREnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        float successThreshold = -0.01f;
        int maxGenerations = 100;

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            if (bestFitness >= successThreshold)
                return gen;

            evolver.StepGeneration(population);
        }

        return maxGenerations; // Failed to converge
    }

    /// <summary>
    /// Run XOR evolution with adaptive mutation schedule.
    /// Exploration -> exploitation transition.
    /// </summary>
    private int RunXOREvolutionWithAdaptiveSchedule(int seed)
    {
        var topology = CreateXORTopology();
        var evolver = new Evolver(seed: seed);

        // Start with high exploration
        var config = new EvolutionConfig
        {
            SpeciesCount = 4,
            IndividualsPerSpecies = 50,
            Elites = 2,
            TournamentSize = 3,
            MutationRates = new MutationRates
            {
                WeightJitter = 0.99f,      // Very high initial exploration
                WeightJitterStdDev = 0.5f, // Large initial mutations
                WeightReset = 0.15f,
                ActivationSwap = 0.1f
            }
        };

        var population = evolver.InitializePopulation(config, topology);
        var environment = new XOREnvironment();
        var evaluator = new SimpleFitnessEvaluator();

        float successThreshold = -0.01f;
        int maxGenerations = 100;

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            if (bestFitness >= successThreshold)
                return gen;

            // Adaptive schedule: decay mutation rates over time
            // Transition from exploration (large mutations) to exploitation (fine-tuning)
            float progress = gen / 50f; // Assume ~50 generation horizon
            float explorationFactor = MathF.Max(0.1f, 1f - progress); // 1.0 -> 0.1

            config.MutationRates.WeightJitterStdDev = 0.5f * explorationFactor + 0.05f * (1f - explorationFactor);
            config.MutationRates.WeightReset = 0.15f * explorationFactor + 0.02f * (1f - explorationFactor);
            config.MutationRates.ActivationSwap = 0.1f * explorationFactor + 0.01f * (1f - explorationFactor);

            evolver.StepGeneration(population);
        }

        return maxGenerations;
    }

    private void AnalyzeFactorImpact<T>(List<SweepResult> results, string factorName, Func<SweepResult, T> selector)
    {
        var grouped = results.GroupBy(selector);
        var analysis = grouped.Select(g => new
        {
            Value = g.Key,
            MeanGens = (float)g.Average(r => r.MeanGenerations),
            Count = g.Count()
        }).OrderBy(x => x.MeanGens).ToList();

        _output.WriteLine($"{factorName}:");
        foreach (var item in analysis)
        {
            _output.WriteLine($"  {item.Value,-12} -> {item.MeanGens,5:F1} gens (n={item.Count})");
        }
        _output.WriteLine("");
    }

    private float CalculateStdDev(int[] values)
    {
        float mean = (float)values.Average();
        return MathF.Sqrt((float)values.Select(x => (x - mean) * (x - mean)).Average());
    }

    private SpeciesSpec CreateXORTopology()
    {
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(4, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(1, ActivationType.Tanh)
            .FullyConnect(fromRow: 0, toRow: 1)
            .FullyConnect(fromRow: 1, toRow: 2)
            .WithMaxInDegree(8)
            .Build();
    }

    private class SweepResult
    {
        public int PopulationSize { get; set; }
        public float WeightJitter { get; set; }
        public float WeightJitterStdDev { get; set; }
        public float WeightReset { get; set; }
        public float MeanGenerations { get; set; }
        public float StdGenerations { get; set; }
        public int[] Trials { get; set; } = Array.Empty<int>();
    }
}
