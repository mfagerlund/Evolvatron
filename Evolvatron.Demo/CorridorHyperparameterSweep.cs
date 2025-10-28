using System.Diagnostics;

namespace Evolvatron.Demo;

/// <summary>
/// Hyperparameter sweep for FollowTheCorridor evolution.
/// Optimizes for wall-time performance (minimize time to solve).
/// </summary>
public static class CorridorHyperparameterSweep
{
    private const int MaxTimeoutSeconds = 60; // 1 minute per trial - very tight
    private const int SeedsPerConfig = 3; // Multiple seeds for statistical validity
    private const float SolvedThreshold = 1.0f; // 100% completion - must finish!
    private const int MaxStepsForSuccess = 20; // Extremely tight generation limit

    public static void Run()
    {
        Console.WriteLine("=== FollowTheCorridor Hyperparameter Sweep ===");
        Console.WriteLine($"Max timeout: {MaxTimeoutSeconds}s per trial");
        Console.WriteLine($"Seeds per config: {SeedsPerConfig}");
        Console.WriteLine($"Solved threshold: {SolvedThreshold * 100}%");
        Console.WriteLine();

        var configs = GenerateConfigurations();
        Console.WriteLine($"Testing {configs.Count} configurations...");
        Console.WriteLine();

        var results = new List<SweepResult>();

        int configIdx = 0;
        foreach (var config in configs)
        {
            configIdx++;
            Console.Write($"[{configIdx}/{configs.Count}] {config.Description}... ");

            var result = EvaluateConfiguration(config);
            results.Add(result);

            Console.WriteLine($"{result.SolvedCount}/{SeedsPerConfig} solved, avg: {result.AvgTimeToSolve:F1}s, {result.AvgGenerations:F1} gens");
        }

        // Print summary
        PrintSummary(results);
    }

    private static List<ConfigToTest> GenerateConfigurations()
    {
        var configs = new List<ConfigToTest>();

        // Population size variations (keep baseline values for others)
        int[] speciesCounts = { 20, 40, 80 };
        int[] individualsPerSpecies = { 20, 40, 80 };

        // Selection pressure variations
        int[] eliteCounts = { 1, 2, 4 };
        float[] parentPoolPercentages = { 0.1f, 0.2f, 0.4f };
        int[] tournamentSizes = { 2, 4, 8 };

        // Generate combinations
        foreach (var speciesCount in speciesCounts)
        {
            foreach (var individualsPerSpec in individualsPerSpecies)
            {
                int totalPop = speciesCount * individualsPerSpec;

                // Skip very large populations (>3200) to keep runtime reasonable
                if (totalPop > 3200) continue;

                foreach (var elites in eliteCounts)
                {
                    foreach (var parentPool in parentPoolPercentages)
                    {
                        foreach (var tournamentSize in tournamentSizes)
                        {
                            configs.Add(new ConfigToTest
                            {
                                Description = $"S{speciesCount}xI{individualsPerSpec}={totalPop}, E={elites}, P={parentPool:F1}, T={tournamentSize}",
                                SpeciesCount = speciesCount,
                                IndividualsPerSpecies = individualsPerSpec,
                                Elites = elites,
                                ParentPoolPercentage = parentPool,
                                TournamentSize = tournamentSize
                            });
                        }
                    }
                }
            }
        }

        Console.WriteLine($"Generated {configs.Count} configurations");
        return configs;
    }

    private class TrialProgress
    {
        public int Generation;
        public float BestFitness;
        public long ElapsedMs;
        public bool Completed;
        public bool Solved;
    }

    private static SweepResult EvaluateConfiguration(ConfigToTest config)
    {
        var result = new SweepResult
        {
            Config = config,
            SolvedCount = 0,
            TimesToSolve = new List<double>(),
            Generations = new List<int>()
        };

        var results = new (bool solved, long timeMs, int generations)[SeedsPerConfig];

        // Run sequentially to avoid issues with species diversification changing population size
        for (int seed = 0; seed < SeedsPerConfig; seed++)
        {
            var progress = new TrialProgress();
            results[seed] = RunSingleTrial(config, seed, progress);
        }

        for (int seed = 0; seed < SeedsPerConfig; seed++)
        {
            var (solved, timeMs, generations) = results[seed];
            if (solved)
            {
                result.SolvedCount++;
                result.TimesToSolve.Add(timeMs / 1000.0);
                result.Generations.Add(generations);
            }
        }

        return result;
    }

    private static (bool solved, long timeMs, int generations) RunSingleTrial(
        ConfigToTest config,
        int seed,
        TrialProgress progress)
    {
        progress.Generation = -1;
        progress.ElapsedMs = 0;

        var runConfig = new CorridorEvaluationRunner.RunConfig
        {
            SpeciesCount = config.SpeciesCount,
            IndividualsPerSpecies = config.IndividualsPerSpecies,
            Elites = config.Elites,
            TournamentSize = config.TournamentSize,
            ParentPoolPercentage = config.ParentPoolPercentage,
            MinSpeciesCount = 4,
            EvolutionSeed = seed,
            MaxGenerations = MaxStepsForSuccess,
            SolvedThreshold = SolvedThreshold,
            MaxTimeoutMs = MaxTimeoutSeconds * 1000,
            MaxStepsPerEpisode = 320
        };

        var runner = new CorridorEvaluationRunner(
            config: runConfig,
            progressCallback: update =>
            {
                progress.Generation = update.Generation;
                progress.BestFitness = update.BestFitness;
                progress.ElapsedMs = update.ElapsedMs;

                if (update.Generation % 100 == 0)
                {
                    Console.WriteLine($"  Gen {update.Generation}: Best={update.BestFitness:F3} ({update.BestFitness * 100:F1}%), Time={update.ElapsedMs / 1000.0:F1}s");
                }
            }
        );

        var result = runner.Run();

        progress.Completed = true;
        progress.Solved = result.solved;
        progress.Generation = result.generation;
        progress.ElapsedMs = result.elapsedMs;

        return (result.solved, result.elapsedMs, result.generation);
    }

    private static void PrintSummary(List<SweepResult> results)
    {
        Console.WriteLine("\n=== SWEEP SUMMARY ===\n");
        Console.WriteLine("Ranked by success rate, then average time to solve:\n");

        var ranked = results
            .OrderByDescending(r => r.SolvedCount)
            .ThenBy(r => r.AvgTimeToSolve)
            .ToList();

        int rank = 1;
        foreach (var result in ranked)
        {
            Console.WriteLine($"#{rank}: {result.Config.Description}");
            Console.WriteLine($"    Solved: {result.SolvedCount}/{SeedsPerConfig} ({result.SolvedCount * 100.0 / SeedsPerConfig:F1}%)");

            if (result.SolvedCount > 0)
            {
                Console.WriteLine($"    Avg time: {result.AvgTimeToSolve:F1}s (± {result.StdDevTime:F1}s)");
                Console.WriteLine($"    Avg gens: {result.AvgGenerations:F1} (± {result.StdDevGenerations:F1})");
            }
            else
            {
                Console.WriteLine($"    No solutions found");
            }

            Console.WriteLine();
            rank++;
        }

        // Print best configuration
        var best = ranked.First();
        Console.WriteLine("=== BEST CONFIGURATION ===");
        Console.WriteLine($"{best.Config.Description}");
        Console.WriteLine($"Species: {best.Config.SpeciesCount}");
        Console.WriteLine($"Individuals per species: {best.Config.IndividualsPerSpecies}");
        Console.WriteLine($"Elites: {best.Config.Elites}");
        Console.WriteLine($"Parent pool: {best.Config.ParentPoolPercentage * 100:F0}%");
        Console.WriteLine($"Tournament size: {best.Config.TournamentSize}");
        Console.WriteLine($"Success rate: {best.SolvedCount}/{SeedsPerConfig}");
        if (best.SolvedCount > 0)
        {
            Console.WriteLine($"Average time to solve: {best.AvgTimeToSolve:F1}s");
            Console.WriteLine($"Average generations: {best.AvgGenerations:F1}");
        }
    }

    private class ConfigToTest
    {
        public string Description { get; set; } = "";
        public int SpeciesCount { get; set; }
        public int IndividualsPerSpecies { get; set; }
        public int Elites { get; set; }
        public float ParentPoolPercentage { get; set; }
        public int TournamentSize { get; set; }
    }

    private class SweepResult
    {
        public ConfigToTest Config { get; set; } = new();
        public int SolvedCount { get; set; }
        public List<double> TimesToSolve { get; set; } = new();
        public List<int> Generations { get; set; } = new();

        public double AvgTimeToSolve => TimesToSolve.Count > 0 ? TimesToSolve.Average() : double.MaxValue;
        public double StdDevTime => TimesToSolve.Count > 1
            ? Math.Sqrt(TimesToSolve.Select(x => Math.Pow(x - AvgTimeToSolve, 2)).Average())
            : 0;

        public double AvgGenerations => Generations.Count > 0 ? Generations.Average() : 0;
        public double StdDevGenerations => Generations.Count > 1
            ? Math.Sqrt(Generations.Select(x => Math.Pow(x - AvgGenerations, 2)).Average())
            : 0;
    }
}
