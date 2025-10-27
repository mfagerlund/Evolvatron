using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using System.Diagnostics;
using System.Text;
using Colonel.Tests.HagridTests.FollowTheCorridor;
using static Colonel.Tests.HagridTests.FollowTheCorridor.SimpleCarWorld;

namespace Evolvatron.Demo;

/// <summary>
/// Hyperparameter sweep for FollowTheCorridor evolution.
/// Optimizes for wall-time performance (minimize time to solve).
/// </summary>
public static class CorridorHyperparameterSweep
{
    private const int MaxTimeoutSeconds = 120;
    private const int SeedsPerConfig = 3;
    private const float SolvedThreshold = 1.0f; // 100% completion
    private const int MaxStepsForSuccess = 200; // Must complete in 200 steps (not full 320)

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

        // Baseline population configurations
        configs.Add(new ConfigToTest
        {
            Description = "Small Pop (10x40=400, E=1, P=0.2, T=4)",
            SpeciesCount = 10,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "Medium Pop (20x40=800, E=1, P=0.2, T=4)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "Large Pop Square (40x40=1600, E=1, P=0.2, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "Large Pop Wide (80x20=1600, E=1, P=0.2, T=4)",
            SpeciesCount = 80,
            IndividualsPerSpecies = 20,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        // Mutation rate variations (using Medium Pop as baseline)
        configs.Add(new ConfigToTest
        {
            Description = "Med Pop + High Jitter (WJ=0.99, SD=0.5)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4,
            WeightJitter = 0.99f,
            WeightJitterStdDev = 0.5f
        });

        configs.Add(new ConfigToTest
        {
            Description = "Med Pop + Low Jitter (WJ=0.8, SD=0.15)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4,
            WeightJitter = 0.8f,
            WeightJitterStdDev = 0.15f
        });

        configs.Add(new ConfigToTest
        {
            Description = "Med Pop + High Reset (WR=0.2)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4,
            WeightReset = 0.2f
        });

        configs.Add(new ConfigToTest
        {
            Description = "Med Pop + Low Reset (WR=0.05)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4,
            WeightReset = 0.05f
        });

        configs.Add(new ConfigToTest
        {
            Description = "Med Pop + High ActSwap (AS=0.05)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4,
            ActivationSwap = 0.05f
        });

        configs.Add(new ConfigToTest
        {
            Description = "Med Pop + No ActSwap (AS=0.0)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4,
            ActivationSwap = 0.0f
        });

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

        var resultLock = new object();
        var results = new (bool solved, long timeMs, int generations)[SeedsPerConfig];

        Parallel.For(0, SeedsPerConfig, new ParallelOptions { MaxDegreeOfParallelism = 16 }, seed =>
        {
            var progress = new TrialProgress();
            results[seed] = RunSingleTrial(config, seed, progress);
        });

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

        var topology = CreateCorridorTopology();

        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = config.SpeciesCount,
            IndividualsPerSpecies = config.IndividualsPerSpecies,
            Elites = config.Elites,
            TournamentSize = config.TournamentSize,
            ParentPoolPercentage = config.ParentPoolPercentage,
            MinSpeciesCount = config.SpeciesCount, // Disable species culling/diversification
            MutationRates = new MutationRates
            {
                WeightJitter = config.WeightJitter,
                WeightJitterStdDev = config.WeightJitterStdDev,
                WeightReset = config.WeightReset,
                ActivationSwap = config.ActivationSwap
            }
        };

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(evolutionConfig, topology);

        var sharedWorld = SimpleCarWorld.LoadFromFile(maxSteps: 320);

        var environments = new List<FollowTheCorridorEnvironment>();
        foreach (var species in population.AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                environments.Add(new FollowTheCorridorEnvironment(sharedWorld));
            }
        }

        var stopwatch = Stopwatch.StartNew();
        int generation = 0;
        bool solved = false;

        progress.Generation = 0;
        progress.ElapsedMs = 0;

        while (stopwatch.ElapsedMilliseconds < MaxTimeoutSeconds * 1000 && !solved)
        {
            var bestInd = population.GetBestIndividual();
            progress.Generation = generation;
            progress.BestFitness = bestInd?.individual.Fitness ?? 0f;
            progress.ElapsedMs = stopwatch.ElapsedMilliseconds;

            int envIdx = 0;
            var stepCounts = new int[environments.Count];

            foreach (var species in population.AllSpecies)
            {
                var evaluator = new CPUEvaluator(species.Topology);

                for (int indIdx = 0; indIdx < species.Individuals.Count; indIdx++)
                {
                    environments[envIdx].Reset(seed: generation);

                    float totalReward = 0f;
                    var observations = new float[environments[envIdx].InputCount];
                    int steps = 0;

                    while (!environments[envIdx].IsTerminal())
                    {
                        environments[envIdx].GetObservations(observations);
                        var actions = evaluator.Evaluate(species.Individuals[indIdx], observations);
                        float reward = environments[envIdx].Step(actions);
                        totalReward += reward;
                        steps++;
                    }

                    stepCounts[envIdx] = steps;

                    var ind = species.Individuals[indIdx];
                    ind.Fitness = totalReward;
                    species.Individuals[indIdx] = ind;

                    envIdx++;
                }
            }

            var best = population.GetBestIndividual();

            if (best.HasValue && best.Value.individual.Fitness >= SolvedThreshold)
            {
                envIdx = 0;
                foreach (var species in population.AllSpecies)
                {
                    for (int indIdx = 0; indIdx < species.Individuals.Count; indIdx++)
                    {
                        if (species.Individuals[indIdx].Fitness >= SolvedThreshold &&
                            stepCounts[envIdx] <= MaxStepsForSuccess)
                        {
                            solved = true;
                            progress.Solved = true;
                            break;
                        }
                        envIdx++;
                    }
                    if (solved) break;
                }
                if (solved) break;
            }

            evolver.StepGeneration(population);
            generation++;
        }

        stopwatch.Stop();

        progress.Completed = true;
        progress.Generation = generation;
        progress.ElapsedMs = stopwatch.ElapsedMilliseconds;

        return (solved, stopwatch.ElapsedMilliseconds, generation);
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

    private static SpeciesSpec CreateCorridorTopology()
    {
        var topology = new SpeciesSpec
        {
            RowCounts = new[] { 1, 9, 12, 2 },
            AllowedActivationsPerRow = new uint[]
            {
                0b00000000001,
                0b11111111111,
                0b11111111111,
                0b00000000011
            },
            MaxInDegree = 12,
            Edges = new List<(int, int)>()
        };

        for (int src = 0; src < 10; src++)
        {
            for (int dst = 10; dst < 22; dst++)
            {
                topology.Edges.Add((src, dst));
            }
        }

        for (int src = 10; src < 22; src++)
        {
            topology.Edges.Add((src, 22));
            topology.Edges.Add((src, 23));
        }

        topology.BuildRowPlans();
        return topology;
    }

    private class ConfigToTest
    {
        public string Description { get; set; } = "";
        public int SpeciesCount { get; set; }
        public int IndividualsPerSpecies { get; set; }
        public int Elites { get; set; }
        public float ParentPoolPercentage { get; set; }
        public int TournamentSize { get; set; }

        // Mutation parameters
        public float WeightJitter { get; set; } = 0.95f;
        public float WeightJitterStdDev { get; set; } = 0.3f;
        public float WeightReset { get; set; } = 0.1f;
        public float ActivationSwap { get; set; } = 0.01f;
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
