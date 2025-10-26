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
    private const int SeedsPerConfig = 5;
    private const float SolvedThreshold = 0.9f; // 90% of track

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
            Console.WriteLine($"[{configIdx}/{configs.Count}] Testing: {config.Description}");

            var result = EvaluateConfiguration(config);
            results.Add(result);

            Console.WriteLine($"  Result: {result.SolvedCount}/{SeedsPerConfig} solved, " +
                            $"avg time: {result.AvgTimeToSolve:F1}s, " +
                            $"avg gens: {result.AvgGenerations:F1}");
            Console.WriteLine();
        }

        // Print summary
        PrintSummary(results);
    }

    private static List<ConfigToTest> GenerateConfigurations()
    {
        var configs = new List<ConfigToTest>();

        // Current baseline
        configs.Add(new ConfigToTest
        {
            Description = "Baseline (40x40, E=1, P=0.2, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        // Vary population structure
        configs.Add(new ConfigToTest
        {
            Description = "Fewer species (20x80, E=1, P=0.2, T=4)",
            SpeciesCount = 20,
            IndividualsPerSpecies = 80,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "More species (80x20, E=1, P=0.2, T=4)",
            SpeciesCount = 80,
            IndividualsPerSpecies = 20,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "Smaller pop (40x20, E=1, P=0.2, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 20,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        // Vary parent pool percentage
        configs.Add(new ConfigToTest
        {
            Description = "Tighter selection (40x40, E=1, P=0.1, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.1f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "Looser selection (40x40, E=1, P=0.4, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.4f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "No parent pool (40x40, E=1, P=1.0, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 1.0f,
            TournamentSize = 4
        });

        // Vary tournament size
        configs.Add(new ConfigToTest
        {
            Description = "Weaker tournament (40x40, E=1, P=0.2, T=2)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 2
        });

        configs.Add(new ConfigToTest
        {
            Description = "Stronger tournament (40x40, E=1, P=0.2, T=8)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 8
        });

        // Vary elites
        configs.Add(new ConfigToTest
        {
            Description = "More elites (40x40, E=2, P=0.2, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 2,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        configs.Add(new ConfigToTest
        {
            Description = "Many elites (40x40, E=4, P=0.2, T=4)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 4,
            ParentPoolPercentage = 0.2f,
            TournamentSize = 4
        });

        // Combined optimizations
        configs.Add(new ConfigToTest
        {
            Description = "Fast + tight (40x20, E=1, P=0.1, T=8)",
            SpeciesCount = 40,
            IndividualsPerSpecies = 20,
            Elites = 1,
            ParentPoolPercentage = 0.1f,
            TournamentSize = 8
        });

        return configs;
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

        // Run all seeds in parallel
        var tasks = new Task<(bool solved, long timeMs, int generations)>[SeedsPerConfig];
        for (int seed = 0; seed < SeedsPerConfig; seed++)
        {
            int seedCopy = seed; // Capture for closure
            tasks[seed] = Task.Run(() => RunSingleTrial(config, seedCopy));
        }

        // Wait for all trials to complete
        Task.WaitAll(tasks);

        // Collect results
        foreach (var task in tasks)
        {
            var (solved, timeMs, generations) = task.Result;
            if (solved)
            {
                result.SolvedCount++;
                result.TimesToSolve.Add(timeMs / 1000.0); // Convert to seconds
                result.Generations.Add(generations);
            }
        }

        return result;
    }

    private static (bool solved, long timeMs, int generations) RunSingleTrial(
        ConfigToTest config,
        int seed)
    {
        // Create topology
        var topology = CreateCorridorTopology();

        // Configure evolution
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = config.SpeciesCount,
            IndividualsPerSpecies = config.IndividualsPerSpecies,
            Elites = config.Elites,
            TournamentSize = config.TournamentSize,
            ParentPoolPercentage = config.ParentPoolPercentage
        };

        // Initialize population
        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(evolutionConfig, topology);

        // Create evaluators and environments for all individuals
        var environments = new List<FollowTheCorridorEnvironment>();
        var individuals = new List<Individual>();
        var evaluators = new List<CPUEvaluator>();

        foreach (var species in population.AllSpecies)
        {
            var eval = new CPUEvaluator(species.Topology);
            foreach (var individual in species.Individuals)
            {
                environments.Add(new FollowTheCorridorEnvironment(maxSteps: 320));
                individuals.Add(individual);
                evaluators.Add(eval);
            }
        }

        var stopwatch = Stopwatch.StartNew();
        int generation = 0;
        bool solved = false;

        while (stopwatch.ElapsedMilliseconds < MaxTimeoutSeconds * 1000 && !solved)
        {
            // Evaluate all individuals
            for (int i = 0; i < environments.Count; i++)
            {
                environments[i].Reset(seed: generation);
                float totalReward = 0f;
                var observations = new float[environments[i].InputCount];

                while (!environments[i].IsTerminal())
                {
                    environments[i].GetObservations(observations);
                    var actions = evaluators[i].Evaluate(individuals[i], observations);
                    float reward = environments[i].Step(actions);
                    totalReward += reward;
                }

                // Update fitness
                int speciesIdx = 0, individualIdx = 0;
                for (int s = 0; s < population.AllSpecies.Count; s++)
                {
                    if (i < (s + 1) * evolutionConfig.IndividualsPerSpecies)
                    {
                        speciesIdx = s;
                        individualIdx = i - s * evolutionConfig.IndividualsPerSpecies;
                        break;
                    }
                }

                var species = population.AllSpecies[speciesIdx];
                var ind = species.Individuals[individualIdx];
                ind.Fitness = totalReward;
                species.Individuals[individualIdx] = ind;
                individuals[i] = ind;
            }

            // Check if solved
            var best = population.GetBestIndividual();
            if (best.HasValue && best.Value.individual.Fitness >= SolvedThreshold)
            {
                solved = true;
                break;
            }

            // Evolve
            evolver.StepGeneration(population);
            generation++;
        }

        stopwatch.Stop();

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
