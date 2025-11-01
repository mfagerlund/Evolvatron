using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Benchmarks;

namespace Evolvatron.Demo;

public static class LandscapeBenchmarkDemo
{
    public static void Run()
    {
        Console.WriteLine("=== Landscape Navigation Benchmark ===");
        Console.WriteLine();

        RunSphere5DEasy();
        Console.WriteLine();

        RunRosenbrock5DEasy();
        Console.WriteLine();

        RunRastrigin8DMedium();
    }

    private static void RunSphere5DEasy()
    {
        Console.WriteLine("--- Sphere-20D (Gradient-Only, Near-Perfect Required) ---");
        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Sphere,
            dimensions: 20,
            timesteps: 150,
            stepSize: 0.01f,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.GradientOnly,
            seed: 42);

        var config = new EvolutionConfig
        {
            SpeciesCount = 12,
            IndividualsPerSpecies = 100,
            Elites = 4,
            TournamentSize = 4,
        };

        var result = RunEvolution(task, config, maxGenerations: 1000, successThreshold: 50.0f);
        PrintResults("Sphere-20D-GradientOnly", result);
    }

    private static void RunRosenbrock5DEasy()
    {
        Console.WriteLine("--- Rosenbrock-15D (Gradient-Only, Deep Valley) ---");
        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Rosenbrock,
            dimensions: 15,
            timesteps: 300,
            stepSize: 0.01f,
            minBound: -2f,
            maxBound: 2f,
            observationType: ObservationType.GradientOnly,
            seed: 42);

        var config = new EvolutionConfig
        {
            SpeciesCount = 12,
            IndividualsPerSpecies = 100,
            Elites = 4,
            TournamentSize = 4,
        };

        var result = RunEvolution(task, config, maxGenerations: 1000, successThreshold: 1000.0f);
        PrintResults("Rosenbrock-15D-GradientOnly", result);
    }

    private static void RunRastrigin8DMedium()
    {
        Console.WriteLine("--- Rastrigin-20D (Partial Observability, Highly Multimodal) ---");
        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Rastrigin,
            dimensions: 20,
            timesteps: 250,
            stepSize: 0.01f,
            minBound: -5.12f,
            maxBound: 5.12f,
            observationType: ObservationType.PartialObservability,
            seed: 42);

        var config = new EvolutionConfig
        {
            SpeciesCount = 16,
            IndividualsPerSpecies = 100,
            Elites = 4,
            TournamentSize = 4,
        };

        var result = RunEvolution(task, config, maxGenerations: 1000, successThreshold: 100.0f);
        PrintResults("Rastrigin-20D-PartialObs", result);
    }

    private static EvolutionResult RunEvolution(
        LandscapeNavigationTask task,
        EvolutionConfig config,
        int maxGenerations,
        float successThreshold)
    {
        int inputCount = task.GetObservationSize();
        int outputCount = task.GetDimensions();

        var topology = new SpeciesBuilder()
            .AddInputRow(inputCount)
            .AddHiddenRow(12,
                ActivationType.Linear,
                ActivationType.Tanh,
                ActivationType.ReLU,
                ActivationType.Sigmoid,
                ActivationType.LeakyReLU,
                ActivationType.ELU,
                ActivationType.Softsign,
                ActivationType.Softplus)
            .AddOutputRow(outputCount, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeSparse(new Random(42))
            .Build();

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);
        var environment = new LandscapeEnvironmentAdapter(task);
        var evaluator = new SimpleFitnessEvaluator();

        float bestFitness = float.MinValue;
        int bestGeneration = -1;
        var startTime = DateTime.Now;

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            var best = population.GetBestIndividual();
            if (best.HasValue)
            {
                float fitness = best.Value.individual.Fitness;
                if (fitness > bestFitness)
                {
                    bestFitness = fitness;
                    bestGeneration = gen;
                }

                if (gen % 20 == 0 || fitness >= successThreshold)
                {
                    var stats = population.GetStatistics();
                    Console.WriteLine($"Gen {gen,3}: Best={fitness,8:F3} Mean={stats.MeanFitness,8:F3} Median={stats.MedianFitness,8:F3}");
                }

                if (fitness >= successThreshold)
                {
                    var elapsed = DateTime.Now - startTime;
                    return new EvolutionResult
                    {
                        Success = true,
                        BestFitness = fitness,
                        GenerationsToSolve = gen,
                        TotalGenerations = gen + 1,
                        ElapsedSeconds = elapsed.TotalSeconds
                    };
                }
            }

            evolver.StepGeneration(population);
        }

        var totalElapsed = DateTime.Now - startTime;
        return new EvolutionResult
        {
            Success = false,
            BestFitness = bestFitness,
            GenerationsToSolve = -1,
            TotalGenerations = maxGenerations,
            ElapsedSeconds = totalElapsed.TotalSeconds
        };
    }

    private static void PrintResults(string benchmarkName, EvolutionResult result)
    {
        Console.WriteLine($"Benchmark: {benchmarkName}");
        Console.WriteLine($"  Success: {result.Success}");
        Console.WriteLine($"  Best Fitness: {result.BestFitness:F3}");
        if (result.Success)
        {
            Console.WriteLine($"  Generations to Solve: {result.GenerationsToSolve}");
        }
        Console.WriteLine($"  Total Generations: {result.TotalGenerations}");
        Console.WriteLine($"  Elapsed: {result.ElapsedSeconds:F1}s");
    }

    private class EvolutionResult
    {
        public bool Success { get; set; }
        public float BestFitness { get; set; }
        public int GenerationsToSolve { get; set; }
        public int TotalGenerations { get; set; }
        public double ElapsedSeconds { get; set; }
    }
}
