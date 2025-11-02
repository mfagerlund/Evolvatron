using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Benchmarks;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class LandscapeNavigationTest
{
    private readonly ITestOutputHelper _output;

    public LandscapeNavigationTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Sphere5D_SolvesWithOptunaHyperparameters()
    {
        const int maxGenerations = 20;
        const int numSeeds = 10;
        const float solveThreshold = -0.01f;

        _output.WriteLine("=== SPHERE 5D OPTIMIZATION TEST ===");
        _output.WriteLine($"Using Optuna Trial 23 hyperparameters");
        _output.WriteLine($"Max generations: {maxGenerations}");
        _output.WriteLine($"Seeds: {numSeeds}");
        _output.WriteLine($"Solve threshold: {solveThreshold} (1% of optimum)");
        _output.WriteLine("");

        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Sphere,
            dimensions: 5,
            timesteps: 50,
            stepSize: 0.1f,
            minBound: -5f,
            maxBound: 5f,
            observationType: ObservationType.FullPosition,
            seed: 42);

        var results = RunEvolutionOnLandscape(task, maxGenerations, numSeeds, solveThreshold);

        _output.WriteLine("");
        _output.WriteLine("=== SUMMARY ===");

        int solvedCount = results.Count(r => r.Solved);
        var solvedRuns = results.Where(r => r.Solved).ToList();

        if (solvedRuns.Any())
        {
            float avgGenerations = (float)solvedRuns.Average(r => r.GenerationsToSolve);
            float avgEvaluations = (float)solvedRuns.Average(r => r.EvaluationsToSolve);
            int minGenerations = solvedRuns.Min(r => r.GenerationsToSolve);
            int maxGenerations_ = solvedRuns.Max(r => r.GenerationsToSolve);

            _output.WriteLine($"Solved: {solvedCount}/{numSeeds} ({(solvedCount * 100.0f / numSeeds):F1}%)");
            _output.WriteLine($"Avg generations: {avgGenerations:F1}");
            _output.WriteLine($"Avg evaluations: {avgEvaluations:F0}");
            _output.WriteLine($"Generation range: {minGenerations} - {maxGenerations_}");
        }
        else
        {
            _output.WriteLine($"Solved: 0/{numSeeds} (0.0%)");
        }

        _output.WriteLine("");
        _output.WriteLine("Per-seed results:");
        foreach (var result in results)
        {
            if (result.Solved)
            {
                _output.WriteLine($"  Seed {result.Seed}: SOLVED at gen {result.GenerationsToSolve} " +
                    $"(fitness={result.FinalFitness:F6}, evals={result.EvaluationsToSolve})");
            }
            else
            {
                _output.WriteLine($"  Seed {result.Seed}: NOT SOLVED " +
                    $"(best fitness={result.FinalFitness:F6})");
            }
        }

        Assert.True(solvedCount >= 8,
            $"Should solve at least 80% of seeds, but only solved {solvedCount}/{numSeeds}");
    }

    [Fact]
    public void Rosenbrock5D_SolvesWithOptunaHyperparameters()
    {
        const int maxGenerations = 150;
        const int numSeeds = 10;
        const float solveThreshold = -0.10f;

        _output.WriteLine("=== ROSENBROCK 5D OPTIMIZATION TEST ===");
        _output.WriteLine($"Using Optuna Trial 23 hyperparameters");
        _output.WriteLine($"Max generations: {maxGenerations}");
        _output.WriteLine($"Seeds: {numSeeds}");
        _output.WriteLine($"Solve threshold: {solveThreshold} (10% of optimum)");
        _output.WriteLine("");

        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Rosenbrock,
            dimensions: 5,
            timesteps: 150,
            stepSize: 0.1f,
            minBound: -2f,
            maxBound: 2f,
            observationType: ObservationType.FullPosition,
            seed: 42);

        var results = RunEvolutionOnLandscape(task, maxGenerations, numSeeds, solveThreshold);

        _output.WriteLine("");
        _output.WriteLine("=== SUMMARY ===");

        int solvedCount = results.Count(r => r.Solved);
        var solvedRuns = results.Where(r => r.Solved).ToList();

        if (solvedRuns.Any())
        {
            float avgGenerations = (float)solvedRuns.Average(r => r.GenerationsToSolve);
            float avgEvaluations = (float)solvedRuns.Average(r => r.EvaluationsToSolve);
            int minGenerations = solvedRuns.Min(r => r.GenerationsToSolve);
            int maxGenerations_ = solvedRuns.Max(r => r.GenerationsToSolve);

            _output.WriteLine($"Solved: {solvedCount}/{numSeeds} ({(solvedCount * 100.0f / numSeeds):F1}%)");
            _output.WriteLine($"Avg generations: {avgGenerations:F1}");
            _output.WriteLine($"Avg evaluations: {avgEvaluations:F0}");
            _output.WriteLine($"Generation range: {minGenerations} - {maxGenerations_}");
        }
        else
        {
            _output.WriteLine($"Solved: 0/{numSeeds} (0.0%)");
        }

        _output.WriteLine("");
        _output.WriteLine("Per-seed results:");
        foreach (var result in results)
        {
            if (result.Solved)
            {
                _output.WriteLine($"  Seed {result.Seed}: SOLVED at gen {result.GenerationsToSolve} " +
                    $"(fitness={result.FinalFitness:F6}, evals={result.EvaluationsToSolve})");
            }
            else
            {
                _output.WriteLine($"  Seed {result.Seed}: NOT SOLVED " +
                    $"(best fitness={result.FinalFitness:F6})");
            }
        }

        Assert.True(solvedCount >= 7,
            $"Should solve at least 70% of seeds, but only solved {solvedCount}/{numSeeds}");
    }

    [Fact]
    public void Rastrigin8D_SolvesWithOptunaHyperparameters()
    {
        const int maxGenerations = 50;
        const int numSeeds = 10;
        const float solveThreshold = -0.15f;

        _output.WriteLine("=== RASTRIGIN 8D OPTIMIZATION TEST ===");
        _output.WriteLine($"Using Optuna Trial 23 hyperparameters");
        _output.WriteLine($"Max generations: {maxGenerations}");
        _output.WriteLine($"Seeds: {numSeeds}");
        _output.WriteLine($"Solve threshold: {solveThreshold} (15% of optimum)");
        _output.WriteLine("");

        var task = new LandscapeNavigationTask(
            OptimizationLandscapes.Rastrigin,
            dimensions: 8,
            timesteps: 100,
            stepSize: 0.1f,
            minBound: -5.12f,
            maxBound: 5.12f,
            observationType: ObservationType.FullPosition,
            seed: 42);

        var results = RunEvolutionOnLandscape(task, maxGenerations, numSeeds, solveThreshold);

        _output.WriteLine("");
        _output.WriteLine("=== SUMMARY ===");

        int solvedCount = results.Count(r => r.Solved);
        var solvedRuns = results.Where(r => r.Solved).ToList();

        if (solvedRuns.Any())
        {
            float avgGenerations = (float)solvedRuns.Average(r => r.GenerationsToSolve);
            float avgEvaluations = (float)solvedRuns.Average(r => r.EvaluationsToSolve);
            int minGenerations = solvedRuns.Min(r => r.GenerationsToSolve);
            int maxGenerations_ = solvedRuns.Max(r => r.GenerationsToSolve);

            _output.WriteLine($"Solved: {solvedCount}/{numSeeds} ({(solvedCount * 100.0f / numSeeds):F1}%)");
            _output.WriteLine($"Avg generations: {avgGenerations:F1}");
            _output.WriteLine($"Avg evaluations: {avgEvaluations:F0}");
            _output.WriteLine($"Generation range: {minGenerations} - {maxGenerations_}");
        }
        else
        {
            _output.WriteLine($"Solved: 0/{numSeeds} (0.0%)");
        }

        _output.WriteLine("");
        _output.WriteLine("Per-seed results:");
        foreach (var result in results)
        {
            if (result.Solved)
            {
                _output.WriteLine($"  Seed {result.Seed}: SOLVED at gen {result.GenerationsToSolve} " +
                    $"(fitness={result.FinalFitness:F6}, evals={result.EvaluationsToSolve})");
            }
            else
            {
                _output.WriteLine($"  Seed {result.Seed}: NOT SOLVED " +
                    $"(best fitness={result.FinalFitness:F6})");
            }
        }

        Assert.True(solvedCount >= 5,
            $"Should solve at least 50% of seeds, but only solved {solvedCount}/{numSeeds}");
    }

    private List<RunResult> RunEvolutionOnLandscape(
        LandscapeNavigationTask task,
        int maxGenerations,
        int numSeeds,
        float solveThreshold)
    {
        var results = new RunResult[numSeeds];
        var config = new EvolutionConfig();

        _output.WriteLine($"Running {numSeeds} seeds in parallel...");

        Parallel.For(0, numSeeds, seed =>
        {
            var evolver = new Evolver(seed);
            var random = new Random(seed);

            var topology = new SpeciesBuilder()
                .AddInputRow(task.GetObservationSize())
                .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
                .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
                .AddOutputRow(task.GetDimensions(), ActivationType.Tanh)
                .WithMaxInDegree(10)
                .InitializeDense(random, density: 0.3f)
                .Build();

            var population = evolver.InitializePopulation(config, topology);
            var environment = new LandscapeEnvironment(task);
            var evaluator = new SimpleFitnessEvaluator();

            evaluator.EvaluatePopulation(population, environment, seed: 0);

            var result = new RunResult { Seed = seed };

            for (int gen = 1; gen <= maxGenerations; gen++)
            {
                evaluator.EvaluatePopulation(population, environment, seed: 0);

                var stats = population.GetStatistics();

                if (stats.BestFitness >= solveThreshold)
                {
                    result.Solved = true;
                    result.GenerationsToSolve = gen;
                    result.EvaluationsToSolve = gen * config.SpeciesCount * config.IndividualsPerSpecies;
                    result.FinalFitness = stats.BestFitness;
                    break;
                }

                evolver.StepGeneration(population);
            }

            if (!result.Solved)
            {
                evaluator.EvaluatePopulation(population, environment, seed: 0);
                var finalStats = population.GetStatistics();
                result.FinalFitness = finalStats.BestFitness;
            }

            results[seed] = result;
        });

        _output.WriteLine("All seeds completed!");
        _output.WriteLine("");

        foreach (var result in results)
        {
            if (result.Solved)
            {
                _output.WriteLine($"--- Seed {result.Seed}: SOLVED at gen {result.GenerationsToSolve} (fitness={result.FinalFitness:F6})");
            }
            else
            {
                _output.WriteLine($"--- Seed {result.Seed}: NOT SOLVED (best={result.FinalFitness:F6})");
            }
        }

        return results.ToList();
    }

    private class RunResult
    {
        public int Seed { get; set; }
        public bool Solved { get; set; }
        public int GenerationsToSolve { get; set; }
        public int EvaluationsToSolve { get; set; }
        public float FinalFitness { get; set; }
    }
}
