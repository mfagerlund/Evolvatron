using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;

namespace Evolvatron.Demo;

/// <summary>
/// Demo: Evolve neural networks to solve the two-spiral classification problem.
/// This is a classic benchmark that requires learning complex non-linear decision boundaries.
/// </summary>
public static class SpiralClassificationDemo
{
    public static void Run()
    {
        Console.WriteLine("=== Two-Spiral Classification Demo ===");
        Console.WriteLine("Evolving neural networks to classify two interleaved spirals.\n");

        // Create topology: 2 inputs (x,y) -> 8 hidden -> 8 hidden -> 1 output
        var topology = CreateSpiralTopology();

        // Configure evolution
        var config = new EvolutionConfig
        {
            SpeciesCount = 10,
            IndividualsPerSpecies = 80,
            Elites = 2,
            TournamentSize = 4,
            ParentPoolPercentage = 0.3f
        };

        // Initialize population
        var evolver = new Evolver(seed: DateTime.Now.Millisecond);
        var population = evolver.InitializePopulation(config, topology);

        // Create environment (50 points per spiral = 100 total test cases)
        var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
        var evaluator = new SimpleFitnessEvaluator();

        Console.WriteLine($"Population: {config.SpeciesCount} species x {config.IndividualsPerSpecies} individuals = {config.SpeciesCount * config.IndividualsPerSpecies} total");
        Console.WriteLine($"Test cases: {environment.MaxSteps} spiral points");
        Console.WriteLine($"Topology: {topology.RowCounts[0]} inputs -> {topology.RowCounts[1]} hidden -> {topology.RowCounts[2]} hidden -> {topology.RowCounts[3]} output");
        Console.WriteLine();

        // Evolution loop
        int maxGenerations = 1000;
        float successThreshold = -0.05f; // Average squared error < 0.05

        for (int gen = 0; gen < maxGenerations; gen++)
        {
            // Evaluate all individuals
            evaluator.EvaluatePopulation(population, environment, seed: gen);

            // Get best individual
            var best = population.GetBestIndividual();
            float bestFitness = best?.individual.Fitness ?? float.MinValue;

            // Report progress
            if (gen % 10 == 0 || bestFitness >= successThreshold)
            {
                float accuracy = ComputeAccuracy(best.Value.individual, best.Value.species.Topology, environment);
                Console.WriteLine($"Gen {gen,4}: Fitness = {bestFitness,8:F4} | Accuracy = {accuracy * 100,5:F1}% | Edges = {best.Value.species.Topology.Edges.Count}");
            }

            // Check for success
            if (bestFitness >= successThreshold)
            {
                Console.WriteLine($"\nSUCCESS! Solved spiral classification in {gen} generations!");
                PrintDetailedResults(best.Value.individual, best.Value.species.Topology, environment);
                return;
            }

            // Evolve to next generation
            evolver.StepGeneration(population);
        }

        Console.WriteLine($"\nReached {maxGenerations} generations. Evolution incomplete.");
        var final = population.GetBestIndividual();
        if (final.HasValue)
        {
            PrintDetailedResults(final.Value.individual, final.Value.species.Topology, environment);
        }
    }

    private static float ComputeAccuracy(Individual individual, SpeciesSpec topology, SpiralEnvironment environment)
    {
        var cpuEval = new CPUEvaluator(topology);
        var observations = new float[2];
        var allPoints = environment.GetAllPoints();

        int correct = 0;
        foreach (var (x, y, expectedLabel) in allPoints)
        {
            observations[0] = x;
            observations[1] = y;

            var outputs = cpuEval.Evaluate(individual, observations);
            float output = outputs[0];

            // For tanh output: negative = spiral 0, positive = spiral 1
            float predictedLabel = output > 0 ? 1f : -1f;
            if (MathF.Abs(predictedLabel - expectedLabel) < 0.1f)
                correct++;
        }

        return (float)correct / allPoints.Count;
    }

    private static void PrintDetailedResults(Individual individual, SpeciesSpec topology, SpiralEnvironment environment)
    {
        var cpuEval = new CPUEvaluator(topology);
        var observations = new float[2];
        var allPoints = environment.GetAllPoints();

        int correct = 0;
        float totalError = 0f;

        foreach (var (x, y, expectedLabel) in allPoints)
        {
            observations[0] = x;
            observations[1] = y;

            var outputs = cpuEval.Evaluate(individual, observations);
            float output = outputs[0];

            float predictedLabel = output > 0 ? 1f : -1f;
            float error = (output - expectedLabel) * (output - expectedLabel);

            if (MathF.Abs(predictedLabel - expectedLabel) < 0.1f)
                correct++;

            totalError += error;
        }

        float accuracy = (float)correct / allPoints.Count;
        float avgError = totalError / allPoints.Count;

        Console.WriteLine($"\nFinal Results:");
        Console.WriteLine($"  Accuracy: {correct}/{allPoints.Count} ({accuracy * 100:F1}%)");
        Console.WriteLine($"  Avg Squared Error: {avgError:F4}");
        Console.WriteLine($"  Topology: {topology.Edges.Count} edges, {topology.TotalNodes} nodes");

        var activeNodes = ConnectivityValidator.ComputeActiveNodes(topology);
        int activeCount = activeNodes.Count(x => x);
        Console.WriteLine($"  Active nodes: {activeCount}/{topology.TotalNodes}");
    }

    private static SpeciesSpec CreateSpiralTopology()
    {
        var random = new Random(42);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeSparse(random)
            .Build();
    }
}
