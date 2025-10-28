using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using System.Diagnostics;
using Colonel.Tests.HagridTests.FollowTheCorridor;
using static Colonel.Tests.HagridTests.FollowTheCorridor.SimpleCarWorld;

namespace Evolvatron.Demo;

public static class SimpleTimingTest
{
    public static void Run()
    {
        Console.WriteLine("=== Simple Timing Test ===");
        Console.WriteLine("Running 1 generation with 40x40 = 1600 individuals");
        Console.WriteLine();

        // Create topology
        var random = new Random(42);
        var topology = CreateCorridorTopology(random);

        // Configure evolution
        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = 40,
            IndividualsPerSpecies = 40,
            Elites = 1,
            TournamentSize = 4,
            ParentPoolPercentage = 0.2f
        };

        Console.WriteLine("Initializing population...");
        var sw = Stopwatch.StartNew();
        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(evolutionConfig, topology);
        Console.WriteLine($"Population initialized in {sw.ElapsedMilliseconds}ms");

        Console.WriteLine($"Creating {evolutionConfig.SpeciesCount * evolutionConfig.IndividualsPerSpecies} environments and evaluators...");
        sw.Restart();
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
        Console.WriteLine($"Environments created in {sw.ElapsedMilliseconds}ms");

        Console.WriteLine($"\\nEvaluating all {individuals.Count} individuals (generation 0)...");
        sw.Restart();
        int count = 0;
        foreach (var (env, ind, eval) in environments.Zip(individuals, evaluators))
        {
            env.Reset(seed: 0);
            float totalReward = 0f;
            var observations = new float[env.InputCount];

            while (!env.IsTerminal())
            {
                env.GetObservations(observations);
                var actions = eval.Evaluate(ind, observations);
                float reward = env.Step(actions);
                totalReward += reward;
            }

            count++;
            if (count % 200 == 0)
            {
                Console.WriteLine($"  Evaluated {count}/{individuals.Count} individuals ({sw.ElapsedMilliseconds}ms elapsed)");
            }
        }
        Console.WriteLine($"\\nGeneration 0 completed in {sw.ElapsedMilliseconds}ms ({sw.ElapsedMilliseconds / 1000.0:F1}s)");
        Console.WriteLine($"Average per individual: {sw.ElapsedMilliseconds / (double)individuals.Count:F2}ms");
    }

    private static SpeciesSpec CreateCorridorTopology(Random random)
    {
        return new SpeciesBuilder()
            .AddInputRow(9)
            .AddHiddenRow(12, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeSparse(random)
            .Build();
    }
}
