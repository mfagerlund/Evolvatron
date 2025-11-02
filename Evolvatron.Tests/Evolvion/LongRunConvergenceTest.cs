using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Long-run convergence test: Can evolution fully solve spiral classification
/// given enough generations (2000)?
///
/// Goal: Determine if the algorithm can reach near-perfect fitness (MSE ≈ 0)
/// or if it plateaus at a suboptimal solution.
/// </summary>
public class LongRunConvergenceTest
{
    private readonly ITestOutputHelper _output;

    public LongRunConvergenceTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void SpiralClassification_2000Generations_CanItSolve()
    {
        const int generations = 2000;
        const int reportEvery = 100;
        const int seeds = 3;

        _output.WriteLine("=== LONG RUN CONVERGENCE TEST (2000 GENERATIONS) ===");
        _output.WriteLine($"Seeds: {seeds}, Report interval: every {reportEvery} generations");
        _output.WriteLine($"Task: Spiral classification (2→8→8→1)");
        _output.WriteLine($"Goal: Can we reach MSE ≈ 0 (perfect classification)?");
        _output.WriteLine("");

        var config = new EvolutionConfig(); // Use optimized defaults

        var allRuns = new List<RunHistory>();

        for (int seed = 0; seed < seeds; seed++)
        {
            _output.WriteLine($"\n=== SEED {seed} ===");

            var evolver = new Evolver(seed);
            var random = new Random(seed);

            var topology = new SpeciesBuilder()
                .AddInputRow(2)
                .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
                .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
                .AddOutputRow(1, ActivationType.Tanh)
                .WithMaxInDegree(10)
                .InitializeDense(random, density: 0.3f)
                .Build();

            // Compute topology hash and statistics
            int topologyHash = ComputeTopologyHash(topology);
            float density = topology.TotalEdges / (float)(topology.TotalNodes * topology.TotalNodes);

            _output.WriteLine($"Topology: Nodes={topology.TotalNodes}, Edges={topology.TotalEdges}, " +
                $"Density={density:F3}, Hash={topologyHash:X8}");

            var population = evolver.InitializePopulation(config, topology);
            var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
            var evaluator = new SimpleFitnessEvaluator();

            var history = new RunHistory { Seed = seed };

            // Initial evaluation
            evaluator.EvaluatePopulation(population, environment, seed: 0);
            var gen0Stats = population.GetStatistics();
            history.Checkpoints.Add((0, gen0Stats.BestFitness, gen0Stats.MeanFitness));

            // Compute initial population diversity
            var firstSpecies = population.AllSpecies.First();
            float weightVariance = ComputeWeightVariance(firstSpecies);

            _output.WriteLine($"Gen 0: Best={gen0Stats.BestFitness:F6}, Mean={gen0Stats.MeanFitness:F6}, " +
                $"PopSize={config.SpeciesCount * config.IndividualsPerSpecies}, WeightVar={weightVariance:F4}");

            // Evolution loop - check for solve EVERY generation, report periodically
            for (int gen = 1; gen <= generations; gen++)
            {
                evaluator.EvaluatePopulation(population, environment, seed: 0);
                evolver.StepGeneration(population);

                var stats = population.GetStatistics();

                // CHECK FOR SOLVE EVERY GENERATION
                if (stats.BestFitness > -0.01f) // MSE < 0.01
                {
                    history.Checkpoints.Add((gen, stats.BestFitness, stats.MeanFitness));
                    _output.WriteLine($"Gen {gen,4}: Best={stats.BestFitness:F6}, Mean={stats.MeanFitness:F6}, " +
                        $"Species={population.AllSpecies.Count}, TotalCreated={population.TotalSpeciesCreated}");
                    _output.WriteLine($"*** SOLVED at generation {gen}! MSE < 0.01 ***");
                    history.SolvedAtGeneration = gen;
                    break;
                }

                // Periodic reporting
                if (gen % reportEvery == 0)
                {
                    history.Checkpoints.Add((gen, stats.BestFitness, stats.MeanFitness));
                    _output.WriteLine($"Gen {gen,4}: Best={stats.BestFitness:F6}, Mean={stats.MeanFitness:F6}, " +
                        $"Species={population.AllSpecies.Count}, TotalCreated={population.TotalSpeciesCreated}");
                }
            }

            // Final evaluation
            if (!history.SolvedAtGeneration.HasValue)
            {
                evaluator.EvaluatePopulation(population, environment, seed: 0);
                var finalStats = population.GetStatistics();
                if (!history.Checkpoints.Any(c => c.Gen == generations))
                {
                    history.Checkpoints.Add((generations, finalStats.BestFitness, finalStats.MeanFitness));
                }
                _output.WriteLine($"Final Gen {generations}: Best={finalStats.BestFitness:F6}, Mean={finalStats.MeanFitness:F6}");
            }

            allRuns.Add(history);
        }

        // Summary
        _output.WriteLine("\n\n=== SUMMARY ===");
        foreach (var run in allRuns)
        {
            var final = run.Checkpoints.Last();
            if (run.SolvedAtGeneration.HasValue)
            {
                _output.WriteLine($"Seed {run.Seed}: SOLVED at gen {run.SolvedAtGeneration.Value} (Final Best={final.BestFitness:F6})");
            }
            else
            {
                _output.WriteLine($"Seed {run.Seed}: NOT SOLVED (Final Best={final.BestFitness:F6}, MSE={-final.BestFitness:F4})");
            }
        }

        var solvedCount = allRuns.Count(r => r.SolvedAtGeneration.HasValue);
        var avgFinalBest = allRuns.Average(r => r.Checkpoints.Last().BestFitness);
        var bestOverall = allRuns.Max(r => r.Checkpoints.Last().BestFitness);

        _output.WriteLine("");
        _output.WriteLine($"Solved runs: {solvedCount}/{seeds}");
        _output.WriteLine($"Average final best fitness: {avgFinalBest:F6} (MSE={-avgFinalBest:F4})");
        _output.WriteLine($"Best overall fitness: {bestOverall:F6} (MSE={-bestOverall:F4})");

        // Convergence analysis
        _output.WriteLine("\n=== CONVERGENCE ANALYSIS ===");
        var gen150Avg = allRuns.Average(r => r.Checkpoints.FirstOrDefault(c => c.Gen >= 150).BestFitness);
        var gen500Avg = allRuns.Average(r => r.Checkpoints.FirstOrDefault(c => c.Gen >= 500).BestFitness);
        var gen1000Avg = allRuns.Average(r => r.Checkpoints.FirstOrDefault(c => c.Gen >= 1000).BestFitness);
        var gen2000Avg = avgFinalBest;

        _output.WriteLine($"Gen 150  avg: {gen150Avg:F6}");
        _output.WriteLine($"Gen 500  avg: {gen500Avg:F6} (improvement from 150: {(gen500Avg - gen150Avg):+F6})");
        _output.WriteLine($"Gen 1000 avg: {gen1000Avg:F6} (improvement from 500: {(gen1000Avg - gen500Avg):+F6})");
        _output.WriteLine($"Gen 2000 avg: {gen2000Avg:F6} (improvement from 1000: {(gen2000Avg - gen1000Avg):+F6})");

        var improvement500to2000 = gen2000Avg - gen500Avg;
        _output.WriteLine($"\nTotal improvement from gen 500 to 2000: {improvement500to2000:+F6}");

        if (Math.Abs(improvement500to2000) < 0.01f)
        {
            _output.WriteLine("⚠ Evolution appears to have PLATEAUED (< 0.01 improvement in 1500 generations)");
        }

        // We don't fail the test - this is exploratory
        _output.WriteLine("\n✓ Long-run convergence test complete!");
    }

    private static int ComputeTopologyHash(SpeciesSpec topology)
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + topology.TotalNodes;
            hash = hash * 31 + topology.TotalEdges;

            // Hash edge structure (source -> dest pairs)
            foreach (var edge in topology.Edges.OrderBy(e => e.Source).ThenBy(e => e.Dest))
            {
                hash = hash * 31 + edge.Source;
                hash = hash * 31 + edge.Dest;
            }

            return hash;
        }
    }

    private static float ComputeWeightVariance(Species species)
    {
        if (species.Individuals.Count == 0)
            return 0f;

        // Sample weights from first individual in species
        var firstIndividual = species.Individuals[0];
        if (firstIndividual.Weights.Length == 0)
            return 0f;

        // Compute variance across all individuals for first few weights (sample)
        int sampleSize = Math.Min(10, firstIndividual.Weights.Length);
        float totalVariance = 0f;

        for (int w = 0; w < sampleSize; w++)
        {
            float mean = species.Individuals.Average(ind => ind.Weights[w]);
            float variance = species.Individuals.Average(ind =>
            {
                float diff = ind.Weights[w] - mean;
                return diff * diff;
            });
            totalVariance += variance;
        }

        return totalVariance / sampleSize;
    }

    private class RunHistory
    {
        public int Seed { get; set; }
        public List<(int Gen, float BestFitness, float MeanFitness)> Checkpoints { get; } = new();
        public int? SolvedAtGeneration { get; set; }
    }
}
