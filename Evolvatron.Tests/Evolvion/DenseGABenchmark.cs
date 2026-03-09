using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// GA on the dense NN kernel — apples-to-apples comparison with CEM/ES.
/// Same GPUDenseDoublePoleEvaluator, same population, same time budget.
/// Only difference: optimization algorithm (tournament + jitter vs distribution fitting).
/// </summary>
public class DenseGABenchmark
{
    private static float[][] Build625Grid()
    {
        float[] cartPositions = { -2.16f, -1.08f, 0f, 1.08f, 2.16f };
        float[] cartVelocities = { -1.35f, -0.675f, 0f, 0.675f, 1.35f };
        float poleAngleStep = 3.6f * (MathF.PI / 180f);
        float[] poleAngles = { -2 * poleAngleStep, -poleAngleStep, 0, poleAngleStep, 2 * poleAngleStep };
        float poleVelStep = (8.6f * MathF.PI / 180f);
        float[] poleVelocities = { -2 * poleVelStep, -poleVelStep, 0, poleVelStep, 2 * poleVelStep };

        var positions = new List<float[]>();
        foreach (var cp in cartPositions)
        foreach (var cv in cartVelocities)
        foreach (var pa in poleAngles)
        foreach (var pv in poleVelocities)
            positions.Add(new[] { cp, cv, pa, pv, 0f, 0f });

        return positions.ToArray();
    }

    private record RunResult(int BestGridScore, int FinalGridScore,
        int BestGridGen, int SolveGen, double SolveTime, int TotalGens);

    private static RunResult RunGA(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int popSize,
        float jitterStdDev,
        int seed,
        double budgetSeconds)
    {
        var ga = new DenseGAOptimizer(topology, popSize, seed)
        {
            JitterStdDev = jitterStdDev,
            EliteCount = 10,
            TournamentSize = 5,
            ParentPoolFraction = 0.5f,
        };
        var rng = new Random(seed);
        var sw = Stopwatch.StartNew();

        int solveGen = -1;
        double solveTime = -1;
        int totalGens = 0;
        int bestGrid = 0;
        int bestGridGen = -1;
        int finalGrid = 0;
        double lastGridCheck = 0;

        for (int gen = 0; gen < 50000; gen++)
        {
            var paramVectors = ga.GetParamVectors();
            var (fitness, solvedCount) = evaluator.EvaluatePopulation(paramVectors, popSize);
            ga.Update(fitness);

            if (solveGen < 0 && solvedCount > 0)
            {
                solveGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
            }

            double elapsed = sw.Elapsed.TotalSeconds;
            if (solveGen >= 0 && elapsed - lastGridCheck >= 3.0)
            {
                lastGridCheck = elapsed;
                var (mu, _) = ga.GetBest();
                int gridScore = evaluator.EvaluateChampionGridScore(mu);
                finalGrid = gridScore;
                if (gridScore > bestGrid)
                {
                    bestGrid = gridScore;
                    bestGridGen = gen;
                }
            }

            ga.StepGeneration(rng);
            totalGens = gen + 1;

            if (elapsed > budgetSeconds) break;
        }

        if (solveGen >= 0)
        {
            var (mu, _) = ga.GetBest();
            finalGrid = evaluator.EvaluateChampionGridScore(mu);
            if (finalGrid > bestGrid)
            {
                bestGrid = finalGrid;
                bestGridGen = totalGens;
            }
        }

        return new RunResult(bestGrid, finalGrid, bestGridGen, solveGen, solveTime, totalGens);
    }

    /// <summary>
    /// Jitter sweep: find optimal WeightJitterStdDev for dense kernel.
    /// </summary>
    [Fact]
    public void DenseGA_JitterSweep_5443()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        Console.WriteLine($"Topology: {topology}");

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int popSize = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Population: {popSize}");
        Console.WriteLine();

        Console.WriteLine("=== Jitter Sweep (2 seeds, 30s budget, tracked best grid) ===");
        var jitters = new[] { 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.40f, 0.50f };
        float bestJitter = 0.15f;
        int bestScore = 0;

        foreach (var j in jitters)
        {
            int totalBest = 0;
            for (int seed = 0; seed < 2; seed++)
            {
                var r = RunGA(evaluator, topology, popSize, j, seed, budgetSeconds: 30);
                totalBest += r.BestGridScore;
            }
            int avg = totalBest / 2;
            Console.WriteLine($"  Jitter={j:F2}: avg best grid={avg}/625");
            if (avg > bestScore) { bestScore = avg; bestJitter = j; }
        }
        Console.WriteLine($"  >> Best Jitter={bestJitter:F2} (grid={bestScore})\n");

        // Validation (10 seeds, 100s)
        Console.WriteLine($"=== Validation (10 seeds, 100s, jitter={bestJitter:F2}) ===");
        var bestGridScores = new List<int>();
        var finalGridScores = new List<int>();

        for (int seed = 0; seed < 10; seed++)
        {
            var r = RunGA(evaluator, topology, popSize, bestJitter, seed, budgetSeconds: 100);
            bestGridScores.Add(r.BestGridScore);
            finalGridScores.Add(r.FinalGridScore);
            Console.WriteLine($"  Seed {seed}: best={r.BestGridScore}/625 @gen{r.BestGridGen}, " +
                              $"final={r.FinalGridScore}/625, solve=gen{r.SolveGen}/{r.SolveTime:F1}s, " +
                              $"gens={r.TotalGens}");
        }

        bestGridScores.Sort();
        finalGridScores.Sort();
        int bestMedian = bestGridScores[bestGridScores.Count / 2];
        int finalMedian = finalGridScores[finalGridScores.Count / 2];

        Console.WriteLine($"\n=== RESULT (Dense GA, 5→4→4→3, jitter={bestJitter:F2}) ===");
        Console.WriteLine($"Best grid (tracked): median={bestMedian}/625 (range {bestGridScores.Min()}-{bestGridScores.Max()})");
        Console.WriteLine($"Final grid (end):    median={finalMedian}/625 (range {finalGridScores.Min()}-{finalGridScores.Max()})");
        Console.WriteLine($"GA (sparse kernel):  median=314/625 (range 195-356)");
        Console.WriteLine($"CEM (tracked):       median=223/625 (range 214-231)");
        Console.WriteLine($"ES (tracked):        median=164/625 (range 143-202)");
        Console.WriteLine($"Pass >=200 (best):   {bestGridScores.Count(s => s >= 200)}/10");
        Console.WriteLine($"Pass >=200 (final):  {finalGridScores.Count(s => s >= 200)}/10");
    }

    /// <summary>
    /// Same for 5→8→3.
    /// </summary>
    [Fact]
    public void DenseGA_JitterSweep_583()
    {
        var topology = DenseTopology.ForDPNV(new[] { 8 }, contextSize: 2);
        Console.WriteLine($"Topology: {topology}");

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int popSize = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Population: {popSize}");
        Console.WriteLine();

        Console.WriteLine("=== Jitter Sweep (2 seeds, 30s budget, tracked best grid) ===");
        var jitters = new[] { 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.40f, 0.50f };
        float bestJitter = 0.15f;
        int bestScore = 0;

        foreach (var j in jitters)
        {
            int totalBest = 0;
            for (int seed = 0; seed < 2; seed++)
            {
                var r = RunGA(evaluator, topology, popSize, j, seed, budgetSeconds: 30);
                totalBest += r.BestGridScore;
            }
            int avg = totalBest / 2;
            Console.WriteLine($"  Jitter={j:F2}: avg best grid={avg}/625");
            if (avg > bestScore) { bestScore = avg; bestJitter = j; }
        }
        Console.WriteLine($"  >> Best Jitter={bestJitter:F2} (grid={bestScore})\n");

        // Validation (10 seeds, 100s)
        Console.WriteLine($"=== Validation (10 seeds, 100s, jitter={bestJitter:F2}) ===");
        var bestGridScores = new List<int>();
        var finalGridScores = new List<int>();

        for (int seed = 0; seed < 10; seed++)
        {
            var r = RunGA(evaluator, topology, popSize, bestJitter, seed, budgetSeconds: 100);
            bestGridScores.Add(r.BestGridScore);
            finalGridScores.Add(r.FinalGridScore);
            Console.WriteLine($"  Seed {seed}: best={r.BestGridScore}/625 @gen{r.BestGridGen}, " +
                              $"final={r.FinalGridScore}/625, solve=gen{r.SolveGen}/{r.SolveTime:F1}s, " +
                              $"gens={r.TotalGens}");
        }

        bestGridScores.Sort();
        finalGridScores.Sort();
        int bestMedian = bestGridScores[bestGridScores.Count / 2];
        int finalMedian = finalGridScores[finalGridScores.Count / 2];

        Console.WriteLine($"\n=== RESULT (Dense GA, 5→8→3, jitter={bestJitter:F2}) ===");
        Console.WriteLine($"Best grid (tracked): median={bestMedian}/625 (range {bestGridScores.Min()}-{bestGridScores.Max()})");
        Console.WriteLine($"Final grid (end):    median={finalMedian}/625 (range {finalGridScores.Min()}-{finalGridScores.Max()})");
        Console.WriteLine($"GA (sparse kernel):  median=287/625 (range 228-418)");
        Console.WriteLine($"CEM (tracked):       median=82/625 (range 25-181)");
        Console.WriteLine($"ES (tracked):        median=198/625 (range 143-223)");
        Console.WriteLine($"Pass >=200 (best):   {bestGridScores.Count(s => s >= 200)}/10");
        Console.WriteLine($"Pass >=200 (final):  {finalGridScores.Count(s => s >= 200)}/10");
    }
}
