using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// OpenAI-style ES parameter sweep on DPNV 625-grid.
/// ES key advantage: sigma is FIXED (never collapses), acting as implicit regularization.
/// Key parameters: ESSigma, ESLearningRate, Adam betas.
/// </summary>
public class ESParameterSweep
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

    private record SweepResult(int BestGridScore, int FinalGridScore,
        int BestGridGen, int SolveGen, double SolveTime, int TotalGens);

    private static SweepResult RunESConfig(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        int seed,
        double budgetSeconds)
    {
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(seed);
        var sw = Stopwatch.StartNew();

        int solveGen = -1;
        double solveTime = -1;
        int totalGens = 0;
        int bestGrid = 0;
        int bestGridGen = -1;
        int finalGrid = 0;
        double lastGridCheck = 0;

        for (int gen = 0; gen < 10000; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, solvedCount) = evaluator.EvaluatePopulation(paramVectors, optimizer.TotalPopulation);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);
            totalGens = gen + 1;

            if (solveGen < 0 && solvedCount > 0)
            {
                solveGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
            }

            double elapsed = sw.Elapsed.TotalSeconds;
            if (solveGen >= 0 && elapsed - lastGridCheck >= 3.0)
            {
                lastGridCheck = elapsed;
                var (mu, _) = optimizer.GetBestSolution();
                int gridScore = evaluator.EvaluateChampionGridScore(mu);
                finalGrid = gridScore;
                if (gridScore > bestGrid)
                {
                    bestGrid = gridScore;
                    bestGridGen = gen;
                }
            }

            if (elapsed > budgetSeconds) break;
        }

        if (solveGen >= 0)
        {
            var (mu, _) = optimizer.GetBestSolution();
            finalGrid = evaluator.EvaluateChampionGridScore(mu);
            if (finalGrid > bestGrid)
            {
                bestGrid = finalGrid;
                bestGridGen = totalGens;
            }
        }

        return new SweepResult(bestGrid, finalGrid, bestGridGen, solveGen, solveTime, totalGens);
    }

    private static int RunMultiSeed(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        double budgetSeconds,
        int numSeeds = 2)
    {
        int totalBest = 0;
        for (int seed = 0; seed < numSeeds; seed++)
        {
            var result = RunESConfig(evaluator, topology, gpuCapacity, config, seed, budgetSeconds);
            totalBest += result.BestGridScore;
        }
        return totalBest / numSeeds;
    }

    /// <summary>
    /// Systematic ES sweep on 5→4→4→3 DPNV.
    /// Key params: ESSigma (noise scale), ESLearningRate, Adam betas.
    /// ES sigma is fixed (never collapses) — fundamentally different from CEM.
    /// </summary>
    [Fact]
    public void ES_ParameterSweep_5443()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        Console.WriteLine($"Topology: {topology}");

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"GPU capacity: {gpuCapacity}");
        Console.WriteLine();

        // === Phase A: ESSigma (noise scale) ===
        Console.WriteLine("=== Phase A: ESSigma ===");
        var sigmaValues = new[] { 0.01f, 0.02f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.5f };
        float bestSigma = 0.05f;
        int bestScore = 0;

        foreach (var sigma in sigmaValues)
        {
            var config = MakeESConfig(sigma, 0.01f, 0.9f, 0.999f);
            int avg = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 30);
            Console.WriteLine($"  ESSigma={sigma:F3}: avg best grid={avg}/625");
            if (avg > bestScore) { bestScore = avg; bestSigma = sigma; }
        }
        Console.WriteLine($"  >> Best ESSigma={bestSigma:F3} (grid={bestScore})\n");

        // === Phase B: Learning Rate ===
        Console.WriteLine("=== Phase B: ESLearningRate ===");
        var lrValues = new[] { 0.001f, 0.003f, 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f };
        float bestLR = 0.01f;
        bestScore = 0;

        foreach (var lr in lrValues)
        {
            var config = MakeESConfig(bestSigma, lr, 0.9f, 0.999f);
            int avg = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 30);
            Console.WriteLine($"  LR={lr:F4}: avg best grid={avg}/625");
            if (avg > bestScore) { bestScore = avg; bestLR = lr; }
        }
        Console.WriteLine($"  >> Best LR={bestLR:F4} (grid={bestScore})\n");

        // === Phase C: Adam Beta1 ===
        Console.WriteLine("=== Phase C: AdamBeta1 ===");
        var beta1Values = new[] { 0.0f, 0.5f, 0.8f, 0.9f, 0.95f, 0.99f };
        float bestBeta1 = 0.9f;
        bestScore = 0;

        foreach (var b1 in beta1Values)
        {
            var config = MakeESConfig(bestSigma, bestLR, b1, 0.999f);
            int avg = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 30);
            Console.WriteLine($"  Beta1={b1:F2}: avg best grid={avg}/625");
            if (avg > bestScore) { bestScore = avg; bestBeta1 = b1; }
        }
        Console.WriteLine($"  >> Best Beta1={bestBeta1:F2} (grid={bestScore})\n");

        // === Phase D: Adam Beta2 ===
        Console.WriteLine("=== Phase D: AdamBeta2 ===");
        var beta2Values = new[] { 0.9f, 0.99f, 0.999f, 0.9999f };
        float bestBeta2 = 0.999f;
        bestScore = 0;

        foreach (var b2 in beta2Values)
        {
            var config = MakeESConfig(bestSigma, bestLR, bestBeta1, b2);
            int avg = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 30);
            Console.WriteLine($"  Beta2={b2:F4}: avg best grid={avg}/625");
            if (avg > bestScore) { bestScore = avg; bestBeta2 = b2; }
        }
        Console.WriteLine($"  >> Best Beta2={bestBeta2:F4} (grid={bestScore})\n");

        // === Phase E: Final validation (10 seeds, 100s) ===
        Console.WriteLine("=== Phase E: Final Validation (10 seeds, 100s budget) ===");
        var bestConfig = MakeESConfig(bestSigma, bestLR, bestBeta1, bestBeta2);
        Console.WriteLine($"Config: ESSigma={bestSigma:F3}, LR={bestLR:F4}, " +
                          $"Beta1={bestBeta1:F2}, Beta2={bestBeta2:F4}");

        var bestGridScores = new List<int>();
        var finalGridScores = new List<int>();

        for (int seed = 0; seed < 10; seed++)
        {
            var result = RunESConfig(evaluator, topology, gpuCapacity, bestConfig, seed, budgetSeconds: 100);
            bestGridScores.Add(result.BestGridScore);
            finalGridScores.Add(result.FinalGridScore);
            Console.WriteLine($"  Seed {seed}: best={result.BestGridScore}/625 @gen{result.BestGridGen}, " +
                              $"final={result.FinalGridScore}/625, solve=gen{result.SolveGen}/{result.SolveTime:F1}s, " +
                              $"gens={result.TotalGens}");
        }

        bestGridScores.Sort();
        finalGridScores.Sort();
        int bestMedian = bestGridScores[bestGridScores.Count / 2];
        int finalMedian = finalGridScores[finalGridScores.Count / 2];

        Console.WriteLine($"\n=== FINAL RESULT (ES, 5→4→4→3) ===");
        Console.WriteLine($"Best grid (tracked): median={bestMedian}/625 (range {bestGridScores.Min()}-{bestGridScores.Max()})");
        Console.WriteLine($"Final grid (end):    median={finalMedian}/625 (range {finalGridScores.Min()}-{finalGridScores.Max()})");
        Console.WriteLine($"CEM tracked best:    median=223/625 (range 214-231)");
        Console.WriteLine($"GA baseline:         median=314/625 (range 195-356)");
        Console.WriteLine($"Pass >=200 (best):   {bestGridScores.Count(s => s >= 200)}/10");
        Console.WriteLine($"Pass >=200 (final):  {finalGridScores.Count(s => s >= 200)}/10");
    }

    /// <summary>
    /// Same sweep for 5→8→3.
    /// </summary>
    [Fact]
    public void ES_ParameterSweep_583()
    {
        var topology = DenseTopology.ForDPNV(new[] { 8 }, contextSize: 2);
        Console.WriteLine($"Topology: {topology}");

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"GPU capacity: {gpuCapacity}");
        Console.WriteLine();

        // Quick grid: ESSigma × LearningRate
        Console.WriteLine("=== Grid: ESSigma × LearningRate ===");
        var sigmas = new[] { 0.02f, 0.05f, 0.1f, 0.15f, 0.3f };
        var lrs = new[] { 0.005f, 0.01f, 0.02f, 0.05f, 0.1f };

        float bestSigma = 0.05f, bestLR = 0.01f;
        int bestScore = 0;

        foreach (var sigma in sigmas)
        foreach (var lr in lrs)
        {
            var config = MakeESConfig(sigma, lr, 0.9f, 0.999f);
            int avg = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 25);
            if (avg > bestScore)
            {
                bestScore = avg;
                bestSigma = sigma; bestLR = lr;
                Console.WriteLine($"  NEW BEST: sigma={sigma:F3},lr={lr:F3} → avg best grid={avg}/625");
            }
        }
        Console.WriteLine($"\nBest: ESSigma={bestSigma:F3}, LR={bestLR:F3} (grid={bestScore})\n");

        // Validation (10 seeds, 100s)
        Console.WriteLine("=== Validation (10 seeds, 100s) ===");
        var bestConfig = MakeESConfig(bestSigma, bestLR, 0.9f, 0.999f);

        var bestGridScores = new List<int>();
        var finalGridScores = new List<int>();

        for (int seed = 0; seed < 10; seed++)
        {
            var result = RunESConfig(evaluator, topology, gpuCapacity, bestConfig, seed, budgetSeconds: 100);
            bestGridScores.Add(result.BestGridScore);
            finalGridScores.Add(result.FinalGridScore);
            Console.WriteLine($"  Seed {seed}: best={result.BestGridScore}/625 @gen{result.BestGridGen}, " +
                              $"final={result.FinalGridScore}/625, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
        }

        bestGridScores.Sort();
        finalGridScores.Sort();
        int bestMedian = bestGridScores[bestGridScores.Count / 2];
        int finalMedian = finalGridScores[finalGridScores.Count / 2];

        Console.WriteLine($"\n=== FINAL RESULT (ES, 5→8→3) ===");
        Console.WriteLine($"Best grid (tracked): median={bestMedian}/625 (range {bestGridScores.Min()}-{bestGridScores.Max()})");
        Console.WriteLine($"Final grid (end):    median={finalMedian}/625 (range {finalGridScores.Min()}-{finalGridScores.Max()})");
        Console.WriteLine($"CEM tracked best:    median=82/625 (range 25-181)");
        Console.WriteLine($"GA baseline:         median=287/625 (range 228-418)");
        Console.WriteLine($"Pass >=200 (best):   {bestGridScores.Count(s => s >= 200)}/10");
    }

    private static IslandConfig MakeESConfig(float sigma, float lr, float beta1, float beta2)
    {
        return new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.ES,
            ESSigma = sigma,
            ESLearningRate = lr,
            ESAdamBeta1 = beta1,
            ESAdamBeta2 = beta2,
            InitialSigma = 0.25f,
            StagnationThreshold = 9999,
        };
    }
}
