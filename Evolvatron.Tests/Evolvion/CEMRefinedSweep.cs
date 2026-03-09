using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Refined CEM sweep with two key improvements over CEMParameterSweep:
/// 1. Tracks best grid score over entire training (not just final champion)
/// 2. Uses 2 seeds per config to reduce noise
/// 3. Tests algorithmic variants (sigma floor boost, warm restarts)
/// </summary>
public class CEMRefinedSweep
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

    private record SweepResult(string Label, int BestGridScore, int FinalGridScore,
        int BestGridGen, int SolveGen, double SolveTime, int TotalGens);

    /// <summary>
    /// Run one CEM config, tracking best grid score throughout training.
    /// Checks grid every 3s after first solve.
    /// </summary>
    private static SweepResult RunConfigTracked(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        int seed,
        double budgetSeconds,
        int warmRestartEveryGens = 0,
        float warmRestartSigmaBump = 3.0f)
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

            // Warm restart: periodically boost sigma
            if (warmRestartEveryGens > 0 && gen > 0 && gen % warmRestartEveryGens == 0 && solveGen >= 0)
            {
                optimizer.WarmStart(warmRestartSigmaBump);
            }

            // Check 625-grid every 3s after first solve
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

        // Final grid check
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

        string label = $"isl={config.IslandCount},iSig={config.InitialSigma:F2}," +
                        $"mSig={config.MinSigma:F3},sSm={config.CEMSigmaSmoothing:F1}," +
                        $"mSm={config.CEMMuSmoothing:F1},eFr={config.CEMEliteFraction:F2}," +
                        $"stag={config.StagnationThreshold}";

        return new SweepResult(label, bestGrid, finalGrid, bestGridGen, solveGen, solveTime, totalGens);
    }

    private static int RunMultiSeed(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        double budgetSeconds,
        int numSeeds = 2,
        int warmRestartEveryGens = 0,
        float warmRestartSigmaBump = 3.0f)
    {
        int totalBest = 0;
        for (int seed = 0; seed < numSeeds; seed++)
        {
            var result = RunConfigTracked(evaluator, topology, gpuCapacity, config, seed,
                budgetSeconds, warmRestartEveryGens, warmRestartSigmaBump);
            totalBest += result.BestGridScore;
        }
        return totalBest / numSeeds;
    }

    /// <summary>
    /// Part 1: Fine-grid search around the best config from initial sweep.
    /// Best from initial: InitSigma=0.30, MinSigma=0.100, SigmaSmooth=0.3,
    /// MuSmooth=0.2, Elite=0.02, 1 island
    /// Uses 2 seeds and tracks best grid throughout training.
    /// </summary>
    [Fact]
    public void FineGrainedSweep_5443()
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

        // Fine grid around best found params
        // Key insight: track best grid THROUGHOUT training, not just at end
        Console.WriteLine("=== Fine search: InitSigma × MinSigma × EliteFraction ===");
        Console.WriteLine("(2 seeds each, 40s budget, tracking best grid throughout training)");

        var initSigmas = new[] { 0.25f, 0.30f, 0.40f };
        var minSigmas = new[] { 0.08f, 0.10f, 0.12f, 0.15f };
        var eliteFracs = new[] { 0.01f, 0.02f, 0.03f };

        int overallBest = 0;
        float bestIS = 0.3f, bestMS = 0.1f, bestEF = 0.02f;

        foreach (var iSig in initSigmas)
        foreach (var mSig in minSigmas)
        foreach (var ef in eliteFracs)
        {
            var config = new IslandConfig
            {
                IslandCount = 1,
                Strategy = UpdateStrategyType.CEM,
                InitialSigma = iSig,
                MinSigma = mSig,
                MaxSigma = 2.0f,
                CEMEliteFraction = ef,
                CEMSigmaSmoothing = 0.3f,
                CEMMuSmoothing = 0.2f,
                StagnationThreshold = 30,
            };

            int avgBest = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 30);

            if (avgBest > overallBest)
            {
                overallBest = avgBest;
                bestIS = iSig; bestMS = mSig; bestEF = ef;
                Console.WriteLine($"  NEW BEST: iSig={iSig:F2},mSig={mSig:F3},ef={ef:F2} → avg best grid={avgBest}/625");
            }
        }

        Console.WriteLine($"\nBest fine config: InitSigma={bestIS:F2}, MinSigma={bestMS:F3}, Elite={bestEF:F2} → {overallBest}/625");

        // === Part 2: Test warm restarts with best config ===
        Console.WriteLine("\n=== Warm restart variants ===");
        var warmIntervals = new[] { 0, 50, 100, 200 };
        var warmBumps = new[] { 2.0f, 3.0f };

        int bestWarmScore = overallBest;
        int bestWarmInterval = 0;
        float bestWarmBump = 1.0f;

        foreach (var interval in warmIntervals)
        foreach (var bump in warmBumps)
        {
            if (interval == 0 && bump > 2.0f) continue; // skip redundant 0-interval

            var config = new IslandConfig
            {
                IslandCount = 1,
                Strategy = UpdateStrategyType.CEM,
                InitialSigma = bestIS,
                MinSigma = bestMS,
                MaxSigma = 2.0f,
                CEMEliteFraction = bestEF,
                CEMSigmaSmoothing = 0.3f,
                CEMMuSmoothing = 0.2f,
                StagnationThreshold = 30,
            };

            int avgBest = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 40,
                warmRestartEveryGens: interval, warmRestartSigmaBump: bump);

            Console.WriteLine($"  interval={interval},bump={bump:F1}: avg best grid={avgBest}/625");

            if (avgBest > bestWarmScore)
            {
                bestWarmScore = avgBest;
                bestWarmInterval = interval;
                bestWarmBump = bump;
            }
        }

        Console.WriteLine($"\nBest warm: interval={bestWarmInterval},bump={bestWarmBump:F1} → {bestWarmScore}/625");

        // === Part 3: Final validation (10 seeds, 100s) ===
        Console.WriteLine("\n=== Final Validation (10 seeds, 100s budget) ===");
        var bestConfig = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = bestIS,
            MinSigma = bestMS,
            MaxSigma = 2.0f,
            CEMEliteFraction = bestEF,
            CEMSigmaSmoothing = 0.3f,
            CEMMuSmoothing = 0.2f,
            StagnationThreshold = 30,
        };

        Console.WriteLine($"Config: InitSigma={bestIS:F2}, MinSigma={bestMS:F3}, Elite={bestEF:F2}");
        if (bestWarmInterval > 0)
            Console.WriteLine($"WarmRestart: every {bestWarmInterval} gens, bump={bestWarmBump:F1}");

        var bestGridScores = new List<int>();
        var finalGridScores = new List<int>();

        for (int seed = 0; seed < 10; seed++)
        {
            var result = RunConfigTracked(evaluator, topology, gpuCapacity, bestConfig, seed,
                budgetSeconds: 100, warmRestartEveryGens: bestWarmInterval, warmRestartSigmaBump: bestWarmBump);
            bestGridScores.Add(result.BestGridScore);
            finalGridScores.Add(result.FinalGridScore);
            Console.WriteLine($"  Seed {seed}: best={result.BestGridScore}/625 @gen{result.BestGridGen}, " +
                              $"final={result.FinalGridScore}/625, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
        }

        bestGridScores.Sort();
        finalGridScores.Sort();
        int bestMedian = bestGridScores[bestGridScores.Count / 2];
        int finalMedian = finalGridScores[finalGridScores.Count / 2];

        Console.WriteLine($"\n=== FINAL RESULT (5→4→4→3) ===");
        Console.WriteLine($"Best grid (tracked): median={bestMedian}/625 (range {bestGridScores.Min()}-{bestGridScores.Max()})");
        Console.WriteLine($"Final grid (end):    median={finalMedian}/625 (range {finalGridScores.Min()}-{finalGridScores.Max()})");
        Console.WriteLine($"GA baseline:         median=314/625 (range 195-356)");
        Console.WriteLine($"Pass >=200 (best):   {bestGridScores.Count(s => s >= 200)}/10");
        Console.WriteLine($"Pass >=200 (final):  {finalGridScores.Count(s => s >= 200)}/10");
    }

    /// <summary>
    /// Same refined sweep for 5→8→3 topology.
    /// </summary>
    [Fact]
    public void FineGrainedSweep_583()
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

        // Fine grid search
        Console.WriteLine("=== Fine search: InitSigma × MinSigma × EliteFraction ===");
        Console.WriteLine("(2 seeds each, 40s budget, tracking best grid throughout training)");

        var initSigmas = new[] { 0.2f, 0.3f, 0.4f, 0.5f, 0.7f };
        var minSigmas = new[] { 0.05f, 0.08f, 0.10f, 0.12f, 0.15f, 0.20f };
        var eliteFracs = new[] { 0.01f, 0.02f, 0.03f, 0.05f };

        int overallBest = 0;
        float bestIS = 0.3f, bestMS = 0.1f, bestEF = 0.02f;

        foreach (var iSig in initSigmas)
        foreach (var mSig in minSigmas)
        foreach (var ef in eliteFracs)
        {
            var config = new IslandConfig
            {
                IslandCount = 1,
                Strategy = UpdateStrategyType.CEM,
                InitialSigma = iSig,
                MinSigma = mSig,
                MaxSigma = 2.0f,
                CEMEliteFraction = ef,
                CEMSigmaSmoothing = 0.3f,
                CEMMuSmoothing = 0.2f,
                StagnationThreshold = 30,
            };

            int avgBest = RunMultiSeed(evaluator, topology, gpuCapacity, config, budgetSeconds: 30);

            if (avgBest > overallBest)
            {
                overallBest = avgBest;
                bestIS = iSig; bestMS = mSig; bestEF = ef;
                Console.WriteLine($"  NEW BEST: iSig={iSig:F2},mSig={mSig:F3},ef={ef:F2} → avg best grid={avgBest}/625");
            }
        }

        Console.WriteLine($"\nBest fine config: InitSigma={bestIS:F2}, MinSigma={bestMS:F3}, Elite={bestEF:F2} → {overallBest}/625");

        // Final validation (10 seeds, 100s)
        Console.WriteLine("\n=== Final Validation (10 seeds, 100s budget) ===");
        var bestConfig = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = bestIS,
            MinSigma = bestMS,
            MaxSigma = 2.0f,
            CEMEliteFraction = bestEF,
            CEMSigmaSmoothing = 0.3f,
            CEMMuSmoothing = 0.2f,
            StagnationThreshold = 30,
        };

        Console.WriteLine($"Config: InitSigma={bestIS:F2}, MinSigma={bestMS:F3}, Elite={bestEF:F2}");

        var bestGridScores = new List<int>();
        var finalGridScores = new List<int>();

        for (int seed = 0; seed < 10; seed++)
        {
            var result = RunConfigTracked(evaluator, topology, gpuCapacity, bestConfig, seed, budgetSeconds: 100);
            bestGridScores.Add(result.BestGridScore);
            finalGridScores.Add(result.FinalGridScore);
            Console.WriteLine($"  Seed {seed}: best={result.BestGridScore}/625 @gen{result.BestGridGen}, " +
                              $"final={result.FinalGridScore}/625, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
        }

        bestGridScores.Sort();
        finalGridScores.Sort();
        int bestMedian = bestGridScores[bestGridScores.Count / 2];
        int finalMedian = finalGridScores[finalGridScores.Count / 2];

        Console.WriteLine($"\n=== FINAL RESULT (5→8→3) ===");
        Console.WriteLine($"Best grid (tracked): median={bestMedian}/625 (range {bestGridScores.Min()}-{bestGridScores.Max()})");
        Console.WriteLine($"Final grid (end):    median={finalMedian}/625 (range {finalGridScores.Min()}-{finalGridScores.Max()})");
        Console.WriteLine($"GA baseline:         median=287/625 (range 228-418)");
        Console.WriteLine($"Pass >=200 (best):   {bestGridScores.Count(s => s >= 200)}/10");
        Console.WriteLine($"Pass >=200 (final):  {finalGridScores.Count(s => s >= 200)}/10");
    }
}
