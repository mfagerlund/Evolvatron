using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Systematic CEM parameter sweep on DPNV 625-grid.
/// Sequential optimization: tune one parameter at a time, carry best forward.
/// Goal: find CEM config that maximizes 625-grid generalization.
/// GA baseline: 5→4→4→3 median 314/625 (range 195-356).
/// </summary>
public class CEMParameterSweep
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

    private record SweepResult(string Label, int GridScore, int SolveGen, double SolveTime, int TotalGens);

    /// <summary>
    /// Run one CEM config and return its best grid score.
    /// </summary>
    private static SweepResult RunConfig(
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

        for (int gen = 0; gen < 5000; gen++)
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

            if (sw.Elapsed.TotalSeconds > budgetSeconds) break;
        }

        // Evaluate champion on 625-grid
        int gridScore = 0;
        if (solveGen >= 0)
        {
            var (mu, _) = optimizer.GetBestSolution();
            gridScore = evaluator.EvaluateChampionGridScore(mu);
        }

        string label = $"isl={config.IslandCount},iSig={config.InitialSigma:F2}," +
                        $"mSig={config.MinSigma:F3},sSm={config.CEMSigmaSmoothing:F1}," +
                        $"mSm={config.CEMMuSmoothing:F1},eFr={config.CEMEliteFraction:F2}," +
                        $"stag={config.StagnationThreshold}";

        return new SweepResult(label, gridScore, solveGen, solveTime, totalGens);
    }

    /// <summary>
    /// Phase A-F: Sequential parameter optimization.
    /// Each phase sweeps one parameter while holding the rest at their best-known values.
    /// </summary>
    [Fact]
    public void FullParameterSweep()
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

        // Starting config (current defaults)
        float bestInitSigma = 0.3f;
        float bestMinSigma = 0.02f;
        float bestSigmaSmoothing = 0.4f;
        float bestMuSmoothing = 0.0f;
        float bestEliteFraction = 0.1f;
        int bestIslandCount = 1;
        int bestStagnation = 30;

        // === Phase A: InitialSigma ===
        Console.WriteLine("=== Phase A: InitialSigma ===");
        var sigmaValues = new[] { 0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 1.0f, 2.0f };
        int bestScore = 0;
        foreach (var sigma in sigmaValues)
        {
            var config = MakeConfig(bestIslandCount, sigma, bestMinSigma, bestSigmaSmoothing,
                bestMuSmoothing, bestEliteFraction, bestStagnation);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 30);
            Console.WriteLine($"  InitSigma={sigma:F2}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
            if (result.GridScore > bestScore) { bestScore = result.GridScore; bestInitSigma = sigma; }
        }
        Console.WriteLine($"  >> Best InitialSigma={bestInitSigma:F2} (grid={bestScore})\n");

        // === Phase B: MinSigma ===
        Console.WriteLine("=== Phase B: MinSigma ===");
        var minSigmaValues = new[] { 0.001f, 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.15f };
        bestScore = 0;
        foreach (var ms in minSigmaValues)
        {
            var config = MakeConfig(bestIslandCount, bestInitSigma, ms, bestSigmaSmoothing,
                bestMuSmoothing, bestEliteFraction, bestStagnation);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 30);
            Console.WriteLine($"  MinSigma={ms:F3}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
            if (result.GridScore > bestScore) { bestScore = result.GridScore; bestMinSigma = ms; }
        }
        Console.WriteLine($"  >> Best MinSigma={bestMinSigma:F3} (grid={bestScore})\n");

        // === Phase C: SigmaSmoothing ===
        Console.WriteLine("=== Phase C: SigmaSmoothing ===");
        var ssValues = new[] { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
        bestScore = 0;
        foreach (var ss in ssValues)
        {
            var config = MakeConfig(bestIslandCount, bestInitSigma, bestMinSigma, ss,
                bestMuSmoothing, bestEliteFraction, bestStagnation);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 30);
            Console.WriteLine($"  SigmaSmoothing={ss:F1}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
            if (result.GridScore > bestScore) { bestScore = result.GridScore; bestSigmaSmoothing = ss; }
        }
        Console.WriteLine($"  >> Best SigmaSmoothing={bestSigmaSmoothing:F1} (grid={bestScore})\n");

        // === Phase D: EliteFraction ===
        Console.WriteLine("=== Phase D: EliteFraction ===");
        var efValues = new[] { 0.01f, 0.02f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.5f };
        bestScore = 0;
        foreach (var ef in efValues)
        {
            var config = MakeConfig(bestIslandCount, bestInitSigma, bestMinSigma, bestSigmaSmoothing,
                bestMuSmoothing, ef, bestStagnation);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 30);
            Console.WriteLine($"  EliteFraction={ef:F2}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
            if (result.GridScore > bestScore) { bestScore = result.GridScore; bestEliteFraction = ef; }
        }
        Console.WriteLine($"  >> Best EliteFraction={bestEliteFraction:F2} (grid={bestScore})\n");

        // === Phase E: MuSmoothing ===
        Console.WriteLine("=== Phase E: MuSmoothing ===");
        var msValues = new[] { 0.0f, 0.1f, 0.2f, 0.3f, 0.5f, 0.7f };
        bestScore = 0;
        foreach (var mu in msValues)
        {
            var config = MakeConfig(bestIslandCount, bestInitSigma, bestMinSigma, bestSigmaSmoothing,
                mu, bestEliteFraction, bestStagnation);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 30);
            Console.WriteLine($"  MuSmoothing={mu:F1}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
            if (result.GridScore > bestScore) { bestScore = result.GridScore; bestMuSmoothing = mu; }
        }
        Console.WriteLine($"  >> Best MuSmoothing={bestMuSmoothing:F1} (grid={bestScore})\n");

        // === Phase F: IslandCount × StagnationThreshold ===
        Console.WriteLine("=== Phase F: IslandCount x StagnationThreshold ===");
        var islandCounts = new[] { 1, 3, 5, 8 };
        var stagnations = new[] { 10, 20, 30, 50 };
        bestScore = 0;
        foreach (var ic in islandCounts)
        foreach (var st in stagnations)
        {
            var config = MakeConfig(ic, bestInitSigma, bestMinSigma, bestSigmaSmoothing,
                bestMuSmoothing, bestEliteFraction, st);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 45);
            Console.WriteLine($"  Islands={ic},Stag={st}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
            if (result.GridScore > bestScore) { bestScore = result.GridScore; bestIslandCount = ic; bestStagnation = st; }
        }
        Console.WriteLine($"  >> Best Islands={bestIslandCount},Stag={bestStagnation} (grid={bestScore})\n");

        // === Phase G: Final validation with 5 seeds ===
        Console.WriteLine("=== Phase G: Final Validation (5 seeds, 100s budget) ===");
        var bestConfig = MakeConfig(bestIslandCount, bestInitSigma, bestMinSigma, bestSigmaSmoothing,
            bestMuSmoothing, bestEliteFraction, bestStagnation);
        Console.WriteLine($"Config: {bestConfig.IslandCount} islands, InitSigma={bestInitSigma:F2}, " +
                          $"MinSigma={bestMinSigma:F3}, SigmaSmooth={bestSigmaSmoothing:F1}, " +
                          $"MuSmooth={bestMuSmoothing:F1}, Elite={bestEliteFraction:F2}, Stag={bestStagnation}");

        var gridScores = new List<int>();
        for (int seed = 0; seed < 5; seed++)
        {
            var result = RunConfig(evaluator, topology, gpuCapacity, bestConfig, seed, budgetSeconds: 100);
            gridScores.Add(result.GridScore);
            Console.WriteLine($"  Seed {seed}: grid={result.GridScore}/625, " +
                              $"solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
        }

        gridScores.Sort();
        int median = gridScores[gridScores.Count / 2];
        Console.WriteLine($"\n=== FINAL RESULT ===");
        Console.WriteLine($"Median grid: {median}/625 (range {gridScores.Min()}-{gridScores.Max()})");
        Console.WriteLine($"GA baseline: median 314/625 (range 195-356)");
        Console.WriteLine($"Pass >=200: {gridScores.Count(s => s >= 200)}/5");
    }

    /// <summary>
    /// Same sweep but on 5→8→3 topology (the GA's most reliable topology).
    /// </summary>
    [Fact]
    public void FullParameterSweep_5_8_3()
    {
        var topology = DenseTopology.ForDPNV(new[] { 8 }, contextSize: 2);
        Console.WriteLine($"Topology: {topology}");

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;

        // Use best config from 5→4→4→3 sweep as starting point, re-sweep key params
        Console.WriteLine("=== Quick sweep: InitialSigma x MinSigma x SigmaSmoothing ===");
        var sigmas = new[] { 0.1f, 0.3f, 0.5f, 1.0f };
        var minSigmas = new[] { 0.01f, 0.05f, 0.1f, 0.15f };
        var smoothings = new[] { 0.0f, 0.3f, 0.5f, 0.7f, 0.9f };

        int bestScore = 0;
        float bestIS = 0.3f, bestMS = 0.02f, bestSS = 0.4f;

        foreach (var iSig in sigmas)
        foreach (var mSig in minSigmas)
        foreach (var ss in smoothings)
        {
            var config = MakeConfig(1, iSig, mSig, ss, 0f, 0.1f, 30);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 42, budgetSeconds: 20);
            if (result.GridScore > bestScore)
            {
                bestScore = result.GridScore;
                bestIS = iSig; bestMS = mSig; bestSS = ss;
                Console.WriteLine($"  NEW BEST: iSig={iSig:F1},mSig={mSig:F3},ss={ss:F1} → grid={result.GridScore}/625");
            }
        }

        Console.WriteLine($"\nBest 5→8→3 config: InitSigma={bestIS:F1}, MinSigma={bestMS:F3}, SigmaSmooth={bestSS:F1} (grid={bestScore})");

        // Validate with 5 seeds
        Console.WriteLine("\n=== Validation (5 seeds, 100s) ===");
        var gridScores = new List<int>();
        for (int seed = 0; seed < 5; seed++)
        {
            var config = MakeConfig(1, bestIS, bestMS, bestSS, 0f, 0.1f, 30);
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed, budgetSeconds: 100);
            gridScores.Add(result.GridScore);
            Console.WriteLine($"  Seed {seed}: grid={result.GridScore}/625, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
        }

        gridScores.Sort();
        int median = gridScores[gridScores.Count / 2];
        Console.WriteLine($"\nMedian: {median}/625 (range {gridScores.Min()}-{gridScores.Max()})");
        Console.WriteLine($"GA baseline: median 287/625 (range 228-418)");
    }

    private static IslandConfig MakeConfig(int islandCount, float initSigma, float minSigma,
        float sigmaSmoothing, float muSmoothing, float eliteFraction, int stagnation)
    {
        return new IslandConfig
        {
            IslandCount = islandCount,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = initSigma,
            MinSigma = minSigma,
            MaxSigma = 2.0f,
            CEMEliteFraction = eliteFraction,
            CEMSigmaSmoothing = sigmaSmoothing,
            CEMMuSmoothing = muSmoothing,
            StagnationThreshold = stagnation,
            ReinitSigma = initSigma,
        };
    }
}
