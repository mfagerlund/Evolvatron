using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Regularization experiments for CEM and ES on DPNV 625-grid.
/// Measures FINAL champion only (no mid-training tracking).
/// Early stopping when grid >= 200 (practical: stop when good enough).
/// Topology: 5→4→4→3 only.
/// Baseline: GA final=314/625, CEM final=112/625, ES final=21/625.
/// </summary>
public class RegularizationBenchmark
{
    private const int GRID_TARGET = 200;

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

    private record RunResult(
        int FinalGrid, int SolveGen, double SolveTime,
        double TimeToTarget, int TotalGens, bool HitTarget);

    /// <summary>
    /// Run one config measuring FINAL champion grid score.
    /// Early stops if grid >= target (practical stopping criterion).
    /// Grid checked every 3s after first solve.
    /// </summary>
    private static RunResult RunConfig(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        int seed,
        double budgetSeconds,
        bool earlyStop = true)
    {
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(seed);
        var sw = Stopwatch.StartNew();

        int solveGen = -1;
        double solveTime = -1;
        int totalGens = 0;
        double lastGridCheck = 0;
        int finalGrid = 0;
        double timeToTarget = -1;

        for (int gen = 0; gen < 100_000; gen++)
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
                finalGrid = evaluator.EvaluateChampionGridScore(mu);

                if (earlyStop && finalGrid >= GRID_TARGET)
                {
                    timeToTarget = elapsed;
                    break;
                }
            }

            if (elapsed > budgetSeconds) break;
        }

        // Final grid check if we didn't early-stop
        if (solveGen >= 0 && timeToTarget < 0)
        {
            var (mu, _) = optimizer.GetBestSolution();
            finalGrid = evaluator.EvaluateChampionGridScore(mu);
            if (finalGrid >= GRID_TARGET)
                timeToTarget = sw.Elapsed.TotalSeconds;
        }

        return new RunResult(finalGrid, solveGen, solveTime, timeToTarget, totalGens,
            timeToTarget >= 0);
    }

    private static int SweepAvgFinal(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        int numSeeds,
        double budgetSeconds)
    {
        int total = 0;
        for (int seed = 0; seed < numSeeds; seed++)
        {
            var result = RunConfig(evaluator, topology, gpuCapacity, config, seed,
                budgetSeconds, earlyStop: false);
            total += result.FinalGrid;
        }
        return total / numSeeds;
    }

    private static IslandConfig MakeCEMConfig(float weightDecay)
    {
        return new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.25f,
            MinSigma = 0.08f,
            MaxSigma = 2.0f,
            CEMEliteFraction = 0.01f,
            CEMSigmaSmoothing = 0.3f,
            CEMMuSmoothing = 0.2f,
            StagnationThreshold = 9999,
        };
    }

    private static IslandConfig MakeESConfig(float weightDecay)
    {
        return new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.ES,
            ESSigma = 0.05f,
            ESLearningRate = 0.05f,
            ESAdamBeta1 = 0.80f,
            ESAdamBeta2 = 0.999f,
            InitialSigma = 0.25f,
            StagnationThreshold = 9999,
        };
    }

    /// <summary>
    /// Quick smoke test: run one CEM config for 30s, evaluate grid at end.
    /// </summary>
    [Fact]
    public void SmokeTest()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");

        var config = MakeCEMConfig(0f);
        var result = RunConfig(evaluator, topology, gpuCapacity, config, seed: 0,
            budgetSeconds: 30, earlyStop: false);
        Console.WriteLine($"Final grid: {result.FinalGrid}/625, solve=gen{result.SolveGen}/{result.SolveTime:F1}s, gens={result.TotalGens}");
    }

    /// <summary>
    /// CEM + ES weight decay sweep in one test (avoids test host restart between phases).
    /// Tests whether L2 regularization prevents generalization decay.
    /// 2 seeds, 45s budget, NO early stopping (want to see if final stays good).
    /// </summary>
    [Fact]
    public void WeightDecay_Sweep()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");

        var wdValues = new[] { 0f, 0.00003f, 0.0001f, 0.0003f, 0.001f, 0.003f };

        // === CEM ===
        Console.WriteLine("\n=== CEM Weight Decay (2 seeds, 45s) ===");
        Console.WriteLine("Baseline: CEM final=112/625, GA=314/625");
        float cemBestWD = 0f;
        int cemBestScore = 0;

        foreach (var wd in wdValues)
        {
            var config = MakeCEMConfig(wd);
            int avg = SweepAvgFinal(evaluator, topology, gpuCapacity, config, numSeeds: 2, budgetSeconds: 45);
            Console.WriteLine($"  CEM WD={wd:F5}: avg final={avg}/625");
            if (avg > cemBestScore) { cemBestScore = avg; cemBestWD = wd; }
        }
        Console.WriteLine($"  >> CEM best: WD={cemBestWD:F5} ({cemBestScore}/625)");

        // === ES ===
        Console.WriteLine("\n=== ES Weight Decay (2 seeds, 45s) ===");
        Console.WriteLine("Baseline: ES final=21/625, GA=314/625");
        float esBestWD = 0f;
        int esBestScore = 0;

        foreach (var wd in wdValues)
        {
            var config = MakeESConfig(wd);
            int avg = SweepAvgFinal(evaluator, topology, gpuCapacity, config, numSeeds: 2, budgetSeconds: 45);
            Console.WriteLine($"  ES  WD={wd:F5}: avg final={avg}/625");
            if (avg > esBestScore) { esBestScore = avg; esBestWD = wd; }
        }
        Console.WriteLine($"  >> ES best: WD={esBestWD:F5} ({esBestScore}/625)");
    }

    /// <summary>
    /// Phase 3: Final validation with best WD configs + early stopping.
    /// 10 seeds, 120s budget, early stop at grid >= 200.
    /// Reports: final grid (at stop), time to target, success rate.
    /// </summary>
    [Fact]
    public void FinalValidation()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"Early stopping at grid >= {GRID_TARGET}");
        Console.WriteLine("Budget: 120s per seed, 10 seeds per config\n");

        // Update these after running sweep phases:
        float cemWD = 0.0003f;  // placeholder — update after Phase 1
        float esWD = 0.0001f;   // placeholder — update after Phase 2

        var configs = new (string name, IslandConfig config)[]
        {
            ("CEM (no reg)", MakeCEMConfig(0f)),
            ($"CEM (WD={cemWD:F5})", MakeCEMConfig(cemWD)),
            ("ES (no reg)", MakeESConfig(0f)),
            ($"ES (WD={esWD:F5})", MakeESConfig(esWD)),
        };

        foreach (var (name, config) in configs)
        {
            Console.WriteLine($"=== {name} ===");
            var grids = new List<int>();
            var times = new List<double>();
            int hitCount = 0;

            for (int seed = 0; seed < 10; seed++)
            {
                var result = RunConfig(evaluator, topology, gpuCapacity, config, seed,
                    budgetSeconds: 120, earlyStop: true);
                grids.Add(result.FinalGrid);
                if (result.HitTarget)
                {
                    hitCount++;
                    times.Add(result.TimeToTarget);
                }
                Console.WriteLine($"  Seed {seed}: grid={result.FinalGrid}/625" +
                    (result.HitTarget ? $" (stopped @{result.TimeToTarget:F1}s)" : $" (timeout, {result.TotalGens} gens)"));
            }

            grids.Sort();
            int median = grids[grids.Count / 2];
            Console.WriteLine($"  Median: {median}/625, range {grids.Min()}-{grids.Max()}, " +
                $"hit>={GRID_TARGET}: {hitCount}/10");
            if (times.Count > 0)
            {
                times.Sort();
                Console.WriteLine($"  Time to target: median={times[times.Count / 2]:F1}s, " +
                    $"range {times.Min():F1}-{times.Max():F1}s");
            }
            Console.WriteLine();
        }

        Console.WriteLine("GA baseline: median=314/625 (range 195-356), final champion");
    }
}
