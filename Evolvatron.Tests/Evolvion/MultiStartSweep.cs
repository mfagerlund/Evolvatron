using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Multi-position training for CEM and ES on DPNV 625-grid.
/// Instead of training on a single starting position (where sigma collapse → specialist),
/// train on N positions simultaneously so fitness directly selects for generalization.
///
/// With 16K GPU pop, each generation evaluates pop × N_positions episodes.
/// More positions = slower generations but MUCH richer fitness signal.
///
/// Topology: 5→4→4→3 only (Elman ctx=2, 59 params).
/// Baseline: GA final=314/625, CEM single-pos final=112/625, ES single-pos final=21/625.
/// </summary>
public class MultiStartSweep
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

    /// <summary>
    /// Sample N positions from the full 625 grid using Fisher-Yates shuffle.
    /// </summary>
    private static float[][] SampleTrainingPositions(float[][] fullGrid, int count, int seed)
    {
        if (count >= fullGrid.Length) return fullGrid;

        var rng = new Random(seed);
        var indices = new int[fullGrid.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;

        for (int i = 0; i < count; i++)
        {
            int j = rng.Next(i, indices.Length);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var result = new float[count][];
        for (int i = 0; i < count; i++)
            result[i] = fullGrid[indices[i]];
        return result;
    }

    private record RunResult(int FinalGrid, int SolveGen, double SolveTime, int TotalGens);

    /// <summary>
    /// Train with multi-position, test final champion on full 625 grid.
    /// </summary>
    private static RunResult RunMultiPositionConfig(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        float[][] trainPositions,
        float[][] testGrid,
        int seed,
        double budgetSeconds)
    {
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(seed);
        var sw = Stopwatch.StartNew();

        int solveGen = -1;
        double solveTime = -1;
        int totalGens = 0;

        evaluator.SetStartingPositions(trainPositions);

        for (int gen = 0; gen < 100_000; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, solvedCount) = evaluator.EvaluateAllPositions(
                paramVectors, optimizer.TotalPopulation);
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

        // Always test final champion on full 625 grid
        evaluator.SetStartingPositions(testGrid);
        var (mu, _) = optimizer.GetBestSolution();
        int finalGrid = evaluator.EvaluateChampionGridScore(mu);

        return new RunResult(finalGrid, solveGen, solveTime, totalGens);
    }

    private static IslandConfig MakeCEMConfig()
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

    private static IslandConfig MakeESConfig()
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
    /// Combined CEM + ES multi-position sweep.
    /// Sweeps training positions: 1, 5, 10, 25, 50.
    /// 2 seeds per config, 120s budget.
    /// Reports FINAL champion grid score (what you'd deploy).
    /// </summary>
    [Fact]
    public void MultiPosition_Sweep()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;

        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"Budget: 60s per seed, 2 seeds per config");
        Console.WriteLine($"Fitness: EvaluateAllPositions (solvedCount*MaxSteps + meanSteps)");
        Console.WriteLine($"Test: final champion on 625 grid\n");

        var trainCounts = new[] { 1, 5, 10, 25 };
        int numSeeds = 2;
        double budget = 60;

        // === CEM ===
        Console.WriteLine("=== CEM Multi-Position Training ===");
        Console.WriteLine("Baseline: CEM single-pos final=112/625, GA=314/625\n");

        int cemBestCount = 0;
        int cemBestScore = 0;

        foreach (var nTrain in trainCounts)
        {
            int totalGrid = 0;
            int totalGens = 0;

            for (int seed = 0; seed < numSeeds; seed++)
            {
                var trainPositions = nTrain == 1
                    ? new[] { new float[] { 0, 0, MathF.PI / 45f, 0, 0, 0 } }
                    : SampleTrainingPositions(fullGrid, nTrain, seed * 1000);

                var config = MakeCEMConfig();
                var result = RunMultiPositionConfig(evaluator, topology, gpuCapacity,
                    config, trainPositions, fullGrid, seed, budget);
                totalGrid += result.FinalGrid;
                totalGens += result.TotalGens;
                Console.WriteLine($"  CEM train={nTrain}: seed={seed}, final={result.FinalGrid}/625, " +
                    $"gens={result.TotalGens}, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
            }

            int avgGrid = totalGrid / numSeeds;
            int avgGens = totalGens / numSeeds;
            Console.WriteLine($"  >> CEM train={nTrain}: avg final={avgGrid}/625, avg gens={avgGens}\n");

            if (avgGrid > cemBestScore) { cemBestScore = avgGrid; cemBestCount = nTrain; }
        }
        Console.WriteLine($"  >> CEM BEST: train={cemBestCount} ({cemBestScore}/625)\n");

        // === ES ===
        Console.WriteLine("=== ES Multi-Position Training ===");
        Console.WriteLine("Baseline: ES single-pos final=21/625, GA=314/625\n");

        int esBestCount = 0;
        int esBestScore = 0;

        foreach (var nTrain in trainCounts)
        {
            int totalGrid = 0;
            int totalGens = 0;

            for (int seed = 0; seed < numSeeds; seed++)
            {
                var trainPositions = nTrain == 1
                    ? new[] { new float[] { 0, 0, MathF.PI / 45f, 0, 0, 0 } }
                    : SampleTrainingPositions(fullGrid, nTrain, seed * 1000);

                var config = MakeESConfig();
                var result = RunMultiPositionConfig(evaluator, topology, gpuCapacity,
                    config, trainPositions, fullGrid, seed, budget);
                totalGrid += result.FinalGrid;
                totalGens += result.TotalGens;
                Console.WriteLine($"  ES  train={nTrain}: seed={seed}, final={result.FinalGrid}/625, " +
                    $"gens={result.TotalGens}, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
            }

            int avgGrid = totalGrid / numSeeds;
            int avgGens = totalGens / numSeeds;
            Console.WriteLine($"  >> ES  train={nTrain}: avg final={avgGrid}/625, avg gens={avgGens}\n");

            if (avgGrid > esBestScore) { esBestScore = avgGrid; esBestCount = nTrain; }
        }
        Console.WriteLine($"  >> ES BEST: train={esBestCount} ({esBestScore}/625)\n");
    }

    /// <summary>
    /// Final validation with best multi-position config.
    /// 10 seeds, 120s budget, reports final champion grid score.
    /// UPDATE cemTrainCount and esTrainCount after running sweep.
    /// </summary>
    [Fact]
    public void MultiPosition_Validation()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;

        // Results from sweep: CEM best at train=10 (318/625), ES best at train=25 (287/625)
        int cemTrainCount = 10;
        int esTrainCount = 25;

        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"10 seeds, 120s budget, final champion on 625 grid\n");

        var configs = new (string name, IslandConfig config, int trainCount)[]
        {
            ("CEM single-pos", MakeCEMConfig(), 1),
            ($"CEM multi-pos({cemTrainCount})", MakeCEMConfig(), cemTrainCount),
            ("ES single-pos", MakeESConfig(), 1),
            ($"ES multi-pos({esTrainCount})", MakeESConfig(), esTrainCount),
        };

        foreach (var (name, config, trainCount) in configs)
        {
            Console.WriteLine($"=== {name} ===");
            var grids = new List<int>();

            for (int seed = 0; seed < 10; seed++)
            {
                var trainPositions = trainCount == 1
                    ? new[] { new float[] { 0, 0, MathF.PI / 45f, 0, 0, 0 } }
                    : SampleTrainingPositions(fullGrid, trainCount, seed * 1000);

                var result = RunMultiPositionConfig(evaluator, topology, gpuCapacity,
                    config, trainPositions, fullGrid, seed, 120);
                grids.Add(result.FinalGrid);
                Console.WriteLine($"  Seed {seed}: grid={result.FinalGrid}/625, " +
                    $"gens={result.TotalGens}, solve=gen{result.SolveGen}");
            }

            grids.Sort();
            int median = grids[grids.Count / 2];
            Console.WriteLine($"  Median: {median}/625, range {grids.Min()}-{grids.Max()}, " +
                $"pass>=200: {grids.Count(g => g >= 200)}/10\n");
        }

        Console.WriteLine("GA baseline: median=314/625 (range 195-356), final champion");
    }
}
