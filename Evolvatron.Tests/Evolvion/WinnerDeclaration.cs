using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Definitive comparison: CEM vs ES vs GA on DPNV 625-grid.
/// All multi-pos training, 10 seeds, 120s budget, 5→4→4→3 (Elman ctx=2, 59 params).
/// Test final champion on full 625 grid (what you'd actually deploy).
/// Goal: declare a clear winner for the Evolvatron game.
/// </summary>
public class WinnerDeclaration
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

    private record SeedResult(int GridScore, int TotalGens, int SolveGen, double SolveTime);

    private static SeedResult RunCEMOrES(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        float[][] trainPositions,
        float[][] testGrid,
        int seed,
        double budget)
    {
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(seed);
        var sw = Stopwatch.StartNew();
        int solveGen = -1;
        double solveTime = -1;

        evaluator.SetStartingPositions(trainPositions);

        int totalGens = 0;
        for (int gen = 0; gen < 100_000; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, solvedCount) = evaluator.EvaluateAllPositions(paramVectors, optimizer.TotalPopulation);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);
            totalGens = gen + 1;

            if (solveGen < 0 && solvedCount > 0)
            {
                solveGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
            }
            if (sw.Elapsed.TotalSeconds > budget) break;
        }

        evaluator.SetStartingPositions(testGrid);
        var (mu, _) = optimizer.GetBestSolution();
        int gridScore = evaluator.EvaluateChampionGridScore(mu);
        return new SeedResult(gridScore, totalGens, solveGen, solveTime);
    }

    private static SeedResult RunGA(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        int numSpecies,
        float[][] trainPositions,
        float[][] testGrid,
        int seed,
        double budget)
    {
        var ga = new DenseGAOptimizer(topology, gpuCapacity, numSpecies, seed)
        {
            JitterStdDev = 0.15f,
            EliteCount = Math.Max(2, 10 / numSpecies),
            TournamentSize = Math.Max(3, 5),
            ParentPoolFraction = 0.5f,
            StagnationThreshold = 50,
        };
        var rng = new Random(seed);
        var sw = Stopwatch.StartNew();
        int solveGen = -1;
        double solveTime = -1;

        evaluator.SetStartingPositions(trainPositions);

        int totalGens = 0;
        for (int gen = 0; gen < 50_000; gen++)
        {
            var paramVectors = ga.GetParamVectors();
            var (fitness, solvedCount) = evaluator.EvaluateAllPositions(paramVectors, ga.TotalPopulation);
            ga.Update(fitness);
            ga.StepGeneration(rng);
            ga.ManageSpecies(rng);
            totalGens = gen + 1;

            if (solveGen < 0 && solvedCount > 0)
            {
                solveGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
            }
            if (sw.Elapsed.TotalSeconds > budget) break;
        }

        evaluator.SetStartingPositions(testGrid);
        var (bestParams, _) = ga.GetBest();
        int gridScore = evaluator.EvaluateChampionGridScore(bestParams);
        return new SeedResult(gridScore, totalGens, solveGen, solveTime);
    }

    /// <summary>
    /// THE definitive test. 10 seeds, 120s, all algorithms multi-pos.
    /// Reports final champion grid score (what you'd deploy).
    /// </summary>
    [Fact]
    public void Definitive_Comparison()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;
        int numSeeds = 10;
        double budget = 120;

        Console.WriteLine($"=== DEFINITIVE ALGORITHM COMPARISON ===");
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"{numSeeds} seeds, {budget}s budget, final champion on 625 grid\n");

        // --- CEM multi-pos(10) ---
        {
            Console.WriteLine("=== CEM multi-pos(10) ===");
            var config = new IslandConfig
            {
                IslandCount = 1, Strategy = UpdateStrategyType.CEM,
                InitialSigma = 0.25f, MinSigma = 0.08f, MaxSigma = 2.0f,
                CEMEliteFraction = 0.01f, CEMSigmaSmoothing = 0.3f, CEMMuSmoothing = 0.2f,
                StagnationThreshold = 9999,
            };
            RunAndReport(evaluator, topology, gpuCapacity, config, null, 10,
                fullGrid, numSeeds, budget);
        }

        // --- CEM multi-pos(25) ---
        {
            Console.WriteLine("=== CEM multi-pos(25) ===");
            var config = new IslandConfig
            {
                IslandCount = 1, Strategy = UpdateStrategyType.CEM,
                InitialSigma = 0.25f, MinSigma = 0.08f, MaxSigma = 2.0f,
                CEMEliteFraction = 0.01f, CEMSigmaSmoothing = 0.3f, CEMMuSmoothing = 0.2f,
                StagnationThreshold = 9999,
            };
            RunAndReport(evaluator, topology, gpuCapacity, config, null, 25,
                fullGrid, numSeeds, budget);
        }

        // --- ES multi-pos(25) ---
        {
            Console.WriteLine("=== ES multi-pos(25) ===");
            var config = new IslandConfig
            {
                IslandCount = 1, Strategy = UpdateStrategyType.ES,
                ESSigma = 0.05f, ESLearningRate = 0.05f,
                ESAdamBeta1 = 0.80f, ESAdamBeta2 = 0.999f,
                InitialSigma = 0.25f, StagnationThreshold = 9999,
            };
            RunAndReport(evaluator, topology, gpuCapacity, config, null, 25,
                fullGrid, numSeeds, budget);
        }

        // --- GA 3 species multi-pos(10) ---
        {
            Console.WriteLine("=== GA 3sp multi-pos(10) ===");
            RunAndReport(evaluator, topology, gpuCapacity, null, 3, 10,
                fullGrid, numSeeds, budget);
        }

        Console.WriteLine("\n=== SUMMARY ===");
        Console.WriteLine("Previous baselines: sparse GA single-pos=314, CEM tracked=223, ES tracked=164");
        Console.WriteLine("Multi-pos baselines: CEM(10)=303, ES(25)=315, GA 3sp(10)=271");
    }

    private void RunAndReport(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig? cemEsConfig,
        int? gaSpecies,
        int trainCount,
        float[][] fullGrid,
        int numSeeds,
        double budget)
    {
        var grids = new List<int>();
        var gens = new List<int>();

        for (int seed = 0; seed < numSeeds; seed++)
        {
            var trainPositions = SampleTrainingPositions(fullGrid, trainCount, seed * 1000);
            SeedResult result;

            if (cemEsConfig != null)
                result = RunCEMOrES(evaluator, topology, gpuCapacity, cemEsConfig,
                    trainPositions, fullGrid, seed, budget);
            else
                result = RunGA(evaluator, topology, gpuCapacity, gaSpecies!.Value,
                    trainPositions, fullGrid, seed, budget);

            grids.Add(result.GridScore);
            gens.Add(result.TotalGens);
            Console.WriteLine($"  Seed {seed}: grid={result.GridScore}/625, " +
                $"gens={result.TotalGens}, solve=gen{result.SolveGen}/{result.SolveTime:F1}s");
        }

        grids.Sort();
        gens.Sort();
        int median = grids[grids.Count / 2];
        int medianGens = gens[gens.Count / 2];
        Console.WriteLine($"  >> Median: {median}/625, range {grids.Min()}-{grids.Max()}, " +
            $"pass>=200: {grids.Count(g => g >= 200)}/{numSeeds}, " +
            $"median gens: {medianGens}\n");
    }
}
