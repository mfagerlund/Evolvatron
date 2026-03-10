using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// SNES vs CEM comparison using GENERATION budget (not wall-clock time).
/// This makes results fair regardless of GPU contention.
/// Both strategies use identical evaluator, population size, and topology.
/// </summary>
public class SNESBenchmark
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

    private record SeedResult(int GridScore, int TotalGens);

    private static SeedResult RunStrategy(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        float[][] trainPositions,
        float[][] testGrid,
        int seed,
        int genBudget)
    {
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(seed);

        evaluator.SetStartingPositions(trainPositions);

        for (int gen = 0; gen < genBudget; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, _) = evaluator.EvaluateAllPositions(paramVectors, optimizer.TotalPopulation);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);
        }

        evaluator.SetStartingPositions(testGrid);
        var (mu, _) = optimizer.GetBestSolution();
        int gridScore = evaluator.EvaluateChampionGridScore(mu);
        return new SeedResult(gridScore, genBudget);
    }

    /// <summary>
    /// Quick smoke test: one seed, 100 generations.
    /// Verify SNES runs and produces reasonable output.
    /// </summary>
    [Fact]
    public void SNES_SmokeTest()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;
        var trainPositions = SampleTrainingPositions(fullGrid, 25, seed: 42);

        var config = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.SNES,
            InitialSigma = 0.25f,
            MinSigma = 0.08f,
            MaxSigma = 2.0f,
            SNESEtaMu = 1.0f,
            SNESEtaSigma = 0.2f,
            StagnationThreshold = 9999,
        };

        var result = RunStrategy(evaluator, topology, gpuCapacity, config,
            trainPositions, fullGrid, seed: 0, genBudget: 100);

        Console.WriteLine($"SNES smoke: grid={result.GridScore}/625 after {result.TotalGens} gens");
        Console.WriteLine($"GPU pop: {gpuCapacity}");
    }

    /// <summary>
    /// SNES eta sweep: find optimal learning rates before head-to-head.
    /// Tests eta_mu x eta_sigma combinations, 3 seeds, 500 generations.
    /// </summary>
    [Fact]
    public void SNES_EtaSweep()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;
        int genBudget = 500;
        int numSeeds = 3;

        Console.WriteLine($"=== SNES Eta Sweep ===");
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"{numSeeds} seeds, {genBudget} gen budget, multi-pos(25)\n");

        var etaMuValues = new[] { 0.5f, 1.0f, 1.5f, 2.0f };
        var etaSigmaValues = new[] { 0.1f, 0.2f, 0.3f, 0.5f };

        float bestAvg = 0;
        float bestEtaMu = 0, bestEtaSigma = 0;

        foreach (var etaMu in etaMuValues)
        foreach (var etaSigma in etaSigmaValues)
        {
            var config = new IslandConfig
            {
                IslandCount = 1,
                Strategy = UpdateStrategyType.SNES,
                InitialSigma = 0.25f,
                MinSigma = 0.08f,
                MaxSigma = 2.0f,
                SNESEtaMu = etaMu,
                SNESEtaSigma = etaSigma,
                StagnationThreshold = 9999,
            };

            int total = 0;
            for (int seed = 0; seed < numSeeds; seed++)
            {
                var trainPositions = SampleTrainingPositions(fullGrid, 25, seed * 1000);
                var result = RunStrategy(evaluator, topology, gpuCapacity, config,
                    trainPositions, fullGrid, seed, genBudget);
                total += result.GridScore;
            }

            float avg = total / (float)numSeeds;
            Console.WriteLine($"  etaMu={etaMu:F1}, etaSigma={etaSigma:F1}: avg={avg:F0}/625");

            if (avg > bestAvg)
            {
                bestAvg = avg;
                bestEtaMu = etaMu;
                bestEtaSigma = etaSigma;
            }
        }

        Console.WriteLine($"\n  >> Best: etaMu={bestEtaMu:F1}, etaSigma={bestEtaSigma:F1} ({bestAvg:F0}/625)");
    }

    /// <summary>
    /// Comprehensive head-to-head: CEM vs SNES vs SNES+mirrored.
    /// 10 seeds, 700 gen budget (matches CEM's median from 120s definitive test).
    /// Multi-pos(25), 5->4->4->3 (59 params), final champion on 625 grid.
    ///
    /// IMPORTANT: Uses generation budget, not wall-clock time.
    /// Per-generation compute is identical for all strategies (same evaluator,
    /// same population, same topology), so this is a fair comparison
    /// regardless of GPU contention from other processes.
    /// </summary>
    [Fact]
    public void SNES_vs_CEM_GenerationMatched()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;
        int genBudget = 700;
        int numSeeds = 10;

        Console.WriteLine($"=== SNES vs CEM (Generation-Matched) ===");
        Console.WriteLine($"Topology: {topology}, GPU pop: {gpuCapacity}");
        Console.WriteLine($"{numSeeds} seeds, {genBudget} gen budget, multi-pos(25)");
        Console.WriteLine($"Comparison: generation-matched (immune to GPU contention)\n");

        // --- CEM (current champion) ---
        {
            Console.WriteLine("--- CEM multi-pos(25) [current champion] ---");
            var config = new IslandConfig
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
            RunAndReport(evaluator, topology, gpuCapacity, config, fullGrid, numSeeds, genBudget);
        }

        // --- SNES ---
        {
            Console.WriteLine("--- SNES multi-pos(25) ---");
            var config = new IslandConfig
            {
                IslandCount = 1,
                Strategy = UpdateStrategyType.SNES,
                InitialSigma = 0.25f,
                MinSigma = 0.08f,
                MaxSigma = 2.0f,
                SNESEtaMu = 2.0f,
                SNESEtaSigma = 0.1f,
                SNESMirrored = false,
                StagnationThreshold = 9999,
            };
            RunAndReport(evaluator, topology, gpuCapacity, config, fullGrid, numSeeds, genBudget);
        }

        // --- SNES + mirrored sampling ---
        {
            Console.WriteLine("--- SNES+mirrored multi-pos(25) ---");
            var config = new IslandConfig
            {
                IslandCount = 1,
                Strategy = UpdateStrategyType.SNES,
                InitialSigma = 0.25f,
                MinSigma = 0.08f,
                MaxSigma = 2.0f,
                SNESEtaMu = 2.0f,
                SNESEtaSigma = 0.1f,
                SNESMirrored = true,
                StagnationThreshold = 9999,
            };
            RunAndReport(evaluator, topology, gpuCapacity, config, fullGrid, numSeeds, genBudget);
        }

        Console.WriteLine("=== REFERENCE ===");
        Console.WriteLine("Previous definitive results (120s wall-clock, 10 seeds):");
        Console.WriteLine("  CEM multi-pos(25):  median 338/625, range 293-353, 10/10 pass>=200, median 667 gens");
        Console.WriteLine("  ES multi-pos(25):   median 316/625, range 216-395, 10/10 pass>=200, median 565 gens");
        Console.WriteLine("  CEM multi-pos(10):  median 307/625, range 175-362, 9/10 pass>=200, median 1089 gens");
    }

    private void RunAndReport(
        GPUDenseDoublePoleEvaluator evaluator,
        DenseTopology topology,
        int gpuCapacity,
        IslandConfig config,
        float[][] fullGrid,
        int numSeeds,
        int genBudget)
    {
        var grids = new List<int>();

        for (int seed = 0; seed < numSeeds; seed++)
        {
            var trainPositions = SampleTrainingPositions(fullGrid, 25, seed * 1000);
            var result = RunStrategy(evaluator, topology, gpuCapacity, config,
                trainPositions, fullGrid, seed, genBudget);
            grids.Add(result.GridScore);
            Console.WriteLine($"  Seed {seed}: grid={result.GridScore}/625");
        }

        grids.Sort();
        int median = grids[grids.Count / 2];
        float mean = (float)grids.Average();
        Console.WriteLine($"  >> Median: {median}/625, mean: {mean:F0}/625, range {grids.Min()}-{grids.Max()}, " +
            $"pass>=200: {grids.Count(g => g >= 200)}/{numSeeds}\n");
    }
}
