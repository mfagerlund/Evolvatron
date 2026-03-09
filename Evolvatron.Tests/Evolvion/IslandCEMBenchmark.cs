using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Benchmark: Island-model CEM on DPNV (double pole no velocity).
/// Validates that CEM with dense NN solves DPNV and measures 625-grid generalization.
/// GA baseline: 5→4→4→3 median 314/625, 5→8→3 median 287/625.
/// </summary>
public class IslandCEMBenchmark
{
    /// <summary>
    /// Standard 625-grid: 5^4 grid over (cartPos, cartVel, pole1Angle, pole1AngVel).
    /// Pole2 always at (0, 0). Matches Gruau 1996 benchmark.
    /// </summary>
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
    /// CEM single-position solve test. Validates the full loop works.
    /// Expected: solves in <30s (GA solves in ~2s, CEM should be comparable or faster).
    /// </summary>
    [Fact]
    public void CEM_SolvesDoublePole_SinglePosition()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        Console.WriteLine($"Topology: {topology}");

        var config = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.3f,
            CEMEliteFraction = 0.1f,
            CEMSigmaSmoothing = 0.4f,
            MinSigma = 0.02f,
        };

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 10_000;
        evaluator.ContextSize = 2;

        int gpuCapacity = evaluator.OptimalPopulationSize;
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(42);

        Console.WriteLine($"Population: {optimizer.TotalPopulation}, Islands: {optimizer.Islands.Count}, " +
                          $"Params: {topology.TotalParams}");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool solved = false;

        for (int gen = 0; gen < 500; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, solvedCount) = evaluator.EvaluatePopulation(paramVectors, optimizer.TotalPopulation);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            if (gen % 10 == 0 || solvedCount > 0)
            {
                var (_, bestFit) = optimizer.GetBestSolution();
                Console.WriteLine($"Gen {gen}: best={bestFit:F1}, solved={solvedCount}, " +
                                  $"elapsed={sw.Elapsed.TotalSeconds:F1}s");
            }

            if (solvedCount > 0)
            {
                Console.WriteLine($"SOLVED at gen {gen} in {sw.Elapsed.TotalSeconds:F1}s");
                solved = true;
                break;
            }

            if (sw.Elapsed.TotalSeconds > 60)
            {
                Console.WriteLine($"TIMEOUT at gen {gen}");
                break;
            }
        }

        Assert.True(solved, "CEM should solve single-position DPNV within 60s");
    }

    /// <summary>
    /// CEM 625-grid generalization benchmark.
    /// Train on single position, test champion on 625-grid periodically.
    /// GA baseline: 5→4→4→3 median 314/625.
    /// </summary>
    [Theory]
    [InlineData(new[] { 4, 4 }, 0, "5>4>4>3")]
    [InlineData(new[] { 8 }, 1, "5>8>3")]
    public void CEM_625Grid_Generalization(int[] hiddenSizes, int seed, string label)
    {
        var topology = DenseTopology.ForDPNV(hiddenSizes, contextSize: 2);
        Console.WriteLine($"[{label}] Topology: {topology}, seed={seed}");

        var config = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.3f,
            CEMEliteFraction = 0.1f,
            CEMSigmaSmoothing = 0.4f,
            MinSigma = 0.02f,
        };

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000; // Standard Gruau: 1000 steps
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(seed);

        Console.WriteLine($"Population: {optimizer.TotalPopulation}, Params: {topology.TotalParams}");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool hasSolved = false;
        int bestGrid = 0;
        int bestGridGen = -1;
        double lastGridCheck = 0;

        for (int gen = 0; gen < 2000; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, solvedCount) = evaluator.EvaluatePopulation(paramVectors, optimizer.TotalPopulation);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            double elapsed = sw.Elapsed.TotalSeconds;

            if (!hasSolved && solvedCount > 0)
            {
                hasSolved = true;
                Console.WriteLine($"[{label}] First solve: gen {gen}, {elapsed:F1}s");
            }

            // Check 625-grid every 5s after first solve
            if (hasSolved && elapsed - lastGridCheck >= 5.0)
            {
                lastGridCheck = elapsed;
                var (mu, _) = optimizer.GetBestSolution();
                int gridScore = evaluator.EvaluateChampionGridScore(mu);

                if (gridScore > bestGrid)
                {
                    bestGrid = gridScore;
                    bestGridGen = gen;
                }

                Console.WriteLine($"[{label}] Gen {gen} ({elapsed:F1}s): grid={gridScore}/625, " +
                                  $"best={bestGrid}/625 @ gen {bestGridGen}");
            }

            if (elapsed > 100) break;
        }

        Console.WriteLine($"[{label}] Final: best grid={bestGrid}/625 @ gen {bestGridGen}");
        // GA baseline: 5→4→4→3 median 314/625, 5→8→3 median 287/625
        // CEM from single-position training may not match — this validates the pipeline works
        Assert.True(bestGrid >= 50, $"Expected at least 50/625 grid score, got {bestGrid}");
    }

    /// <summary>
    /// Compare 1 island vs 5 islands on 625-grid.
    /// Tests whether island overhead hurts generalization.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public void CEM_IslandCount_Comparison(int islandCount)
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);

        var config = new IslandConfig
        {
            IslandCount = islandCount,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.3f,
            CEMEliteFraction = 0.1f,
            CEMSigmaSmoothing = 0.4f,
            MinSigma = 0.02f,
            StagnationThreshold = 30,
        };

        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;
        evaluator.SetStartingPositions(Build625Grid());

        int gpuCapacity = evaluator.OptimalPopulationSize;
        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(42);

        Console.WriteLine($"Islands={islandCount}, PopPerIsland={optimizer.IndividualsPerIsland}, " +
                          $"TotalPop={optimizer.TotalPopulation}, Params={topology.TotalParams}");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool hasSolved = false;
        int bestGrid = 0;
        double lastGridCheck = 0;

        for (int gen = 0; gen < 2000; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, solvedCount) = evaluator.EvaluatePopulation(paramVectors, optimizer.TotalPopulation);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            double elapsed = sw.Elapsed.TotalSeconds;

            if (!hasSolved && solvedCount > 0)
            {
                hasSolved = true;
                Console.WriteLine($"[{islandCount}isl] First solve: gen {gen}, {elapsed:F1}s");
            }

            if (hasSolved && elapsed - lastGridCheck >= 5.0)
            {
                lastGridCheck = elapsed;
                var (mu, _) = optimizer.GetBestSolution();
                int gridScore = evaluator.EvaluateChampionGridScore(mu);
                bestGrid = Math.Max(bestGrid, gridScore);
                Console.WriteLine($"[{islandCount}isl] Gen {gen} ({elapsed:F1}s): grid={gridScore}/625, best={bestGrid}");
            }

            if (elapsed > 60) break;
        }

        Console.WriteLine($"[{islandCount}isl] Final best grid: {bestGrid}/625");
    }

    /// <summary>
    /// CPU-only test: validates CEM converges on simple sphere function.
    /// No GPU required. Verifies strategy logic independently.
    /// </summary>
    [Fact]
    public void CEM_ConvergesOnSphere_CPUOnly()
    {
        int dims = 50;
        var config = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 1.0f,
            CEMEliteFraction = 0.1f,
            CEMSigmaSmoothing = 0.2f,
            MinSigma = 1e-6f,
        };

        var topology = new DenseTopology(dims, 1); // dummy topology for param count
        var island = new Island(dims, config.InitialSigma);
        var strategy = new CEMStrategy(config);
        var rng = new Random(42);

        int popSize = 200;
        var paramVectors = new float[popSize * dims];
        var fitnesses = new float[popSize];

        for (int gen = 0; gen < 100; gen++)
        {
            strategy.GenerateSamples(island, paramVectors, popSize, rng);

            // Sphere function: f(x) = -||x||^2 (maximize toward 0)
            for (int i = 0; i < popSize; i++)
            {
                float sumSq = 0;
                for (int d = 0; d < dims; d++)
                {
                    float v = paramVectors[i * dims + d];
                    sumSq += v * v;
                }
                fitnesses[i] = -sumSq;
            }

            strategy.Update(island, fitnesses, paramVectors, popSize);
        }

        // mu should be close to zero
        float muNorm = 0;
        float sigmaMax = 0;
        for (int d = 0; d < dims; d++)
        {
            muNorm += island.Mu[d] * island.Mu[d];
            sigmaMax = MathF.Max(sigmaMax, island.Sigma[d]);
        }
        muNorm = MathF.Sqrt(muNorm);

        Console.WriteLine($"After 100 gens: |mu|={muNorm:E3}, max(sigma)={sigmaMax:E3}");
        Assert.True(muNorm < 0.1f, $"CEM should converge to origin on sphere, got |mu|={muNorm}");
        Assert.True(sigmaMax < 0.01f, $"Sigma should collapse on sphere, got max(sigma)={sigmaMax}");
    }

    /// <summary>
    /// CPU-only test: validates ES converges on sphere function.
    /// </summary>
    [Fact]
    public void ES_ConvergesOnSphere_CPUOnly()
    {
        int dims = 50;
        var config = new IslandConfig
        {
            Strategy = UpdateStrategyType.ES,
            ESSigma = 0.1f,
            ESLearningRate = 0.05f,
        };

        var island = new Island(dims, 0.1f);
        // Initialize mu away from origin
        var rng = new Random(42);
        for (int d = 0; d < dims; d++)
            island.Mu[d] = Island.SampleGaussian(rng) * 2f;

        var strategy = new ESStrategy(config);
        int popSize = 200; // must be even for antithetic sampling
        var paramVectors = new float[popSize * dims];
        var fitnesses = new float[popSize];

        for (int gen = 0; gen < 200; gen++)
        {
            strategy.GenerateSamples(island, paramVectors, popSize, rng);

            for (int i = 0; i < popSize; i++)
            {
                float sumSq = 0;
                for (int d = 0; d < dims; d++)
                {
                    float v = paramVectors[i * dims + d];
                    sumSq += v * v;
                }
                fitnesses[i] = -sumSq;
            }

            strategy.Update(island, fitnesses, paramVectors, popSize);
        }

        float muNorm = 0;
        for (int d = 0; d < dims; d++)
            muNorm += island.Mu[d] * island.Mu[d];
        muNorm = MathF.Sqrt(muNorm);

        Console.WriteLine($"ES after 200 gens: |mu|={muNorm:E3}");
        Assert.True(muNorm < 0.5f, $"ES should converge toward origin on sphere, got |mu|={muNorm}");
    }
}
