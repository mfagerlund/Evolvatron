using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Standard DPNV 625-grid benchmark (Gruau 1996).
/// Train on single starting position (pole1=4°), test generalization on 5^4 grid.
/// Directly comparable with published CMA-ES/NEAT/CoSyNE/ESP results.
///
/// Literature (Gruau fitness, generalization /625):
///   CoSyNE:  3,416 evals,  score N/A
///   CMA-ES:  6,061 evals,  250/625 (3 hidden, 28 weights)
///   NEAT:    33,184 evals, 286/625
///   ESP:     169,466 evals, 289/625
/// </summary>
public class StandardDPNVBenchmark
{
    private readonly ITestOutputHelper _output;
    private const string ResultsDir = @"C:\Dev\Evolvatron\scratch\standard_dpnv";

    public StandardDPNVBenchmark(ITestOutputHelper output) => _output = output;

    #region 625-Grid Generation

    /// <summary>
    /// Standard 625-position generalization grid (Gruau 1996, Igel 2003, Gomez 2008).
    /// 5^4 grid over (cartPos, cartVel, pole1Angle, pole1AngVel), pole2 always (0,0).
    /// k in {0.05, 0.25, 0.5, 0.75, 0.95}.
    /// </summary>
    private static float[][] GenerateStandard625Grid()
    {
        float[] k = { 0.05f, 0.25f, 0.5f, 0.75f, 0.95f };
        var positions = new float[625][];
        int idx = 0;
        foreach (float kx in k)
        foreach (float kv in k)
        foreach (float ka in k)
        foreach (float kw in k)
        {
            positions[idx++] = new[]
            {
                kx * 4.32f - 2.16f,              // cartPos: [-2.16, +2.16] m
                kv * 2.70f - 1.35f,               // cartVel: [-1.35, +1.35] m/s
                ka * 0.12566371f - 0.06283185f,   // pole1Angle: [-3.6°, +3.6°] rad
                kw * 0.30019663f - 0.15009832f,   // pole1AngVel: [-8.6°/s, +8.6°/s] rad/s
                0f,                                // pole2Angle: always 0
                0f                                 // pole2AngVel: always 0
            };
        }
        return positions;
    }

    #endregion

    #region Topology Helpers

    private static SpeciesSpec BuildDenseTopology(int ctx, int[] hidden)
    {
        int inputSize = 3 + ctx;
        int outputSize = 1 + ctx;
        var builder = new SpeciesBuilder().AddInputRow(inputSize);
        foreach (int h in hidden)
            builder = builder.AddHiddenRow(h);
        return builder.AddOutputRow(outputSize).InitializeDense(new Random(42)).Build();
    }

    #endregion

    /// <summary>
    /// Main sweep: 6 topologies × 5 seeds × 100s budget.
    /// Train single-position (4° pole1), MaxSteps=1000, Gruau fitness, 1 species, 16K pop.
    /// Test champion on 625-grid every 5s after first solve.
    /// Total budget: ~55 minutes.
    /// </summary>
    [Fact]
    public void Standard625_Sweep()
    {
        var grid625 = GenerateStandard625Grid();
        Assert.Equal(625, grid625.Length);

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine("=== Standard DPNV 625-Grid Benchmark ===");
        _output.WriteLine($"Device: {gpu.DeviceName}");
        _output.WriteLine($"Population: {optPop}");
        _output.WriteLine($"Training: single pos (4deg pole1), MaxSteps=1000, Gruau fitness");
        _output.WriteLine($"Grid test: 625 positions, 1000 steps each, pass=survived");
        _output.WriteLine($"Literature: CMA-ES=250/625, NEAT=286/625, ESP=289/625");
        _output.WriteLine("");

        WarmupGpu(gpu);

        var configs = new (string name, int[] hidden)[]
        {
            ("5>3>3",   new[] { 3 }),      // 24 edges — CMA-ES equivalent
            ("5>6>3",   new[] { 6 }),      // 48 edges — current DPNV optimal
            ("5>8>3",   new[] { 8 }),      // 64 edges
            ("5>10>3",  new[] { 10 }),     // 80 edges
            ("5>4>4>3", new[] { 4, 4 }),   // 48 edges — deeper
            ("5>6>6>3", new[] { 6, 6 }),   // 66 edges — deeper+wider
        };

        int numSeeds = 5;
        int budgetSec = 100;

        var report = new StringBuilder();
        report.AppendLine("# Standard DPNV 625-Grid Benchmark");
        report.AppendLine($"Device: {gpu.DeviceName}, Pop: {optPop}");
        report.AppendLine($"Training: single pos, MaxSteps=1000, Gruau, Elman ctx=2, no edge mutations");
        report.AppendLine($"Budget: {budgetSec}s per run, {numSeeds} seeds per config");
        report.AppendLine($"Literature: CMA-ES=250/625, NEAT=286/625, ESP=289/625, CE=300/625");
        report.AppendLine();

        foreach (var (name, hidden) in configs)
        {
            var topology = BuildDenseTopology(2, hidden);
            int edgeCount = topology.Edges.Count;

            _output.WriteLine($"--- {name} ({edgeCount} edges) ---");
            report.AppendLine($"## {name} ({edgeCount} edges)");

            var runResults = new List<GridRunResult>();

            for (int seed = 0; seed < numSeeds; seed++)
            {
                var r = RunStandard625Experiment(gpu, hidden, optPop, seed, grid625, budgetSec);
                runResults.Add(r);

                string solveStr = r.FirstSolveGen >= 0
                    ? $"solve=gen{r.FirstSolveGen}/{r.FirstSolveTime:F1}s"
                    : "NO SOLVE";
                string line = $"  Seed {seed}: {solveStr}, " +
                    $"grid={r.BestGridScore}/625 @ gen{r.BestGridGen} ({r.BestGridTime:F1}s), " +
                    $"{r.TotalGens} gens ({r.GenPerSec:F0} gen/s), " +
                    $"evals={r.TotalGens * (long)optPop:N0}";
                _output.WriteLine(line);
                report.AppendLine(line);
            }

            // Summary stats
            int solved = runResults.Count(r => r.FirstSolveGen >= 0);
            var gridScores = runResults.Select(r => r.BestGridScore).ToList();
            double medianGrid = Median(gridScores.Select(s => (double)s).ToList());
            int maxGrid = gridScores.Max();
            int minGrid = gridScores.Min();
            int pass200 = gridScores.Count(s => s >= 200);

            var solveGens = runResults.Where(r => r.FirstSolveGen >= 0)
                .Select(r => (double)r.FirstSolveGen).ToList();
            string solveGenStr = solveGens.Count > 0
                ? $"median gen {Median(solveGens):F0}"
                : "N/A";

            string summary = $"  >> {solved}/{numSeeds} solved ({solveGenStr}), " +
                $"grid: median={medianGrid:F0} [{minGrid}-{maxGrid}], " +
                $"pass(>=200): {pass200}/{numSeeds}";
            _output.WriteLine(summary);
            _output.WriteLine("");
            report.AppendLine(summary);
            report.AppendLine();
        }

        Directory.CreateDirectory(ResultsDir);
        File.WriteAllText(Path.Combine(ResultsDir, "standard_625_sweep.md"), report.ToString());
        _output.WriteLine($"Results saved to {ResultsDir}");
    }

    private GridRunResult RunStandard625Experiment(
        GPUDoublePoleEvaluator gpu, int[] hidden, int optPop, int seed,
        float[][] grid625, int budgetSec)
    {
        int ctx = 2;

        // Configure for training: single position, Gruau fitness
        gpu.IncludeVelocity = false;
        gpu.ContextSize = ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = 1000;
        gpu.UseGruauFitness = true;

        var topology = BuildDenseTopology(ctx, hidden);

        var config = new EvolutionConfig
        {
            SpeciesCount = 1,
            IndividualsPerSpecies = optPop,
            MinSpeciesCount = 1,
            Elites = 10,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
            EdgeMutations = new EdgeMutationConfig
            {
                EdgeAdd = 0, EdgeDeleteRandom = 0, EdgeSplit = 0,
                EdgeRedirect = 0, EdgeSwap = 0
            }
        };
        config.MutationRates.WeightJitterStdDev = 0.15f;

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        int gen = 0;
        int firstSolveGen = -1;
        double firstSolveTime = 0;
        int bestGridScore = 0;
        int bestGridGen = 0;
        double bestGridTime = 0;
        double lastGridTestTime = -999;

        while (sw.Elapsed.TotalSeconds < budgetSec)
        {
            var sp = population.AllSpecies[0];
            var (fitness, solved) = gpu.EvaluatePopulation(sp.Topology, sp.Individuals, seed: gen);

            float bestFitness = float.MinValue;
            int bestIdx = 0;
            for (int i = 0; i < sp.Individuals.Count; i++)
            {
                var ind = sp.Individuals[i];
                ind.Fitness = fitness[i];
                sp.Individuals[i] = ind;
                if (fitness[i] > bestFitness) { bestFitness = fitness[i]; bestIdx = i; }
            }

            if (solved > 0 && firstSolveGen < 0)
            {
                firstSolveGen = gen;
                firstSolveTime = sw.Elapsed.TotalSeconds;
            }

            // Test 625-grid every 5s after first solve
            bool shouldTestGrid = firstSolveGen >= 0 &&
                (sw.Elapsed.TotalSeconds - lastGridTestTime >= 5.0 || firstSolveGen == gen);

            if (shouldTestGrid)
            {
                var champion = sp.Individuals[bestIdx];
                int gridScore = TestGrid625(gpu, sp.Topology, champion, grid625);
                lastGridTestTime = sw.Elapsed.TotalSeconds;

                if (gridScore > bestGridScore)
                {
                    bestGridScore = gridScore;
                    bestGridGen = gen;
                    bestGridTime = sw.Elapsed.TotalSeconds;
                }
            }

            evolver.StepGeneration(population);
            gen++;
        }

        // Final grid test
        if (firstSolveGen >= 0)
        {
            var sp = population.AllSpecies[0];
            // Re-evaluate to find current best
            gpu.MaxSteps = 1000;
            gpu.UseGruauFitness = true;
            var (fitness, _) = gpu.EvaluatePopulation(sp.Topology, sp.Individuals, seed: gen);
            int bestIdx = 0;
            for (int i = 1; i < fitness.Length; i++)
                if (fitness[i] > fitness[bestIdx]) bestIdx = i;

            int finalScore = TestGrid625(gpu, sp.Topology, sp.Individuals[bestIdx], grid625);
            if (finalScore > bestGridScore)
            {
                bestGridScore = finalScore;
                bestGridGen = gen;
                bestGridTime = sw.Elapsed.TotalSeconds;
            }
        }

        double wall = sw.Elapsed.TotalSeconds;
        return new GridRunResult(
            firstSolveGen, firstSolveTime,
            bestGridScore, bestGridGen, bestGridTime,
            gen, gen > 0 ? gen / wall : 0, wall);
    }

    /// <summary>
    /// Test champion on 625-grid. Returns positions solved (survived 1000 steps).
    /// Uses EvaluateAllPositions with pop=1 for 625 sequential mini-evals.
    /// </summary>
    private int TestGrid625(GPUDoublePoleEvaluator gpu, SpeciesSpec topology,
        Individual champion, float[][] grid625)
    {
        var savedMaxSteps = gpu.MaxSteps;
        var savedGruau = gpu.UseGruauFitness;

        gpu.MaxSteps = 1000;
        gpu.UseGruauFitness = false; // raw steps for scoring
        gpu.SetStartingPositions(grid625);

        var miniPop = new List<Individual> { champion };
        var (fitness, _) = gpu.EvaluateAllPositions(topology, miniPop, seed: 0);

        // fitness = solvedCount * 1000 + meanSteps, where meanSteps < 1000
        int solvedCount = Math.Min(grid625.Length, (int)(fitness[0] / 1000f));

        gpu.MaxSteps = savedMaxSteps;
        gpu.UseGruauFitness = savedGruau;

        return solvedCount;
    }

    private void WarmupGpu(GPUDoublePoleEvaluator gpu)
    {
        gpu.MaxSteps = 100;
        gpu.ContextSize = 2;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;

        var topology = BuildDenseTopology(2, new[] { 6 });
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 100 };
        var evolver = new Evolver(seed: 0);
        var pop = evolver.InitializePopulation(config, topology);
        gpu.EvaluatePopulation(pop.AllSpecies[0].Topology, pop.AllSpecies[0].Individuals, seed: 0);
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0;
        var sorted = values.OrderBy(v => v).ToList();
        int n = sorted.Count;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
    }

    private record GridRunResult(
        int FirstSolveGen, double FirstSolveTime,
        int BestGridScore, int BestGridGen, double BestGridTime,
        int TotalGens, double GenPerSec, double WallTime);
}
