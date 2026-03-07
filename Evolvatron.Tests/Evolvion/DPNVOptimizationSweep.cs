using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// DPNV Optimization Sweep: find the fastest wall-time configuration for solving
/// Double Pole No-Velocity (non-Markovian) using Elman recurrence.
/// Phase 1: screen 31 configs at 60s each (~31 min total).
/// Phase 2: top configs with longer budgets + multiple seeds.
/// </summary>
public class DPNVOptimizationSweep
{
    private readonly ITestOutputHelper _output;
    private const string ResultsPath = @"C:\Dev\Evolvatron\scratch\dpnv_sweep_phase1.md";

    public DPNVOptimizationSweep(ITestOutputHelper output) => _output = output;

    private record Exp(
        string Name,
        int[] Hidden,
        int Ctx = 3,
        float Density = 1.0f,
        int Species = 1,
        float? Jitter = null,
        float? Reset = null,
        bool NoEdge = false,
        int Elites = 10,
        int Tournament = 5,
        bool Adaptive = false,
        int Budget = 60,
        int MaxInDegree = 10);

    [Fact]
    public void Phase1_Screening()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        Warmup(gpu);

        Directory.CreateDirectory(Path.GetDirectoryName(ResultsPath)!);
        File.WriteAllText(ResultsPath, "");

        Log($"# DPNV Optimization Sweep - Phase 1 Screening\n");
        Log($"- **Device**: {gpu.DeviceName} ({gpu.NumMultiprocessors} SMs, warp={gpu.WarpSize})");
        Log($"- **Optimal pop**: {optPop:N0}");
        Log($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}");
        Log($"- **Mode**: Elman recurrence, no-velocity (3 obs), MaxSteps=10K");
        Log($"- **Seed**: 42 (all experiments)");
        Log($"- **Budget**: 60s per experiment\n");

        var exps = new List<Exp>
        {
            // Group A: Hidden layer width
            new("A1_h4",         new[]{4}),
            new("A2_h6",         new[]{6}),
            new("A3_h8",         new[]{8}),
            new("A4_h10",        new[]{10}),
            new("A5_h15",        new[]{15}),
            new("A6_h20",        new[]{20}),

            // Group B: Depth
            new("B1_4L_8",       new[]{8, 8}),
            new("B2_4L_6",       new[]{6, 6}),
            new("B3_5L_5",       new[]{5, 5, 5}),
            new("B4_5L_4",       new[]{4, 4, 4}),
            new("B5_4L_sp",      new[]{8, 8}, Density: 0f),
            new("B6_5L_sp",      new[]{5, 5, 5}, Density: 0f),

            // Group C: Context size
            new("C1_ctx1",       new[]{10}, Ctx: 1),
            new("C2_ctx2",       new[]{10}, Ctx: 2),
            new("C3_ctx5",       new[]{10}, Ctx: 5),
            new("C4_ctx8",       new[]{10}, Ctx: 8),

            // Group D: Initialization
            new("D1_dense50",    new[]{10}, Density: 0.5f),
            new("D2_dense30",    new[]{10}, Density: 0.3f),
            new("D3_sparse",     new[]{10}, Density: 0f),

            // Group E: Species count (total pop = optPop, divided among species)
            new("E1_sp2",        new[]{10}, Species: 2),
            new("E2_sp4",        new[]{10}, Species: 4),

            // Group F: Mutation variants
            new("F1_no_edge",    new[]{10}, NoEdge: true),
            new("F2_jit15",      new[]{10}, Jitter: 0.15f),
            new("F3_jit30",      new[]{10}, Jitter: 0.30f),
            new("F4_jit15_ne",   new[]{10}, Jitter: 0.15f, NoEdge: true),
            new("F5_elite20",    new[]{10}, Elites: 20),
            new("F6_tourn15",    new[]{10}, Tournament: 15),
            new("F7_gentle",     new[]{10}, Jitter: 0.03f, Reset: 0f, NoEdge: true),
            new("F8_adaptive",   new[]{10}, Adaptive: true),

            // Group H: Combo - tiny hidden with more context
            new("H1_tiny_ctx5",  new[]{4}, Ctx: 5),
        };

        Log($"| {"#",3} | {"Name",-16} | {"Topology",-20} | {"Edges",5} | {"Solved",6} | {"Solve(s)",8} | {"Best",8} | {"Gens",5} | {"Gen/s",6} | {"Wall(s)",7} |");
        Log($"|----:|:-----------------|:-------------------|-------:|-------:|---------:|---------:|------:|-------:|--------:|");

        var results = new List<(Exp exp, string topo, int edges, int solvedGen, double solveTime, float best, int gens, double genPerSec, double wall)>();

        for (int i = 0; i < exps.Count; i++)
        {
            var r = RunExperiment(gpu, exps[i], optPop);
            results.Add((exps[i], r.topo, r.edges, r.solvedGen, r.solveTime, r.best, r.gens, r.genPerSec, r.wall));

            string solvedStr = r.solvedGen >= 0 ? $"{r.solvedGen}" : "DNF";
            string solveStr = r.solvedGen >= 0 ? $"{r.solveTime:F1}" : "-";
            Log($"| {i + 1,3} | {exps[i].Name,-16} | {r.topo,-20} | {r.edges,5} | {solvedStr,6} | {solveStr,8} | {r.best,8:F0} | {r.gens,5} | {r.genPerSec,6:F1} | {r.wall,7:F1} |");
        }

        // Ranking
        Log("\n## Ranking (fastest solve time)\n");
        var solved = results.Where(r => r.solvedGen >= 0).OrderBy(r => r.solveTime).ToList();
        for (int i = 0; i < solved.Count; i++)
        {
            var r = solved[i];
            Log($"{i + 1}. **{r.exp.Name}** - {r.solveTime:F1}s (gen {r.solvedGen}, {r.topo}, {r.edges} edges, {r.genPerSec:F1} gen/s)");
        }

        var unsolved = results.Where(r => r.solvedGen < 0).OrderByDescending(r => r.best).ToList();
        if (unsolved.Count > 0)
        {
            Log("\n## Did Not Solve\n");
            foreach (var r in unsolved)
                Log($"- **{r.exp.Name}**: best={r.best:F0} ({r.topo}, {r.edges} edges)");
        }

        // Analysis
        Log("\n## Analysis\n");
        if (solved.Count > 0)
        {
            var fastest = solved[0];
            Log($"**Fastest**: {fastest.exp.Name} at {fastest.solveTime:F1}s");
            Log($"**Solve rate**: {solved.Count}/{results.Count} ({100.0 * solved.Count / results.Count:F0}%)");
            var byGroup = solved.GroupBy(r => r.exp.Name[0]).OrderBy(g => g.Key);
            foreach (var g in byGroup)
            {
                var times = g.Select(r => r.solveTime).ToList();
                Log($"- Group {g.Key}: {g.Count()} solved, median={times[times.Count / 2]:F1}s");
            }
        }

        Assert.True(solved.Count > 0, "At least one experiment should solve DPNV");
    }

    private (string topo, int edges, int solvedGen, double solveTime, float best, int gens, double genPerSec, double wall)
        RunExperiment(GPUDoublePoleEvaluator gpu, Exp exp, int optPop)
    {
        int totalPop = optPop;
        int popPerSpecies = totalPop / exp.Species;

        gpu.IncludeVelocity = false;
        gpu.ContextSize = exp.Ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = 10_000;

        var topology = BuildTopology(exp.Ctx, exp.Hidden, exp.Density, exp.MaxInDegree);
        string topoStr = string.Join("->", topology.RowCounts);
        int initEdges = topology.Edges.Count;

        var config = new EvolutionConfig
        {
            SpeciesCount = exp.Species,
            IndividualsPerSpecies = popPerSpecies,
            MinSpeciesCount = exp.Species,
            Elites = exp.Elites,
            TournamentSize = exp.Tournament,
            ParentPoolPercentage = 0.5f,
        };

        if (exp.Jitter.HasValue)
            config.MutationRates.WeightJitterStdDev = exp.Jitter.Value;
        if (exp.Reset.HasValue)
            config.MutationRates.WeightReset = exp.Reset.Value;
        if (exp.NoEdge)
        {
            config.EdgeMutations.EdgeAdd = 0;
            config.EdgeMutations.EdgeDeleteRandom = 0;
            config.EdgeMutations.EdgeSplit = 0;
            config.EdgeMutations.EdgeRedirect = 0;
            config.EdgeMutations.EdgeSwap = 0;
        }

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Adaptive jitter state
        float baseJitter = config.MutationRates.WeightJitterStdDev;
        int stagnantGens = 0;
        float prevBest = 0f;

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int solvedGen = -1;
        double solveTime = 0;

        while (sw.Elapsed.TotalSeconds < exp.Budget)
        {
            int totalSolved = 0;
            float globalBest = float.MinValue;

            foreach (var species in population.AllSpecies)
            {
                var (fitness, solved) = gpu.EvaluatePopulation(
                    species.Topology, species.Individuals, seed: gen);

                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = fitness[i];
                    species.Individuals[i] = ind;
                    if (fitness[i] > globalBest) globalBest = fitness[i];
                }
                totalSolved += solved;
            }

            if (globalBest > bestEver) bestEver = globalBest;

            if (totalSolved > 0 && solvedGen < 0)
            {
                solvedGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
                break;
            }

            // Adaptive jitter
            if (exp.Adaptive)
            {
                if (bestEver > prevBest + 1f)
                {
                    config.MutationRates.WeightJitterStdDev =
                        MathF.Max(baseJitter * 0.5f, config.MutationRates.WeightJitterStdDev * 0.9f);
                    prevBest = bestEver;
                    stagnantGens = 0;
                }
                else
                {
                    stagnantGens++;
                    if (stagnantGens > 5)
                    {
                        config.MutationRates.WeightJitterStdDev =
                            MathF.Min(0.5f, config.MutationRates.WeightJitterStdDev * 1.2f);
                    }
                }
            }

            evolver.StepGeneration(population);
            gen++;
        }

        double wall = sw.Elapsed.TotalSeconds;
        double genPerSec = gen > 0 ? gen / wall : 0;

        return (topoStr, initEdges, solvedGen, solveTime, bestEver, gen, genPerSec, wall);
    }

    /// <summary>
    /// Phase 1b: Combine winning factors from Phase 1 screening.
    /// Winners: h=6, ctx=2, jit=0.15, dense(50-100%).
    /// </summary>
    [Fact]
    public void Phase1b_Combinations()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        const string path = @"C:\Dev\Evolvatron\scratch\dpnv_sweep_phase1b.md";
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "");

        void Log2(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        Log2($"# DPNV Sweep - Phase 1b: Combinations\n");
        Log2($"- **Device**: {gpu.DeviceName} ({gpu.NumMultiprocessors} SMs)");
        Log2($"- **Pop**: {optPop:N0}");
        Log2($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}");
        Log2($"- **Budget**: 60s per experiment\n");

        var exps = new List<Exp>
        {
            // Combine h=6 + ctx=2 (the two biggest wins)
            new("combo_base",     new[]{6}, Ctx: 2),
            new("combo_jit15",    new[]{6}, Ctx: 2, Jitter: 0.15f),
            new("combo_jit20",    new[]{6}, Ctx: 2, Jitter: 0.20f),
            new("combo_jit25",    new[]{6}, Ctx: 2, Jitter: 0.25f),
            new("combo_jit15_ne", new[]{6}, Ctx: 2, Jitter: 0.15f, NoEdge: true),
            new("combo_d50",      new[]{6}, Ctx: 2, Density: 0.5f),
            new("combo_d50_j15",  new[]{6}, Ctx: 2, Density: 0.5f, Jitter: 0.15f),
            new("combo_d70",      new[]{6}, Ctx: 2, Density: 0.7f),
            new("combo_d70_j15",  new[]{6}, Ctx: 2, Density: 0.7f, Jitter: 0.15f),

            // Width variants around h=6
            new("h5_ctx2_j15",    new[]{5}, Ctx: 2, Jitter: 0.15f),
            new("h7_ctx2_j15",    new[]{7}, Ctx: 2, Jitter: 0.15f),
            new("h8_ctx2_j15",    new[]{8}, Ctx: 2, Jitter: 0.15f),

            // Even smaller ctx
            new("h6_ctx1_j15",    new[]{6}, Ctx: 1, Jitter: 0.15f),

            // Selection pressure variants on best combo
            new("combo_j15_t3",   new[]{6}, Ctx: 2, Jitter: 0.15f, Tournament: 3),
            new("combo_j15_e20",  new[]{6}, Ctx: 2, Jitter: 0.15f, Elites: 20),
            new("combo_j15_e3",   new[]{6}, Ctx: 2, Jitter: 0.15f, Elites: 3),

            // 4-layer with combo params
            new("4L_ctx2_j15",    new[]{6, 6}, Ctx: 2, Jitter: 0.15f),
        };

        Log2($"| {"#",3} | {"Name",-18} | {"Topology",-16} | {"Edges",5} | {"Solved",6} | {"Solve(s)",8} | {"Best",8} | {"Gens",5} | {"Gen/s",6} | {"Wall(s)",7} |");
        Log2($"|----:|:-------------------|:-----------------|-------:|-------:|---------:|---------:|------:|-------:|--------:|");

        var results = new List<(Exp exp, string topo, int edges, int solvedGen, double solveTime, float best, int gens, double genPerSec, double wall)>();

        for (int i = 0; i < exps.Count; i++)
        {
            var r = RunExperiment(gpu, exps[i], optPop);
            results.Add((exps[i], r.topo, r.edges, r.solvedGen, r.solveTime, r.best, r.gens, r.genPerSec, r.wall));

            string solvedStr = r.solvedGen >= 0 ? $"{r.solvedGen}" : "DNF";
            string solveStr = r.solvedGen >= 0 ? $"{r.solveTime:F1}" : "-";
            Log2($"| {i + 1,3} | {exps[i].Name,-18} | {r.topo,-16} | {r.edges,5} | {solvedStr,6} | {solveStr,8} | {r.best,8:F0} | {r.gens,5} | {r.genPerSec,6:F1} | {r.wall,7:F1} |");
        }

        Log2("\n## Ranking\n");
        var solved = results.Where(r => r.solvedGen >= 0).OrderBy(r => r.solveTime).ToList();
        for (int i = 0; i < solved.Count; i++)
        {
            var r = solved[i];
            Log2($"{i + 1}. **{r.exp.Name}** - {r.solveTime:F1}s (gen {r.solvedGen}, {r.topo}, {r.edges} edges, {r.genPerSec:F1} gen/s)");
        }

        Assert.True(solved.Count > 0, "At least one combo should solve DPNV");
    }

    /// <summary>
    /// Phase 2: Top configs with multiple seeds for statistical significance.
    /// Run AFTER Phase 1b identifies the best combinations.
    /// </summary>
    [Fact]
    public void Phase2_MultiSeed()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        const string path = @"C:\Dev\Evolvatron\scratch\dpnv_sweep_phase2.md";
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "");

        void Log2(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        Log2($"# DPNV Sweep - Phase 2: Multi-Seed Validation\n");
        Log2($"- **Device**: {gpu.DeviceName}");
        Log2($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}");
        Log2($"- **Budget**: 120s per run, 10 seeds each\n");

        // Top configs from Phase 1b results
        var configs = new List<Exp>
        {
            new("j20",            new[]{6}, Ctx: 2, Jitter: 0.20f, Budget: 120),
            new("j20_ne",         new[]{6}, Ctx: 2, Jitter: 0.20f, NoEdge: true, Budget: 120),
            new("j20_e20",        new[]{6}, Ctx: 2, Jitter: 0.20f, Elites: 20, Budget: 120),
            new("j20_ne_e20",     new[]{6}, Ctx: 2, Jitter: 0.20f, NoEdge: true, Elites: 20, Budget: 120),
            new("j15_ne",         new[]{6}, Ctx: 2, Jitter: 0.15f, NoEdge: true, Budget: 120),
            new("j15",            new[]{6}, Ctx: 2, Jitter: 0.15f, Budget: 120),
            new("baseline_ctx3",  new[]{6}, Budget: 120),
        };

        int[] seeds = { 42, 123, 456, 789, 1337, 2024, 3141, 9999, 55555, 77777 };

        Log2($"| {"Config",-18} | {"Seed",5} | {"Solved",6} | {"Solve(s)",8} | {"Best",8} | {"Gens",5} | {"Gen/s",6} |");
        Log2($"|:-------------------|------:|-------:|---------:|---------:|------:|-------:|");

        foreach (var cfg in configs)
        {
            var solveTimes = new List<double>();
            foreach (int seed in seeds)
            {
                // Override seed by creating a new evolver inside RunExperiment
                // For now, we vary seed via the EvaluatePopulation seed parameter
                var r = RunExperimentWithSeed(gpu, cfg, optPop, seed);

                string solvedStr = r.solvedGen >= 0 ? $"{r.solvedGen}" : "DNF";
                string solveStr = r.solvedGen >= 0 ? $"{r.solveTime:F1}" : "-";
                Log2($"| {cfg.Name,-16} | {seed,5} | {solvedStr,6} | {solveStr,8} | {r.best,8:F0} | {r.gens,5} | {r.genPerSec,6:F1} |");

                if (r.solvedGen >= 0) solveTimes.Add(r.solveTime);
            }

            if (solveTimes.Count > 0)
            {
                solveTimes.Sort();
                double median = solveTimes[solveTimes.Count / 2];
                Log2($"| {cfg.Name + " (med)",-16} |       | {solveTimes.Count}/{seeds.Length} | {median,8:F1} |          |       |        |");
            }
            else
            {
                Log2($"| {cfg.Name + " (med)",-16} |       |   0/{seeds.Length} |        - |          |       |        |");
            }
        }
    }

    private (string topo, int edges, int solvedGen, double solveTime, float best, int gens, double genPerSec, double wall)
        RunExperimentWithSeed(GPUDoublePoleEvaluator gpu, Exp exp, int optPop, int seed)
    {
        int totalPop = optPop;
        int popPerSpecies = totalPop / exp.Species;

        gpu.IncludeVelocity = false;
        gpu.ContextSize = exp.Ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = 10_000;

        var topology = BuildTopology(exp.Ctx, exp.Hidden, exp.Density, exp.MaxInDegree);
        string topoStr = string.Join("->", topology.RowCounts);
        int initEdges = topology.Edges.Count;

        var config = new EvolutionConfig
        {
            SpeciesCount = exp.Species,
            IndividualsPerSpecies = popPerSpecies,
            MinSpeciesCount = exp.Species,
            Elites = exp.Elites,
            TournamentSize = exp.Tournament,
            ParentPoolPercentage = 0.5f,
        };

        if (exp.Jitter.HasValue)
            config.MutationRates.WeightJitterStdDev = exp.Jitter.Value;
        if (exp.Reset.HasValue)
            config.MutationRates.WeightReset = exp.Reset.Value;
        if (exp.NoEdge)
        {
            config.EdgeMutations.EdgeAdd = 0;
            config.EdgeMutations.EdgeDeleteRandom = 0;
            config.EdgeMutations.EdgeSplit = 0;
            config.EdgeMutations.EdgeRedirect = 0;
            config.EdgeMutations.EdgeSwap = 0;
        }

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        float baseJitter = config.MutationRates.WeightJitterStdDev;
        int stagnantGens = 0;
        float prevBest = 0f;

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int solvedGen = -1;
        double solveTime = 0;

        while (sw.Elapsed.TotalSeconds < exp.Budget)
        {
            int totalSolved = 0;
            float globalBest = float.MinValue;

            foreach (var species in population.AllSpecies)
            {
                var (fitness, solved) = gpu.EvaluatePopulation(
                    species.Topology, species.Individuals, seed: gen);

                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = fitness[i];
                    species.Individuals[i] = ind;
                    if (fitness[i] > globalBest) globalBest = fitness[i];
                }
                totalSolved += solved;
            }

            if (globalBest > bestEver) bestEver = globalBest;

            if (totalSolved > 0 && solvedGen < 0)
            {
                solvedGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
                break;
            }

            if (exp.Adaptive)
            {
                if (bestEver > prevBest + 1f)
                {
                    config.MutationRates.WeightJitterStdDev =
                        MathF.Max(baseJitter * 0.5f, config.MutationRates.WeightJitterStdDev * 0.9f);
                    prevBest = bestEver;
                    stagnantGens = 0;
                }
                else
                {
                    stagnantGens++;
                    if (stagnantGens > 5)
                    {
                        config.MutationRates.WeightJitterStdDev =
                            MathF.Min(0.5f, config.MutationRates.WeightJitterStdDev * 1.2f);
                    }
                }
            }

            evolver.StepGeneration(population);
            gen++;
        }

        double wall = sw.Elapsed.TotalSeconds;
        double genPerSec = gen > 0 ? gen / wall : 0;

        return (topoStr, initEdges, solvedGen, solveTime, bestEver, gen, genPerSec, wall);
    }

    private static SpeciesSpec BuildTopology(int ctx, int[] hidden, float density, int maxInDegree = 10)
    {
        int inputSize = 3 + ctx;
        int outputSize = 1 + ctx;

        var builder = new SpeciesBuilder().AddInputRow(inputSize);

        foreach (int h in hidden)
            builder.AddHiddenRow(h,
                ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.ReLU,
                ActivationType.LeakyReLU, ActivationType.Softsign, ActivationType.Sin);

        builder.AddOutputRow(outputSize, ActivationType.Tanh)
               .WithMaxInDegree(maxInDegree);

        if (density <= 0f)
            builder.InitializeSparse(new Random(42));
        else
            builder.InitializeDense(new Random(42), density);

        return builder.Build();
    }

    private void Warmup(GPUDoublePoleEvaluator gpu)
    {
        var topo = BuildTopology(3, new[] { 10 }, 1.0f);
        var ev = new Evolver(seed: 0);
        var pop = ev.InitializePopulation(
            new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 10 }, topo);
        gpu.MaxSteps = 10;
        gpu.IncludeVelocity = true;
        gpu.ContextSize = 0;
        gpu.EvaluatePopulation(pop.AllSpecies[0].Topology, pop.AllSpecies[0].Individuals, seed: 0);
    }

    /// <summary>
    /// Phase 3: Sparse init + Gruau fitness + edge mutations.
    /// Tests whether the fixed TryEdgeAdd (row 1 targeting) enables
    /// sparse complexification, and whether Gruau anti-jiggle bonus
    /// creates meaningful selection pressure.
    /// </summary>
    [Fact]
    public void Phase3_SparseGruau()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        const string path = @"C:\Dev\Evolvatron\scratch\dpnv_sweep_phase3.md";
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "");

        void Log3(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        Log3($"# DPNV Sweep - Phase 3: Sparse Init + Gruau Fitness\n");
        Log3($"- **Device**: {gpu.DeviceName} ({gpu.NumMultiprocessors} SMs)");
        Log3($"- **Pop**: {optPop:N0}");
        Log3($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}");
        Log3($"- **Budget**: 120s per experiment");
        Log3($"- **Key change**: TryEdgeAdd now targets row >= 1 (was >= 2)");
        Log3($"- **Gruau**: anti-jiggle bonus for survivors (0.75 * MaxSteps / jiggleSum)\n");

        var exps = new List<(Exp exp, bool gruau)>
        {
            // Group S: Sparse init (density=0) with edge mutations — does it work now?
            (new("S1_sp_6",        new[]{6},       Ctx: 2, Density: 0f, Budget: 120), false),
            (new("S2_sp_10",       new[]{10},      Ctx: 2, Density: 0f, Budget: 120), false),
            (new("S3_sp_8_8",      new[]{8, 8},    Ctx: 2, Density: 0f, Budget: 120), false),
            (new("S4_sp_12",       new[]{12},      Ctx: 2, Density: 0f, Budget: 120), false),
            (new("S5_sp_6_6_6",    new[]{6, 6, 6}, Ctx: 2, Density: 0f, Budget: 120), false),
            (new("S6_sp_10_j15",   new[]{10},      Ctx: 2, Density: 0f, Jitter: 0.15f, Budget: 120), false),
            (new("S7_sp_16",       new[]{16},      Ctx: 2, Density: 0f, Budget: 120), false),
            (new("S8_sp_8_8_j15",  new[]{8, 8},    Ctx: 2, Density: 0f, Jitter: 0.15f, Budget: 120), false),

            // Group G: Gruau fitness on dense winners (from Phase 2)
            (new("G1_dense_noG",   new[]{6}, Ctx: 2, Jitter: 0.15f, NoEdge: true, Budget: 120), false),
            (new("G2_dense_G",     new[]{6}, Ctx: 2, Jitter: 0.15f, NoEdge: true, Budget: 120), true),
            (new("G3_dense10_noG", new[]{10}, Ctx: 2, Jitter: 0.15f, NoEdge: true, Budget: 120), false),
            (new("G4_dense10_G",   new[]{10}, Ctx: 2, Jitter: 0.15f, NoEdge: true, Budget: 120), true),

            // Group SG: Sparse + Gruau
            (new("SG1_sp10_G",     new[]{10},      Ctx: 2, Density: 0f, Budget: 120), true),
            (new("SG2_sp88_G",     new[]{8, 8},    Ctx: 2, Density: 0f, Budget: 120), true),
            (new("SG3_sp12_j15_G", new[]{12},      Ctx: 2, Density: 0f, Jitter: 0.15f, Budget: 120), true),
            (new("SG4_sp16_G",     new[]{16},      Ctx: 2, Density: 0f, Budget: 120), true),

            // Group X: Larger sparse for complexification headroom
            (new("X1_sp20",        new[]{20},           Ctx: 2, Density: 0f, Budget: 120), false),
            (new("X2_sp12_12",     new[]{12, 12},       Ctx: 2, Density: 0f, Budget: 120), false),
            (new("X3_sp8_8_8",     new[]{8, 8, 8},      Ctx: 2, Density: 0f, Budget: 120), false),
            (new("X4_sp20_G",      new[]{20},           Ctx: 2, Density: 0f, Budget: 120), true),
            (new("X5_sp12_12_G",   new[]{12, 12},       Ctx: 2, Density: 0f, Budget: 120), true),
        };

        Log3($"| {"#",3} | {"Name",-18} | {"Gruau",5} | {"Topology",-16} | {"Edges",5} | {"Solved",6} | {"Solve(s)",8} | {"Best",10} | {"Gens",5} | {"Gen/s",6} | {"Wall(s)",7} |");
        Log3($"|----:|:-------------------|------:|:-----------------|-------:|-------:|---------:|-----------:|------:|-------:|--------:|");

        var results = new List<(Exp exp, bool gruau, string topo, int edges, int solvedGen, double solveTime, float best, int gens, double genPerSec, double wall)>();

        for (int i = 0; i < exps.Count; i++)
        {
            var (exp, gruau) = exps[i];
            var r = RunExperimentGruau(gpu, exp, optPop, 42, gruau);
            results.Add((exp, gruau, r.topo, r.edges, r.solvedGen, r.solveTime, r.best, r.gens, r.genPerSec, r.wall));

            string solvedStr = r.solvedGen >= 0 ? $"{r.solvedGen}" : "DNF";
            string solveStr = r.solvedGen >= 0 ? $"{r.solveTime:F1}" : "-";
            string gruauStr = gruau ? "YES" : "no";
            Log3($"| {i + 1,3} | {exp.Name,-18} | {gruauStr,5} | {r.topo,-16} | {r.edges,5} | {solvedStr,6} | {solveStr,8} | {r.best,10:F1} | {r.gens,5} | {r.genPerSec,6:F1} | {r.wall,7:F1} |");
        }

        // Ranking
        Log3("\n## Ranking (fastest solve time)\n");
        var solved = results.Where(r => r.solvedGen >= 0).OrderBy(r => r.solveTime).ToList();
        for (int i = 0; i < solved.Count; i++)
        {
            var r = solved[i];
            string g = r.gruau ? " [Gruau]" : "";
            Log3($"{i + 1}. **{r.exp.Name}**{g} - {r.solveTime:F1}s (gen {r.solvedGen}, {r.topo}, {r.edges} edges)");
        }

        var unsolved = results.Where(r => r.solvedGen < 0).OrderByDescending(r => r.best).ToList();
        if (unsolved.Count > 0)
        {
            Log3("\n## Did Not Solve\n");
            foreach (var r in unsolved)
            {
                string g = r.gruau ? " [Gruau]" : "";
                Log3($"- **{r.exp.Name}**{g}: best={r.best:F1} ({r.topo}, {r.edges} edges)");
            }
        }

        // Analysis: sparse vs dense, Gruau vs no-Gruau
        Log3("\n## Analysis\n");
        var sparseResults = results.Where(r => r.exp.Density <= 0f).ToList();
        var denseResults = results.Where(r => r.exp.Density > 0f).ToList();
        var gruauResults = results.Where(r => r.gruau).ToList();
        var noGruauResults = results.Where(r => !r.gruau).ToList();

        Log3($"**Sparse**: {sparseResults.Count(r => r.solvedGen >= 0)}/{sparseResults.Count} solved");
        Log3($"**Dense**: {denseResults.Count(r => r.solvedGen >= 0)}/{denseResults.Count} solved");
        Log3($"**Gruau**: {gruauResults.Count(r => r.solvedGen >= 0)}/{gruauResults.Count} solved");
        Log3($"**No Gruau**: {noGruauResults.Count(r => r.solvedGen >= 0)}/{noGruauResults.Count} solved");

        // Paired comparisons (G1 vs G2, G3 vs G4, etc.)
        Log3("\n### Paired Comparisons (Gruau effect)\n");
        var pairs = new[] { ("G1_dense_noG", "G2_dense_G"), ("G3_dense10_noG", "G4_dense10_G") };
        foreach (var (noG, withG) in pairs)
        {
            var rNoG = results.FirstOrDefault(r => r.exp.Name == noG);
            var rG = results.FirstOrDefault(r => r.exp.Name == withG);
            if (rNoG.exp != null && rG.exp != null)
            {
                string noGStr = rNoG.solvedGen >= 0 ? $"{rNoG.solveTime:F1}s" : $"DNF (best={rNoG.best:F0})";
                string gStr = rG.solvedGen >= 0 ? $"{rG.solveTime:F1}s" : $"DNF (best={rG.best:F0})";
                Log3($"- {noG} vs {withG}: {noGStr} vs {gStr}");
            }
        }
    }

    private (string topo, int edges, int solvedGen, double solveTime, float best, int gens, double genPerSec, double wall)
        RunExperimentGruau(GPUDoublePoleEvaluator gpu, Exp exp, int optPop, int seed, bool gruau)
    {
        int totalPop = optPop;
        int popPerSpecies = totalPop / exp.Species;

        gpu.IncludeVelocity = false;
        gpu.ContextSize = exp.Ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = 10_000;
        gpu.UseGruauFitness = gruau;

        var topology = BuildTopology(exp.Ctx, exp.Hidden, exp.Density, exp.MaxInDegree);
        string topoStr = string.Join("->", topology.RowCounts);
        int initEdges = topology.Edges.Count;

        var config = new EvolutionConfig
        {
            SpeciesCount = exp.Species,
            IndividualsPerSpecies = popPerSpecies,
            MinSpeciesCount = exp.Species,
            Elites = exp.Elites,
            TournamentSize = exp.Tournament,
            ParentPoolPercentage = 0.5f,
        };

        if (exp.Jitter.HasValue)
            config.MutationRates.WeightJitterStdDev = exp.Jitter.Value;
        if (exp.Reset.HasValue)
            config.MutationRates.WeightReset = exp.Reset.Value;
        if (exp.NoEdge)
        {
            config.EdgeMutations.EdgeAdd = 0;
            config.EdgeMutations.EdgeDeleteRandom = 0;
            config.EdgeMutations.EdgeSplit = 0;
            config.EdgeMutations.EdgeRedirect = 0;
            config.EdgeMutations.EdgeSwap = 0;
        }

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int solvedGen = -1;
        double solveTime = 0;

        while (sw.Elapsed.TotalSeconds < exp.Budget)
        {
            int totalSolved = 0;
            float globalBest = float.MinValue;

            foreach (var species in population.AllSpecies)
            {
                var (fitness, solved) = gpu.EvaluatePopulation(
                    species.Topology, species.Individuals, seed: gen);

                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = fitness[i];
                    species.Individuals[i] = ind;
                    if (fitness[i] > globalBest) globalBest = fitness[i];
                }
                totalSolved += solved;
            }

            if (globalBest > bestEver) bestEver = globalBest;

            if (totalSolved > 0 && solvedGen < 0)
            {
                solvedGen = gen;
                solveTime = sw.Elapsed.TotalSeconds;
                break;
            }

            evolver.StepGeneration(population);
            gen++;
        }

        double wall = sw.Elapsed.TotalSeconds;
        double genPerSec = gen > 0 ? gen / wall : 0;

        return (topoStr, initEdges, solvedGen, solveTime, bestEver, gen, genPerSec, wall);
    }

    private void Log(string line)
    {
        _output.WriteLine(line);
        File.AppendAllText(ResultsPath, line + "\n");
    }
}
