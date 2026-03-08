using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Comprehensive study of sparse initialization + complexification for DPNV.
/// Goal: find WINNING SPREADS (topologies that reliably solve) and FAILING SPREADS
/// (topologies that reliably stagnate). NOT looking for a single optimal config.
///
/// Design:
///   Phase A: Topology Census — 25+ topologies × 5 seeds each, 120s budget
///            Map which shapes solve and which don't. Capture edge counts, solve times, fitness curves.
///   Phase B: Species Diversity — winning topologies from A × {1,3,6,12} species, 5 seeds each
///            Does multi-species help sparse complexification?
///   Phase C: Edge Mutation Rates — winning topologies × aggressive/moderate/gentle edge mutation, 5 seeds
///            What rate of complexification works?
///   Phase D: Deep Dive — best configs from A-C, 20 seeds each, 180s budget
///            Statistical validation of winning and failing spreads.
/// </summary>
public class SparseComplexificationStudy
{
    private readonly ITestOutputHelper _output;
    private const string BasePath = @"C:\Dev\Evolvatron\scratch\sparse_study";

    public SparseComplexificationStudy(ITestOutputHelper output) => _output = output;

    #region Topology Definition

    private record Topo(string Name, int[] Hidden, string Category);

    private static readonly Topo[] AllTopologies = new[]
    {
        // Category: Shallow (1 hidden layer) — baseline
        new Topo("sh_4",      new[]{4},           "shallow"),
        new Topo("sh_6",      new[]{6},           "shallow"),
        new Topo("sh_8",      new[]{8},           "shallow"),
        new Topo("sh_10",     new[]{10},          "shallow"),
        new Topo("sh_12",     new[]{12},          "shallow"),
        new Topo("sh_16",     new[]{16},          "shallow"),
        new Topo("sh_20",     new[]{20},          "shallow"),

        // Category: Medium (2 hidden layers) — the question is width vs depth
        new Topo("md_4_4",    new[]{4, 4},        "medium"),
        new Topo("md_6_6",    new[]{6, 6},        "medium"),
        new Topo("md_8_8",    new[]{8, 8},        "medium"),
        new Topo("md_6_4",    new[]{6, 4},        "medium"),  // funnel
        new Topo("md_4_6",    new[]{4, 6},        "medium"),  // hourglass
        new Topo("md_10_6",   new[]{10, 6},       "medium"),  // wide funnel
        new Topo("md_6_10",   new[]{6, 10},       "medium"),  // wide hourglass

        // Category: Deep (3 hidden layers) — the user's preferred shape
        new Topo("dp_4_4_4",  new[]{4, 4, 4},     "deep"),
        new Topo("dp_6_6_6",  new[]{6, 6, 6},     "deep"),    // THE preferred shape
        new Topo("dp_8_8_8",  new[]{8, 8, 8},     "deep"),
        new Topo("dp_6_4_6",  new[]{6, 4, 6},     "deep"),    // bottleneck
        new Topo("dp_4_6_4",  new[]{4, 6, 4},     "deep"),    // diamond
        new Topo("dp_8_6_4",  new[]{8, 6, 4},     "deep"),    // funnel
        new Topo("dp_4_6_8",  new[]{4, 6, 8},     "deep"),    // expanding

        // Category: Very Deep (4 hidden layers) — does depth help complexification?
        new Topo("vd_4_4_4_4",  new[]{4, 4, 4, 4},  "vdeep"),
        new Topo("vd_6_6_6_6",  new[]{6, 6, 6, 6},  "vdeep"),
        new Topo("vd_6_4_4_6",  new[]{6, 4, 4, 6},  "vdeep"),  // double bottleneck
        new Topo("vd_4_6_6_4",  new[]{4, 6, 6, 4},  "vdeep"),  // barrel

        // Category: Asymmetric — real-world topologies won't be uniform
        new Topo("as_12_4",     new[]{12, 4},         "asym"),
        new Topo("as_4_12",     new[]{4, 12},         "asym"),
        new Topo("as_8_4_8",    new[]{8, 4, 8},       "asym"),   // hourglass deep
        new Topo("as_10_4_4_10",new[]{10, 4, 4, 10},  "asym"),   // double hourglass
    };

    #endregion

    #region Phase A: Topology Census

    /// <summary>
    /// Phase A: Run every topology × 5 seeds. Map solve rates and timing.
    /// Budget: ~30 topologies × 5 seeds × 120s max = ~5 hours worst case (many solve fast or DNF early).
    /// </summary>
    [Fact]
    public void PhaseA_TopologyCensus()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        string path = Path.Combine(BasePath, "phase_a_topology_census.md");
        Directory.CreateDirectory(BasePath);
        File.WriteAllText(path, "");

        void Log(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        int[] seeds = { 42, 123, 456, 789, 1337 };

        Log($"# Phase A: Topology Census\n");
        Log($"- **Device**: {gpu.DeviceName} ({gpu.NumMultiprocessors} SMs)");
        Log($"- **Pop**: {optPop:N0} (1 species)");
        Log($"- **Config**: Elman ctx=2, jit=0.15, Gruau ON, edge mutations ON");
        Log($"- **Budget**: 120s per run, {seeds.Length} seeds per topology");
        Log($"- **Topologies**: {AllTopologies.Length}");
        Log($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        // Detailed results table
        Log($"| {"Topology",-16} | {"Cat",-7} | {"Shape",-16} | {"Edges",5} | {"PossEdg",7} | {"Density",7} | {"Seed",5} | {"Solved",6} | {"Time(s)",7} | {"Best",10} | {"Gens",5} | {"Gen/s",5} | {"FinalEdg",8} |");
        Log($"|:-----------------|:--------|:-----------------|-------:|--------:|--------:|------:|-------:|--------:|-----------:|------:|------:|---------:|");

        var summaries = new List<TopoSummary>();

        foreach (var topo in AllTopologies)
        {
            var runs = new List<RunResult>();

            foreach (int seed in seeds)
            {
                var r = RunSparseExperiment(gpu, topo.Hidden, optPop, seed,
                    species: 1, jitter: 0.15f, gruau: true,
                    edgeAdd: 0.05f, edgeDelete: 0.02f, edgeSplit: 0.01f,
                    edgeRedirect: 0.05f, edgeSwap: 0.02f,
                    budget: 120);
                runs.Add(r);

                string solvedStr = r.SolvedGen >= 0 ? $"{r.SolvedGen}" : "DNF";
                string timeStr = r.SolvedGen >= 0 ? $"{r.SolveTime:F1}" : "-";
                Log($"| {topo.Name,-16} | {topo.Category,-7} | {r.Shape,-16} | {r.InitEdges,5} | {r.PossibleEdges,7} | {r.InitDensity,7:P0} | {seed,5} | {solvedStr,6} | {timeStr,7} | {r.Best,10:F1} | {r.Gens,5} | {r.GenPerSec,5:F1} | {r.FinalEdges,8} |");
            }

            var summary = ComputeSummary(topo, runs);
            summaries.Add(summary);
        }

        // Summary table
        Log("\n## Summary by Topology\n");
        Log($"| {"Topology",-16} | {"Cat",-7} | {"SolveRate",9} | {"Med(s)",6} | {"Min(s)",6} | {"Max(s)",6} | {"InitEdg",7} | {"PossEdg",7} | {"MedFinal",8} | {"Verdict",-12} |");
        Log($"|:-----------------|:--------|----------:|-------:|-------:|-------:|--------:|--------:|---------:|:-------------|");

        foreach (var s in summaries.OrderBy(s => CategoryOrder(s.Category)).ThenBy(s => s.MedianSolveTime ?? 999))
        {
            string rate = $"{s.SolveCount}/{s.TotalRuns}";
            string med = s.MedianSolveTime.HasValue ? $"{s.MedianSolveTime:F1}" : "-";
            string min = s.MinSolveTime.HasValue ? $"{s.MinSolveTime:F1}" : "-";
            string max = s.MaxSolveTime.HasValue ? $"{s.MaxSolveTime:F1}" : "-";
            string verdict = s.SolveCount == s.TotalRuns ? "RELIABLE" :
                             s.SolveCount > 0 ? "PARTIAL" : "FAILS";
            Log($"| {s.Name,-16} | {s.Category,-7} | {rate,9} | {med,6} | {min,6} | {max,6} | {s.InitEdges,7} | {s.PossibleEdges,7} | {s.MedianFinalEdges,8} | {verdict,-12} |");
        }

        // Category analysis
        Log("\n## Category Analysis\n");
        foreach (var cat in summaries.GroupBy(s => s.Category).OrderBy(g => CategoryOrder(g.Key)))
        {
            int total = cat.Sum(s => s.TotalRuns);
            int solved = cat.Sum(s => s.SolveCount);
            var solvedTimes = cat.Where(s => s.MedianSolveTime.HasValue).Select(s => s.MedianSolveTime!.Value).ToList();
            string medTime = solvedTimes.Count > 0 ? $"{Median(solvedTimes):F1}s" : "-";
            Log($"- **{cat.Key}**: {solved}/{total} solved ({100.0 * solved / total:F0}%), median solve time: {medTime}");
        }

        // Winning spread
        var reliable = summaries.Where(s => s.SolveCount == s.TotalRuns).OrderBy(s => s.MedianSolveTime).ToList();
        var partial = summaries.Where(s => s.SolveCount > 0 && s.SolveCount < s.TotalRuns).OrderByDescending(s => s.SolveCount).ToList();
        var failing = summaries.Where(s => s.SolveCount == 0).OrderByDescending(s => s.MedianBest).ToList();

        Log("\n## Winning Spread (RELIABLE: all seeds solve)\n");
        foreach (var s in reliable)
            Log($"- **{s.Name}** ({s.Category}): median {s.MedianSolveTime:F1}s, edges {s.InitEdges}→{s.MedianFinalEdges}");

        Log("\n## Partial Spread (some seeds solve)\n");
        foreach (var s in partial)
            Log($"- **{s.Name}** ({s.Category}): {s.SolveCount}/{s.TotalRuns}, median best={s.MedianBest:F0}");

        Log("\n## Failing Spread (no seeds solve)\n");
        foreach (var s in failing)
            Log($"- **{s.Name}** ({s.Category}): best={s.MedianBest:F0}, edges {s.InitEdges}→{s.MedianFinalEdges}");

        Assert.True(reliable.Count > 0 || partial.Count > 0, "At least some topologies should solve");
    }

    #endregion

    #region Phase B: Species Diversity

    /// <summary>
    /// Phase B: Take winning + borderline topologies from Phase A, vary species count.
    /// Tests whether multi-species helps sparse complexification.
    /// </summary>
    [Fact]
    public void PhaseB_SpeciesDiversity()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        string path = Path.Combine(BasePath, "phase_b_species_diversity.md");
        Directory.CreateDirectory(BasePath);
        File.WriteAllText(path, "");

        void Log(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        // Select topologies to test — mix of shapes and categories
        var topos = new Topo[]
        {
            // From Phase 3 winners
            new("sh_6",       new[]{6},           "shallow"),
            new("sh_12",      new[]{12},          "shallow"),
            new("sh_20",      new[]{20},          "shallow"),
            // Deep shapes — the user's preference
            new("dp_6_6_6",   new[]{6, 6, 6},     "deep"),
            new("dp_4_4_4",   new[]{4, 4, 4},     "deep"),
            new("dp_8_8_8",   new[]{8, 8, 8},     "deep"),
            new("dp_8_6_4",   new[]{8, 6, 4},     "deep"),
            // Medium
            new("md_8_8",     new[]{8, 8},        "medium"),
            new("md_6_6",     new[]{6, 6},        "medium"),
            // Very deep
            new("vd_6_6_6_6", new[]{6, 6, 6, 6},  "vdeep"),
            new("vd_4_4_4_4", new[]{4, 4, 4, 4},  "vdeep"),
        };

        int[] speciesCounts = { 1, 3, 6, 12 };
        int[] seeds = { 42, 123, 456, 789, 1337 };

        Log($"# Phase B: Species Diversity × Sparse Topologies\n");
        Log($"- **Device**: {gpu.DeviceName}");
        Log($"- **Total Pop**: {optPop:N0} (divided among species)");
        Log($"- **Config**: Elman ctx=2, jit=0.15, Gruau ON, edge mutations ON");
        Log($"- **Budget**: 120s per run");
        Log($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        Log($"| {"Topology",-14} | {"Species",7} | {"PopPerSp",8} | {"Seed",5} | {"Shape",-14} | {"InitEdg",7} | {"Solved",6} | {"Time(s)",7} | {"Best",10} | {"Gens",5} | {"FinalEdg",8} |");
        Log($"|:---------------|--------:|---------:|------:|:---------------|--------:|-------:|--------:|-----------:|------:|---------:|");

        var summaries = new List<(Topo topo, int species, TopoSummary summary)>();

        foreach (var topo in topos)
        {
            foreach (int sp in speciesCounts)
            {
                var runs = new List<RunResult>();

                foreach (int seed in seeds)
                {
                    var r = RunSparseExperiment(gpu, topo.Hidden, optPop, seed,
                        species: sp, jitter: 0.15f, gruau: true,
                        edgeAdd: 0.05f, edgeDelete: 0.02f, edgeSplit: 0.01f,
                        edgeRedirect: 0.05f, edgeSwap: 0.02f,
                        budget: 120);
                    runs.Add(r);

                    string solvedStr = r.SolvedGen >= 0 ? $"{r.SolvedGen}" : "DNF";
                    string timeStr = r.SolvedGen >= 0 ? $"{r.SolveTime:F1}" : "-";
                    Log($"| {topo.Name,-14} | {sp,7} | {optPop / sp,8} | {seed,5} | {r.Shape,-14} | {r.InitEdges,7} | {solvedStr,6} | {timeStr,7} | {r.Best,10:F1} | {r.Gens,5} | {r.FinalEdges,8} |");
                }

                var summary = ComputeSummary(topo, runs);
                summaries.Add((topo, sp, summary));
            }
        }

        // Summary: species effect per topology
        Log("\n## Species Effect Summary\n");
        Log($"| {"Topology",-14} | {"1sp",8} | {"3sp",8} | {"6sp",8} | {"12sp",8} | {"Best#sp",7} |");
        Log($"|:---------------|-------:|-------:|-------:|-------:|--------:|");

        foreach (var topo in topos)
        {
            var row = new StringBuilder();
            row.Append($"| {topo.Name,-14} |");
            int bestSp = 0;
            double bestTime = double.MaxValue;

            foreach (int sp in speciesCounts)
            {
                var s = summaries.First(x => x.topo.Name == topo.Name && x.species == sp).summary;
                string cell = s.SolveCount > 0 ? $"{s.SolveCount}/{s.TotalRuns} {s.MedianSolveTime:F0}s" : $"0/{s.TotalRuns}";
                row.Append($" {cell,8} |");
                if (s.MedianSolveTime.HasValue && s.MedianSolveTime.Value < bestTime)
                {
                    bestTime = s.MedianSolveTime.Value;
                    bestSp = sp;
                }
            }
            row.Append($" {(bestSp > 0 ? bestSp.ToString() : "-"),7} |");
            Log(row.ToString());
        }
    }

    #endregion

    #region Phase B2: Species Diversity (Full GPU Pop Per Species)

    /// <summary>
    /// Phase B2: Re-run species diversity study with CORRECT sizing.
    /// Each species gets the full GPU-optimal population (16K on RTX 4090).
    /// More species = more sequential kernel launches per generation = slower gens,
    /// but each launch fully saturates the GPU.
    /// This isolates the pure effect of topology diversity from population starvation.
    /// </summary>
    [Fact]
    public void PhaseB2_SpeciesDiversity_FullPop()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        string path = Path.Combine(BasePath, "phase_b2_species_full_pop.md");
        Directory.CreateDirectory(BasePath);
        File.WriteAllText(path, "");

        void Log(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        // Focus on topologies where species mattered most in Phase B
        var topos = new Topo[]
        {
            new("dp_6_6_6",   new[]{6, 6, 6},     "deep"),     // 1sp=1/5 in Phase A
            new("dp_4_4_4",   new[]{4, 4, 4},      "deep"),     // already 5/5 at 1sp
            new("dp_8_8_8",   new[]{8, 8, 8},      "deep"),     // 2/5 at 1sp
            new("vd_6_6_6_6", new[]{6, 6, 6, 6},   "vdeep"),    // 2/5 at 1sp
            new("md_8_8",     new[]{8, 8},          "medium"),   // 4/5 at 1sp
        };

        int[] speciesCounts = { 1, 3, 6, 12 };
        int[] seeds = Enumerable.Range(1, 10).ToArray(); // 10 seeds for statistical power

        Log($"# Phase B2: Species Diversity — Full GPU Pop Per Species\n");
        Log($"- **Device**: {gpu.DeviceName}");
        Log($"- **Pop per species**: {optPop:N0} (EACH species gets full GPU-optimal pop)");
        Log($"- **Total pop**: species × {optPop:N0} (e.g., 12sp = {12 * optPop:N0} total)");
        Log($"- **Config**: Elman ctx=2, jit=0.15, Gruau ON, moderate edges");
        Log($"- **Budget**: 120s per run");
        Log($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        Log($"| {"Topology",-14} | {"Species",7} | {"TotalPop",8} | {"Seed",5} | {"Shape",-16} | {"InitEdg",7} | {"Solved",6} | {"Time(s)",7} | {"Best",10} | {"Gens",5} | {"Gen/s",5} | {"FinalEdg",8} |");
        Log($"|:---------------|--------:|---------:|------:|:-----------------|--------:|-------:|--------:|-----------:|------:|------:|---------:|");

        var summaries = new List<(Topo topo, int species, TopoSummary summary)>();

        foreach (var topo in topos)
        {
            foreach (int sp in speciesCounts)
            {
                var runs = new List<RunResult>();

                foreach (int seed in seeds)
                {
                    var r = RunSparseExperiment(gpu, topo.Hidden, optPop, seed,
                        species: sp, jitter: 0.15f, gruau: true,
                        edgeAdd: 0.05f, edgeDelete: 0.02f, edgeSplit: 0.01f,
                        edgeRedirect: 0.05f, edgeSwap: 0.02f,
                        budget: 120);
                    runs.Add(r);

                    string solvedStr = r.SolvedGen >= 0 ? $"{r.SolvedGen}" : "DNF";
                    string timeStr = r.SolvedGen >= 0 ? $"{r.SolveTime:F1}" : "-";
                    Log($"| {topo.Name,-14} | {sp,7} | {sp * optPop,8} | {seed,5} | {r.Shape,-16} | {r.InitEdges,7} | {solvedStr,6} | {timeStr,7} | {r.Best,10:F1} | {r.Gens,5} | {r.GenPerSec,5:F1} | {r.FinalEdges,8} |");
                }

                var summary = ComputeSummary(topo, runs);
                summaries.Add((topo, sp, summary));
            }
        }

        // Summary
        Log("\n## Species Effect Summary (Full Pop)\n");
        Log($"| {"Topology",-14} | {"1sp",12} | {"3sp",12} | {"6sp",12} | {"12sp",12} |");
        Log($"|:---------------|-------------:|-------------:|-------------:|-------------:|");

        foreach (var topo in topos)
        {
            var row = new StringBuilder();
            row.Append($"| {topo.Name,-14} |");

            foreach (int sp in speciesCounts)
            {
                var s = summaries.First(x => x.topo.Name == topo.Name && x.species == sp).summary;
                string med = s.MedianSolveTime.HasValue ? $"{s.MedianSolveTime:F0}s" : "-";
                string cell = $"{s.SolveCount}/{s.TotalRuns} {med}";
                row.Append($" {cell,12} |");
            }
            Log(row.ToString());
        }

    }

    #endregion

    #region Phase C: Edge Mutation Rates

    /// <summary>
    /// Phase C: Vary edge mutation aggressiveness on selected topologies.
    /// Tests what rate of complexification works.
    /// </summary>
    [Fact]
    public void PhaseC_EdgeMutationRates()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        string path = Path.Combine(BasePath, "phase_c_edge_mutation_rates.md");
        Directory.CreateDirectory(BasePath);
        File.WriteAllText(path, "");

        void Log(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        var topos = new Topo[]
        {
            new("sh_12",      new[]{12},          "shallow"),
            new("dp_6_6_6",   new[]{6, 6, 6},     "deep"),
            new("dp_8_6_4",   new[]{8, 6, 4},     "deep"),
            new("md_8_8",     new[]{8, 8},        "medium"),
            new("vd_6_6_6_6", new[]{6, 6, 6, 6},  "vdeep"),
        };

        // Edge mutation profiles: (name, add, delete, split, redirect, swap)
        var profiles = new (string name, float add, float del, float split, float redir, float swap)[]
        {
            ("none",       0f,    0f,    0f,    0f,    0f),      // no topology changes
            ("gentle",     0.02f, 0.01f, 0.005f, 0.02f, 0.01f), // slow growth
            ("moderate",   0.05f, 0.02f, 0.01f,  0.05f, 0.02f), // balanced (default)
            ("aggressive", 0.15f, 0.05f, 0.03f,  0.10f, 0.05f), // fast growth
            ("add_heavy",  0.20f, 0.01f, 0.01f,  0.02f, 0.01f), // mostly adding
            ("prune_heavy",0.05f, 0.10f, 0.01f,  0.05f, 0.02f), // add + heavy pruning
        };

        int[] seeds = { 42, 123, 456, 789, 1337 };

        Log($"# Phase C: Edge Mutation Rate Study\n");
        Log($"- **Device**: {gpu.DeviceName}");
        Log($"- **Pop**: {optPop:N0} (6 species)");
        Log($"- **Config**: Elman ctx=2, jit=0.15, Gruau ON");
        Log($"- **Budget**: 120s per run");
        Log($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        Log($"| {"Topology",-14} | {"Profile",-12} | {"Seed",5} | {"InitEdg",7} | {"Solved",6} | {"Time(s)",7} | {"Best",10} | {"Gens",5} | {"FinalEdg",8} | {"EdgeDelta",9} |");
        Log($"|:---------------|:-------------|------:|--------:|-------:|--------:|-----------:|------:|---------:|----------:|");

        var allResults = new List<(Topo topo, string profile, TopoSummary summary)>();

        foreach (var topo in topos)
        {
            foreach (var (pname, add, del, split, redir, swap) in profiles)
            {
                var runs = new List<RunResult>();

                foreach (int seed in seeds)
                {
                    var r = RunSparseExperiment(gpu, topo.Hidden, optPop, seed,
                        species: 6, jitter: 0.15f, gruau: true,
                        edgeAdd: add, edgeDelete: del, edgeSplit: split,
                        edgeRedirect: redir, edgeSwap: swap,
                        budget: 120);
                    runs.Add(r);

                    string solvedStr = r.SolvedGen >= 0 ? $"{r.SolvedGen}" : "DNF";
                    string timeStr = r.SolvedGen >= 0 ? $"{r.SolveTime:F1}" : "-";
                    int delta = r.FinalEdges - r.InitEdges;
                    string deltaStr = delta >= 0 ? $"+{delta}" : $"{delta}";
                    Log($"| {topo.Name,-14} | {pname,-12} | {seed,5} | {r.InitEdges,7} | {solvedStr,6} | {timeStr,7} | {r.Best,10:F1} | {r.Gens,5} | {r.FinalEdges,8} | {deltaStr,9} |");
                }

                var summary = ComputeSummary(topo, runs);
                allResults.Add((topo, pname, summary));
            }
        }

        // Summary matrix
        Log("\n## Edge Rate Effect Summary\n");
        var profileNames = profiles.Select(p => p.name).ToArray();
        var header = $"| {"Topology",-14} |" + string.Join("", profileNames.Select(p => $" {p,12} |"));
        Log(header);
        Log($"|:---------------|" + string.Join("", profileNames.Select(_ => "-------------:|")));

        foreach (var topo in topos)
        {
            var row = new StringBuilder();
            row.Append($"| {topo.Name,-14} |");
            foreach (var pname in profileNames)
            {
                var s = allResults.First(x => x.topo.Name == topo.Name && x.profile == pname).summary;
                string cell = s.SolveCount > 0 ? $"{s.SolveCount}/{s.TotalRuns} {s.MedianSolveTime:F0}s" : $"0/{s.TotalRuns}";
                row.Append($" {cell,12} |");
            }
            Log(row.ToString());
        }
    }

    #endregion

    #region Phase D: Deep Dive Validation

    /// <summary>
    /// Phase D: Run the best configs from A-C with 20 seeds each for statistical power.
    /// Also includes known failing configs for contrast.
    /// </summary>
    [Fact]
    public void PhaseD_DeepDive()
    {
        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        Warmup(gpu);

        string path = Path.Combine(BasePath, "phase_d_deep_dive.md");
        Directory.CreateDirectory(BasePath);
        File.WriteAllText(path, "");

        void Log(string line) { _output.WriteLine(line); File.AppendAllText(path, line + "\n"); }

        // Configs selected from Phase A-C results
        var configs = new (string name, int[] hidden, int species, string profile)[]
        {
            // WINNING SPREAD: data-driven best configs
            ("dp666_6sp_add",  new[]{6,6,6},    6,  "add_heavy"),  // PhaseC winner for dp666
            ("dp666_3sp_gen",  new[]{6,6,6},    3,  "gentle"),     // PhaseB: 3sp=5/5, PhaseC: gentle=5/5
            ("dp666_6sp_gen",  new[]{6,6,6},    6,  "gentle"),     // PhaseB+C combined best
            ("dp444_6sp_mod",  new[]{4,4,4},    6,  "moderate"),   // PhaseA: 5/5 at all sp counts
            ("dp464_1sp_mod",  new[]{4,6,4},    1,  "moderate"),   // PhaseA: diamond 5/5 1sp
            ("md44_1sp_mod",   new[]{4,4},      1,  "moderate"),   // PhaseA: md_4_4 RELIABLE, fastest medium

            // BORDERLINE: topologies that need multi-species
            ("dp666_1sp_mod",  new[]{6,6,6},    1,  "moderate"),   // PhaseA: only 1/5, compare with above
            ("dp888_6sp_add",  new[]{8,8,8},    6,  "add_heavy"),  // PhaseB: 4/5 at 6sp, PhaseC: add_heavy best
            ("vd6666_3sp_add", new[]{6,6,6,6},  3,  "add_heavy"),  // PhaseB: 3sp best, PhaseC: add_heavy best

            // FAILING SPREAD: for contrast
            ("dp888_1sp_mod",  new[]{8,8,8},    1,  "moderate"),   // PhaseA: 2/5 (18% density)
            ("dp666_6sp_prn",  new[]{6,6,6},    6,  "prune_heavy"),// PhaseC: pruning kills sparse
            ("vd6666_1sp_non", new[]{6,6,6,6},  1,  "none"),       // PhaseA: 2/5, no edge help
        };

        // Profile lookup
        var profileMap = new Dictionary<string, (float add, float del, float split, float redir, float swap)>
        {
            ["none"]        = (0f,    0f,    0f,    0f,    0f),
            ["gentle"]      = (0.02f, 0.01f, 0.005f, 0.02f, 0.01f),
            ["moderate"]    = (0.05f, 0.02f, 0.01f,  0.05f, 0.02f),
            ["aggressive"]  = (0.15f, 0.05f, 0.03f,  0.10f, 0.05f),
            ["add_heavy"]   = (0.20f, 0.01f, 0.01f,  0.02f, 0.01f),
            ["prune_heavy"] = (0.05f, 0.10f, 0.01f,  0.05f, 0.02f),
        };

        int[] seeds = Enumerable.Range(1, 20).ToArray(); // seeds 1-20

        Log($"# Phase D: Deep Dive Validation (20 seeds)\n");
        Log($"- **Device**: {gpu.DeviceName}");
        Log($"- **Pop**: {optPop:N0}");
        Log($"- **Config**: Elman ctx=2, jit=0.15, Gruau ON");
        Log($"- **Budget**: 180s per run");
        Log($"- **Date**: {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        Log($"| {"Config",-16} | {"Seed",5} | {"Shape",-14} | {"Sp",3} | {"InitEdg",7} | {"Solved",6} | {"Time(s)",7} | {"Best",10} | {"Gens",5} | {"FinalEdg",8} |");
        Log($"|:-----------------|------:|:---------------|----:|--------:|-------:|--------:|-----------:|------:|---------:|");

        var configSummaries = new List<(string name, int species, string profile, TopoSummary summary)>();

        foreach (var (name, hidden, species, profile) in configs)
        {
            var (add, del, split, redir, swap) = profileMap[profile];
            var runs = new List<RunResult>();
            var topo = new Topo(name, hidden, "");

            foreach (int seed in seeds)
            {
                var r = RunSparseExperiment(gpu, hidden, optPop, seed,
                    species: species, jitter: 0.15f, gruau: true,
                    edgeAdd: add, edgeDelete: del, edgeSplit: split,
                    edgeRedirect: redir, edgeSwap: swap,
                    budget: 180);
                runs.Add(r);

                string solvedStr = r.SolvedGen >= 0 ? $"{r.SolvedGen}" : "DNF";
                string timeStr = r.SolvedGen >= 0 ? $"{r.SolveTime:F1}" : "-";
                Log($"| {name,-16} | {seed,5} | {r.Shape,-14} | {species,3} | {r.InitEdges,7} | {solvedStr,6} | {timeStr,7} | {r.Best,10:F1} | {r.Gens,5} | {r.FinalEdges,8} |");
            }

            var summary = ComputeSummary(topo, runs);
            configSummaries.Add((name, species, profile, summary));
        }

        // Final summary
        Log("\n## Validation Summary\n");
        Log($"| {"Config",-16} | {"Sp",3} | {"Profile",-10} | {"Rate",8} | {"Med(s)",6} | {"Min(s)",6} | {"Max(s)",6} | {"MedBest",8} | {"InitEdg",7} | {"MedFinal",8} | {"Verdict",-10} |");
        Log($"|:-----------------|----:|:-----------|---------:|-------:|-------:|-------:|---------:|--------:|---------:|:-----------|");

        foreach (var (name, species, profile, s) in configSummaries)
        {
            string rate = $"{s.SolveCount}/{s.TotalRuns}";
            string med = s.MedianSolveTime.HasValue ? $"{s.MedianSolveTime:F1}" : "-";
            string min = s.MinSolveTime.HasValue ? $"{s.MinSolveTime:F1}" : "-";
            string max = s.MaxSolveTime.HasValue ? $"{s.MaxSolveTime:F1}" : "-";
            string verdict = s.SolveCount >= 18 ? "RELIABLE" :   // 90%+
                             s.SolveCount >= 10 ? "GOOD" :        // 50%+
                             s.SolveCount > 0 ? "WEAK" : "FAILS";
            Log($"| {name,-16} | {species,3} | {profile,-10} | {rate,8} | {med,6} | {min,6} | {max,6} | {s.MedianBest,8:F0} | {s.InitEdges,7} | {s.MedianFinalEdges,8} | {verdict,-10} |");
        }
    }

    #endregion

    #region Experiment Runner

    private record RunResult(
        string Shape, int InitEdges, int PossibleEdges, float InitDensity,
        int SolvedGen, double SolveTime, float Best, int Gens, double GenPerSec,
        int FinalEdges, double Wall);

    private record TopoSummary(
        string Name, string Category, int TotalRuns, int SolveCount,
        double? MedianSolveTime, double? MinSolveTime, double? MaxSolveTime,
        float MedianBest, int InitEdges, int PossibleEdges, int MedianFinalEdges);

    private RunResult RunSparseExperiment(
        GPUDoublePoleEvaluator gpu, int[] hidden, int optPop, int seed,
        int species, float jitter, bool gruau,
        float edgeAdd, float edgeDelete, float edgeSplit,
        float edgeRedirect, float edgeSwap,
        int budget)
    {
        int ctx = 2;
        int inputSize = 3 + ctx;
        int outputSize = 1 + ctx;
        int popPerSpecies = optPop; // Each species gets full GPU-optimal pop (free throughput)

        gpu.IncludeVelocity = false;
        gpu.ContextSize = ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = 10_000;
        gpu.UseGruauFitness = gruau;

        var topology = BuildSparseTopology(ctx, hidden, seed);
        string shape = string.Join("->", topology.RowCounts);
        int initEdges = topology.Edges.Count;
        int possibleEdges = ComputePossibleEdges(topology);
        float initDensity = possibleEdges > 0 ? (float)initEdges / possibleEdges : 0f;

        var config = new EvolutionConfig
        {
            SpeciesCount = species,
            IndividualsPerSpecies = popPerSpecies,
            MinSpeciesCount = species,
            Elites = 10,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
        };

        config.MutationRates.WeightJitterStdDev = jitter;
        config.EdgeMutations.EdgeAdd = edgeAdd;
        config.EdgeMutations.EdgeDeleteRandom = edgeDelete;
        config.EdgeMutations.EdgeSplit = edgeSplit;
        config.EdgeMutations.EdgeRedirect = edgeRedirect;
        config.EdgeMutations.EdgeSwap = edgeSwap;

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int solvedGen = -1;
        double solveTime = 0;

        while (sw.Elapsed.TotalSeconds < budget)
        {
            int totalSolved = 0;
            float globalBest = float.MinValue;

            foreach (var sp in population.AllSpecies)
            {
                var (fitness, solved) = gpu.EvaluatePopulation(
                    sp.Topology, sp.Individuals, seed: gen);

                for (int i = 0; i < sp.Individuals.Count; i++)
                {
                    var ind = sp.Individuals[i];
                    ind.Fitness = fitness[i];
                    sp.Individuals[i] = ind;
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

        // Count final edges across all species (max)
        int finalEdges = population.AllSpecies.Max(s => s.Topology.Edges.Count);

        return new RunResult(shape, initEdges, possibleEdges, initDensity,
            solvedGen, solveTime, bestEver, gen, genPerSec, finalEdges, wall);
    }

    private static SpeciesSpec BuildSparseTopology(int ctx, int[] hidden, int seed)
    {
        int inputSize = 3 + ctx;
        int outputSize = 1 + ctx;

        var builder = new SpeciesBuilder().AddInputRow(inputSize);

        foreach (int h in hidden)
            builder.AddHiddenRow(h,
                ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.ReLU,
                ActivationType.LeakyReLU, ActivationType.Softsign, ActivationType.Sin);

        builder.AddOutputRow(outputSize, ActivationType.Tanh)
               .WithMaxInDegree(10);

        builder.InitializeSparse(new Random(seed));
        return builder.Build();
    }

    private static int ComputePossibleEdges(SpeciesSpec spec)
    {
        int total = 0;
        for (int destRow = 1; destRow < spec.RowCounts.Length; destRow++)
        {
            int destCount = spec.RowCounts[destRow];
            int srcCount = 0;
            for (int srcRow = 0; srcRow < destRow; srcRow++)
                srcCount += spec.RowCounts[srcRow];
            // Each dest node can have up to MaxInDegree edges, but also limited by available sources
            total += destCount * Math.Min(srcCount, spec.MaxInDegree);
        }
        return total;
    }

    private static TopoSummary ComputeSummary(Topo topo, List<RunResult> runs)
    {
        var solveTimes = runs.Where(r => r.SolvedGen >= 0).Select(r => r.SolveTime).OrderBy(t => t).ToList();
        var allBest = runs.Select(r => r.Best).OrderBy(b => b).ToList();
        var allFinalEdges = runs.Select(r => r.FinalEdges).OrderBy(e => e).ToList();

        return new TopoSummary(
            topo.Name, topo.Category,
            TotalRuns: runs.Count,
            SolveCount: solveTimes.Count,
            MedianSolveTime: solveTimes.Count > 0 ? solveTimes[solveTimes.Count / 2] : null,
            MinSolveTime: solveTimes.Count > 0 ? solveTimes[0] : null,
            MaxSolveTime: solveTimes.Count > 0 ? solveTimes[^1] : null,
            MedianBest: allBest[allBest.Count / 2],
            InitEdges: runs[0].InitEdges,
            PossibleEdges: runs[0].PossibleEdges,
            MedianFinalEdges: allFinalEdges[allFinalEdges.Count / 2]);
    }

    private static double Median(List<double> sorted) =>
        sorted.Count > 0 ? sorted[sorted.Count / 2] : 0;

    private static int CategoryOrder(string cat) => cat switch
    {
        "shallow" => 0, "medium" => 1, "deep" => 2, "vdeep" => 3, "asym" => 4, _ => 5
    };

    private void Warmup(GPUDoublePoleEvaluator gpu)
    {
        var topo = BuildSparseTopology(2, new[] { 6 }, 0);
        var ev = new Evolver(seed: 0);
        var pop = ev.InitializePopulation(
            new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 10 }, topo);
        gpu.MaxSteps = 10;
        gpu.IncludeVelocity = true;
        gpu.ContextSize = 0;
        gpu.UseGruauFitness = false;
        gpu.EvaluatePopulation(pop.AllSpecies[0].Topology, pop.AllSpecies[0].Individuals, seed: 0);
    }

    #endregion
}
