using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Multi-position DPNV benchmark: evolve controllers that solve ALL 16 starting positions.
/// This is the Evolvion side of the Evolvatron-Verify comparison (PPO baseline).
/// </summary>
public class MultiPositionBenchmark
{
    private readonly ITestOutputHelper _output;
    private const string ResultsPath = @"C:\Dev\Evolvatron\scratch\multi_position";

    public MultiPositionBenchmark(ITestOutputHelper output) => _output = output;

    #region Starting Positions

    private static float[][] LoadStartingPositions()
    {
        var path = Path.Combine(
            Path.GetDirectoryName(typeof(MultiPositionBenchmark).Assembly.Location)!,
            "..", "..", "..", "..", "Evolvatron.Evolvion", "starting_positions.json");

        // Fallback to absolute path
        if (!File.Exists(path))
            path = @"C:\Dev\Evolvatron\Evolvatron.Evolvion\starting_positions.json";

        var json = File.ReadAllText(path);
        using var doc = JsonDocument.Parse(json);
        var positions = doc.RootElement.GetProperty("positions");

        var result = new float[positions.GetArrayLength()][];
        for (int i = 0; i < result.Length; i++)
        {
            var state = positions[i].GetProperty("state");
            result[i] = new float[6];
            for (int j = 0; j < 6; j++)
                result[i][j] = (float)state[j].GetDouble();
        }
        return result;
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
    /// Quick sanity check: verify multi-position evaluation runs and returns sensible results.
    /// Uses tiny population and short MaxSteps.
    /// </summary>
    [Fact]
    public void MultiPositionEval_SmokeTest()
    {
        var positions = LoadStartingPositions();
        Assert.Equal(16, positions.Length);

        using var gpu = new GPUDoublePoleEvaluator(maxIndividuals: 200);
        gpu.MaxSteps = 100;
        gpu.ContextSize = 2;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;
        gpu.SetStartingPositions(positions);

        var topology = BuildDenseTopology(2, new[] { 6 });
        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 100 };
        var population = evolver.InitializePopulation(config, topology);

        var (fitness, allSolved) = gpu.EvaluateAllPositions(
            population.AllSpecies[0].Topology,
            population.AllSpecies[0].Individuals,
            seed: 0);

        _output.WriteLine($"Pop size: {fitness.Length}");
        _output.WriteLine($"Mean fitness: {fitness.Average():F1}");
        _output.WriteLine($"Max fitness: {fitness.Max():F1}");
        _output.WriteLine($"All-16 solved: {allSolved}");

        Assert.Equal(100, fitness.Length);
        Assert.True(fitness.All(f => f > 0), "All fitnesses should be positive");
    }

    /// <summary>
    /// Main benchmark: evolve on all 16 positions with the optimal DPNV config.
    /// Budget: 120s, 10 seeds. Compares directly with PPO results.
    /// </summary>
    [Fact]
    public void MultiPosition_16Pos_120s_10Seeds()
    {
        var positions = LoadStartingPositions();
        int budget = 120;
        int numSeeds = 10;
        int[] hidden = { 6 };
        int ctx = 2;

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Multi-Position DPNV Benchmark ===");
        _output.WriteLine($"Device: {gpu.DeviceName}");
        _output.WriteLine($"Population: {optPop} per species");
        _output.WriteLine($"Topology: 5->6->3 dense, Elman ctx=2");
        _output.WriteLine($"Positions: {positions.Length}, Budget: {budget}s, Seeds: {numSeeds}");
        _output.WriteLine("");

        // Warmup
        Warmup(gpu, positions);

        var results = new List<RunResult>();
        var sb = new StringBuilder();

        for (int seed = 0; seed < numSeeds; seed++)
        {
            var r = RunMultiPositionExperiment(gpu, hidden, optPop, seed, positions, budget);
            results.Add(r);

            string status = r.SolvedGen >= 0
                ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                : $"FAILED best={r.BestFitness:F0}";
            _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
            sb.AppendLine($"Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
        }

        int solved = results.Count(r => r.SolvedGen >= 0);
        var solveTimes = results.Where(r => r.SolvedGen >= 0).Select(r => r.SolveTime).ToList();

        _output.WriteLine("");
        _output.WriteLine($"=== RESULTS ===");
        _output.WriteLine($"Solved: {solved}/{numSeeds}");
        if (solveTimes.Count > 0)
        {
            _output.WriteLine($"Median solve time: {Median(solveTimes):F1}s");
            _output.WriteLine($"Min/Max: {solveTimes.Min():F1}s / {solveTimes.Max():F1}s");
        }

        // Save results
        Directory.CreateDirectory(ResultsPath);
        File.WriteAllText(Path.Combine(ResultsPath, "multi_pos_120s.md"), sb.ToString());
    }

    /// <summary>
    /// Quick 30s test with 3 seeds — for development iteration.
    /// </summary>
    [Fact]
    public void MultiPosition_16Pos_30s_Quick()
    {
        var positions = LoadStartingPositions();
        int budget = 30;

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Quick Multi-Position Test (30s) ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop}");

        Warmup(gpu, positions);

        for (int seed = 0; seed < 3; seed++)
        {
            var r = RunMultiPositionExperiment(gpu, new[] { 6 }, optPop, seed, positions, budget);
            string status = r.SolvedGen >= 0
                ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                : $"FAILED best={r.BestFitness:F0}";
            _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s) | bestSolved={r.BestSolvedCount}/16");
        }
    }

    /// <summary>
    /// Quick test with 2K MaxSteps — much faster generations for more evolutionary pressure.
    /// </summary>
    [Fact]
    public void MultiPosition_16Pos_30s_2KSteps()
    {
        var positions = LoadStartingPositions();

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Quick Multi-Position Test (30s, 2K steps) ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop}");

        Warmup(gpu, positions);

        for (int seed = 0; seed < 3; seed++)
        {
            var r = RunMultiPositionExperiment(gpu, new[] { 6 }, optPop, seed, positions, 30, maxSteps: 2_000);
            string status = r.SolvedGen >= 0
                ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                : $"FAILED best={r.BestFitness:F0}";
            _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s) | bestSolved={r.BestSolvedCount}/16");
        }
    }

    /// <summary>
    /// Topology sweep: test different network widths to find what cracks all 16 positions.
    /// </summary>
    [Fact]
    public void MultiPosition_TopologySweep_60s()
    {
        var positions = LoadStartingPositions();
        int budget = 60;

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Topology Sweep (60s per config) ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop}");

        Warmup(gpu, positions);

        var configs = new[]
        {
            (name: "5->6->3", hidden: new[] { 6 }, species: 1),
            (name: "5->8->3", hidden: new[] { 8 }, species: 1),
            (name: "5->12->3", hidden: new[] { 12 }, species: 1),
            (name: "5->6->6->3", hidden: new[] { 6, 6 }, species: 1),
            (name: "5->6->3 x3sp", hidden: new[] { 6 }, species: 3),
        };

        foreach (var (name, hidden, species) in configs)
        {
            _output.WriteLine($"\n--- {name} ---");
            for (int seed = 0; seed < 3; seed++)
            {
                var r = RunMultiPositionExperiment(gpu, hidden, optPop, seed, positions, budget,
                    speciesCount: species);
                int solvedCount = (int)(r.BestFitness / 10_000);
                string status = r.SolvedGen >= 0
                    ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                    : $"~{solvedCount}/16 best={r.BestFitness:F0}";
                _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
            }
        }
    }

    /// <summary>
    /// Species sweep: find optimal species count for multi-position DPNV.
    /// Tests 1, 2, 3, 4, 6 species × 5 seeds × 120s budget.
    /// Each species gets FULL GPU-optimal population (free throughput).
    /// </summary>
    [Fact]
    public void MultiPosition_SpeciesSweep_120s()
    {
        var positions = LoadStartingPositions();
        int budget = 120;
        int numSeeds = 5;

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Species Sweep for Multi-Position DPNV ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop} per species");
        _output.WriteLine($"Topology: 5->6->3 dense, Elman ctx=2, Budget: {budget}s, Seeds: {numSeeds}");
        _output.WriteLine($"");

        Warmup(gpu, positions);

        int[] speciesCounts = { 1, 2, 3, 4, 6 };

        foreach (int sp in speciesCounts)
        {
            _output.WriteLine($"--- {sp} species ({sp * optPop} total individuals) ---");
            var results = new List<RunResult>();

            for (int seed = 0; seed < numSeeds; seed++)
            {
                var r = RunMultiPositionExperiment(gpu, new[] { 6 }, optPop, seed, positions, budget,
                    speciesCount: sp);
                results.Add(r);

                string status = r.SolvedGen >= 0
                    ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                    : $"FAILED best={r.BestFitness:F0} bestSolved={r.BestSolvedCount}/16";
                _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
            }

            int solved = results.Count(r => r.SolvedGen >= 0);
            double avgGens = results.Average(r => r.TotalGens);
            double avgGenPerSec = results.Average(r => r.GenPerSec);
            int avgBestSolved = (int)results.Average(r => r.BestSolvedCount);
            var solveTimes = results.Where(r => r.SolvedGen >= 0).Select(r => r.SolveTime).ToList();
            string timeStr = solveTimes.Count > 0 ? $"median={Median(solveTimes):F1}s" : "N/A";

            _output.WriteLine($"  >> {sp}sp: {solved}/{numSeeds} solved, {timeStr}, " +
                $"avg {avgGens:F0} gens ({avgGenPerSec:F1} gen/s), avg best={avgBestSolved}/16");
            _output.WriteLine("");
        }
    }

    /// <summary>
    /// Compare single-position (legacy) vs multi-position evaluation speed.
    /// How much slower is 16-position eval per generation?
    /// </summary>
    [Fact]
    public void MultiPosition_SpeedComparison()
    {
        var positions = LoadStartingPositions();

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;
        gpu.MaxSteps = 10_000;
        gpu.ContextSize = 2;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;

        var topology = BuildDenseTopology(2, new[] { 6 });
        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = optPop };
        var population = evolver.InitializePopulation(config, topology);

        var sp = population.AllSpecies[0];

        // Warmup
        gpu.EvaluatePopulation(sp.Topology, sp.Individuals, seed: 0);

        // Single position timing
        var sw = Stopwatch.StartNew();
        int singleReps = 5;
        for (int i = 0; i < singleReps; i++)
            gpu.EvaluatePopulation(sp.Topology, sp.Individuals, seed: i);
        double singleMs = sw.Elapsed.TotalMilliseconds / singleReps;

        // Multi position timing
        gpu.SetStartingPositions(positions);
        sw.Restart();
        int multiReps = 3;
        for (int i = 0; i < multiReps; i++)
            gpu.EvaluateAllPositions(sp.Topology, sp.Individuals, seed: i);
        double multiMs = sw.Elapsed.TotalMilliseconds / multiReps;

        _output.WriteLine($"Single position eval: {singleMs:F1}ms");
        _output.WriteLine($"16-position eval: {multiMs:F1}ms");
        _output.WriteLine($"Slowdown: {multiMs / singleMs:F1}x (ideal: 16x)");
        _output.WriteLine($"Overhead per position: {multiMs / 16:F1}ms");
    }

    private RunResult RunMultiPositionExperiment(
        GPUDoublePoleEvaluator gpu, int[] hidden, int optPop, int seed,
        float[][] positions, int budget, int maxSteps = 10_000, int speciesCount = 3)
    {
        int ctx = 2;
        gpu.IncludeVelocity = false;
        gpu.ContextSize = ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = maxSteps;
        gpu.UseGruauFitness = false;
        gpu.SetStartingPositions(positions);

        var topology = BuildDenseTopology(ctx, hidden);

        var config = new EvolutionConfig
        {
            SpeciesCount = speciesCount,
            IndividualsPerSpecies = optPop,
            MinSpeciesCount = speciesCount,
            Elites = 10,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
        };
        config.MutationRates.WeightJitterStdDev = 0.15f;

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int bestSolvedCount = 0;
        int solvedGen = -1;
        double solveTime = 0;

        while (sw.Elapsed.TotalSeconds < budget)
        {
            float globalBest = float.MinValue;
            int totalAllSolved = 0;

            foreach (var sp in population.AllSpecies)
            {
                var (fitness, allSolved) = gpu.EvaluateAllPositions(
                    sp.Topology, sp.Individuals, seed: gen);

                for (int i = 0; i < sp.Individuals.Count; i++)
                {
                    var ind = sp.Individuals[i];
                    ind.Fitness = fitness[i];
                    sp.Individuals[i] = ind;
                    if (fitness[i] > globalBest) globalBest = fitness[i];
                }
                totalAllSolved += allSolved;
            }

            if (globalBest > bestEver) bestEver = globalBest;
            if (totalAllSolved > bestSolvedCount) bestSolvedCount = totalAllSolved;

            if (totalAllSolved > 0 && solvedGen < 0)
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

        return new RunResult(solvedGen, solveTime, bestEver, gen, genPerSec, wall, bestSolvedCount);
    }

    /// <summary>
    /// Diagnostic: evolve 120s, then evaluate the best individual on each position separately.
    /// Shows exactly which positions fail and how many steps they survive.
    /// </summary>
    [Fact]
    public void MultiPosition_Diagnostic_PerPositionBreakdown()
    {
        var positions = LoadStartingPositions();

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Per-Position Diagnostic ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop}");

        Warmup(gpu, positions);

        // Evolve for 120s with 1 species (max generations)
        var r = RunMultiPositionExperiment(gpu, new[] { 6 }, optPop, seed: 42, positions, budget: 120,
            speciesCount: 1);

        _output.WriteLine($"Evolution: {r.TotalGens} gens, bestFitness={r.BestFitness:F0}, bestSolved={r.BestSolvedCount}/16");
        _output.WriteLine("");

        // Now evaluate each position individually to see per-position breakdown
        // Re-run the evolution to get the final population (can't extract from RunResult)
        // Instead, run a fresh short evolution and evaluate best per position
        gpu.MaxSteps = 10_000;
        gpu.ContextSize = 2;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;
        gpu.IncludeVelocity = false;

        var topology = BuildDenseTopology(2, new[] { 6 });
        var config = new EvolutionConfig
        {
            SpeciesCount = 1,
            IndividualsPerSpecies = optPop,
            MinSpeciesCount = 1,
            Elites = 10,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
        };
        config.MutationRates.WeightJitterStdDev = 0.15f;

        var evolver = new Evolver(seed: 42);
        var population = evolver.InitializePopulation(config, topology);

        // Evolve with multi-position for 120s
        gpu.SetStartingPositions(positions);
        var sw = Stopwatch.StartNew();
        int gen = 0;
        while (sw.Elapsed.TotalSeconds < 120)
        {
            var sp = population.AllSpecies[0];
            var (fitness, allSolved) = gpu.EvaluateAllPositions(sp.Topology, sp.Individuals, seed: gen);
            for (int i = 0; i < sp.Individuals.Count; i++)
            {
                var ind = sp.Individuals[i];
                ind.Fitness = fitness[i];
                sp.Individuals[i] = ind;
            }
            if (allSolved > 0) { _output.WriteLine($"SOLVED ALL at gen {gen}!"); break; }
            evolver.StepGeneration(population);
            gen++;
        }

        _output.WriteLine($"Evolved {gen} gens in {sw.Elapsed.TotalSeconds:F1}s");

        // Find best individual
        var species = population.AllSpecies[0];
        int bestIdx = 0;
        for (int i = 1; i < species.Individuals.Count; i++)
            if (species.Individuals[i].Fitness > species.Individuals[bestIdx].Fitness)
                bestIdx = i;

        _output.WriteLine($"Best individual fitness: {species.Individuals[bestIdx].Fitness:F0}");
        _output.WriteLine("");

        // Evaluate best individual on each position separately
        // Create a mini-population with just the best individual replicated
        var bestInd = species.Individuals[bestIdx];
        var singlePop = new List<Individual> { bestInd };

        using var singleGpu = new GPUDoublePoleEvaluator(maxIndividuals: 10);
        singleGpu.MaxSteps = 10_000;
        singleGpu.ContextSize = 2;
        singleGpu.IsJordan = false;
        singleGpu.UseGruauFitness = false;
        singleGpu.IncludeVelocity = false;

        _output.WriteLine($"{"Pos",4} {"Tier",-10} {"Description",-30} {"Steps",8} {"Status",-8}");
        _output.WriteLine(new string('-', 70));

        int totalSolved = 0;
        for (int posIdx = 0; posIdx < positions.Length; posIdx++)
        {
            // Evaluate on single position
            singleGpu.SetStartingPositions(new[] { positions[posIdx] });
            var (fit, solved) = singleGpu.EvaluateAllPositions(species.Topology, singlePop, seed: 0);

            int steps = (int)fit[0]; // single position: fitness = raw steps
            string status = solved > 0 ? "SOLVED" : "FAILED";
            if (solved > 0) totalSolved++;

            // Get tier from position data
            string tier = posIdx < 4 ? "gentle" : posIdx < 8 ? "moderate" : posIdx < 12 ? "hard" : "extreme";
            string desc = posIdx switch
            {
                0 => "Standard (4 deg)",
                1 => "Mirror of standard",
                2 => "Both poles off (3 deg)",
                3 => "Cart offset right",
                4 => "Wide pole1 (10 deg)",
                5 => "Moving cart",
                6 => "Pole2 falling (8 deg)",
                7 => "Pole1 spinning",
                8 => "Cart right, poles diverge",
                9 => "Fast cart, opposing poles",
                10 => "Angular motion",
                11 => "Everything perturbed",
                12 => "Cart near edge (12+8 deg)",
                13 => "Big angle + spin (15 deg)",
                14 => "Full chaos",
                15 => "Mixed perturbation",
                _ => "?"
            };

            _output.WriteLine($"{posIdx,4} {tier,-10} {desc,-30} {steps,8} {status,-8}");
        }
        _output.WriteLine("");
        _output.WriteLine($"Total solved: {totalSolved}/16");
    }

    /// <summary>
    /// Hypothesis tests: try configs that might crack the remaining positions.
    /// Each runs for 120s × 3 seeds. Tests wider network, larger context, and longer MaxSteps.
    /// </summary>
    [Fact]
    public void MultiPosition_HypothesisTests_120s()
    {
        var positions = LoadStartingPositions();
        int budget = 120;
        int numSeeds = 3;

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Hypothesis Tests for 16-Position DPNV ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop}");
        _output.WriteLine("");

        Warmup(gpu, positions);

        // Baseline: current best config
        var hypotheses = new (string name, int[] hidden, int ctx, int maxSteps, bool gruau)[]
        {
            // Baseline
            ("baseline 5->6->3 ctx=2", new[] { 6 }, 2, 10_000, false),
            // H1: More memory — does ctx=4 help with velocity estimation on chaotic starts?
            ("ctx=4 5->8->3", new[] { 8 }, 4, 10_000, false),
            // H2: Much more memory — ctx=6
            ("ctx=6 5->12->3", new[] { 12 }, 6, 10_000, false),
            // H3: Wider network, same ctx — is it capacity-limited?
            ("wide 5->16->3 ctx=2", new[] { 16 }, 2, 10_000, false),
            // H4: Deep + wide
            ("deep 5->8->8->3 ctx=2", new[] { 8, 8 }, 2, 10_000, false),
            // H5: Short episodes — faster gens, more evolutionary pressure
            ("short 2K steps ctx=2", new[] { 6 }, 2, 2_000, false),
            // H6: Gruau fitness — does anti-jiggle help multi-pos?
            ("gruau 5->6->3 ctx=2", new[] { 6 }, 2, 10_000, true),
        };

        foreach (var (name, hidden, ctx, maxSteps, gruau) in hypotheses)
        {
            _output.WriteLine($"--- {name} ---");

            for (int seed = 0; seed < numSeeds; seed++)
            {
                gpu.ContextSize = ctx;
                gpu.UseGruauFitness = gruau;

                var r = RunMultiPositionExperimentEx(gpu, hidden, ctx, optPop, seed, positions, budget,
                    maxSteps: maxSteps, gruau: gruau);
                string status = r.SolvedGen >= 0
                    ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                    : $"FAILED best={r.BestFitness:F0} bestSolved={r.BestSolvedCount}/16";
                _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
            }
            _output.WriteLine("");
        }
    }

    private RunResult RunMultiPositionExperimentEx(
        GPUDoublePoleEvaluator gpu, int[] hidden, int ctx, int optPop, int seed,
        float[][] positions, int budget, int maxSteps = 10_000, int speciesCount = 1, bool gruau = false)
    {
        gpu.IncludeVelocity = false;
        gpu.ContextSize = ctx;
        gpu.IsJordan = false;
        gpu.MaxSteps = maxSteps;
        gpu.UseGruauFitness = gruau;
        gpu.SetStartingPositions(positions);

        var topology = BuildDenseTopology(ctx, hidden);

        var config = new EvolutionConfig
        {
            SpeciesCount = speciesCount,
            IndividualsPerSpecies = optPop,
            MinSpeciesCount = speciesCount,
            Elites = 10,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
        };
        config.MutationRates.WeightJitterStdDev = 0.15f;

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int bestSolvedCount = 0;
        int solvedGen = -1;
        double solveTime = 0;

        while (sw.Elapsed.TotalSeconds < budget)
        {
            float globalBest = float.MinValue;
            int totalAllSolved = 0;

            foreach (var sp in population.AllSpecies)
            {
                var (fitness, allSolved) = gpu.EvaluateAllPositions(
                    sp.Topology, sp.Individuals, seed: gen);

                for (int i = 0; i < sp.Individuals.Count; i++)
                {
                    var ind = sp.Individuals[i];
                    ind.Fitness = fitness[i];
                    sp.Individuals[i] = ind;
                    if (fitness[i] > globalBest) globalBest = fitness[i];
                }
                totalAllSolved += allSolved;
            }

            if (globalBest > bestEver) bestEver = globalBest;
            if (totalAllSolved > bestSolvedCount) bestSolvedCount = totalAllSolved;

            if (totalAllSolved > 0 && solvedGen < 0)
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

        return new RunResult(solvedGen, solveTime, bestEver, gen, genPerSec, wall, bestSolvedCount);
    }

    /// <summary>
    /// Follow-up: short episodes + curriculum. Test what actually cracks 16/16.
    /// </summary>
    [Fact]
    public void MultiPosition_ShortEpisodes_And_Curriculum()
    {
        var positions = LoadStartingPositions();

        using var gpu = new GPUDoublePoleEvaluator();
        int optPop = gpu.OptimalPopulationSize;

        _output.WriteLine($"=== Short Episodes + Curriculum ===");
        _output.WriteLine($"Device: {gpu.DeviceName}, Pop: {optPop}");
        _output.WriteLine("");

        Warmup(gpu, positions);

        // Test 1: Very short episodes (500, 1K, 2K steps) at 120s
        foreach (int maxSteps in new[] { 500, 1_000, 2_000, 5_000 })
        {
            _output.WriteLine($"--- MaxSteps={maxSteps} (120s × 3 seeds) ---");
            for (int seed = 0; seed < 3; seed++)
            {
                var r = RunMultiPositionExperimentEx(gpu, new[] { 6 }, 2, optPop, seed, positions, 120,
                    maxSteps: maxSteps);
                int bestSolvedApprox = (int)(r.BestFitness / maxSteps);
                string status = r.SolvedGen >= 0
                    ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                    : $"FAILED best={r.BestFitness:F0} ~{bestSolvedApprox}/16";
                _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s) | bestSolved={r.BestSolvedCount}/16");
            }
            _output.WriteLine("");
        }

        // Test 2: Curriculum — evolve at 500 steps for 30s, then 2K for 30s, then 10K for 60s
        _output.WriteLine($"--- Curriculum: 500→2K→10K (30s+30s+60s × 3 seeds) ---");
        for (int seed = 0; seed < 3; seed++)
        {
            var r = RunCurriculumExperiment(gpu, optPop, seed, positions);
            string status = r.SolvedGen >= 0
                ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                : $"FAILED best={r.BestFitness:F0} bestSolved={r.BestSolvedCount}/16";
            _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
        }
        _output.WriteLine("");

        // Test 3: Curriculum 2 — evolve at 1K for 60s, then 10K for 60s
        _output.WriteLine($"--- Curriculum: 1K→10K (60s+60s × 3 seeds) ---");
        for (int seed = 0; seed < 3; seed++)
        {
            var r = RunCurriculumExperiment2(gpu, optPop, seed, positions);
            string status = r.SolvedGen >= 0
                ? $"SOLVED gen {r.SolvedGen} ({r.SolveTime:F1}s)"
                : $"FAILED best={r.BestFitness:F0} bestSolved={r.BestSolvedCount}/16";
            _output.WriteLine($"  Seed {seed}: {status} | {r.TotalGens} gens ({r.GenPerSec:F1} gen/s)");
        }
    }

    private RunResult RunCurriculumExperiment(
        GPUDoublePoleEvaluator gpu, int optPop, int seed, float[][] positions)
    {
        int ctx = 2;
        gpu.IncludeVelocity = false;
        gpu.ContextSize = ctx;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;
        gpu.SetStartingPositions(positions);

        var topology = BuildDenseTopology(ctx, new[] { 6 });
        var config = new EvolutionConfig
        {
            SpeciesCount = 1, IndividualsPerSpecies = optPop, MinSpeciesCount = 1,
            Elites = 10, TournamentSize = 5, ParentPoolPercentage = 0.5f,
        };
        config.MutationRates.WeightJitterStdDev = 0.15f;

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var phases = new[] { (maxSteps: 500, budget: 30), (maxSteps: 2_000, budget: 30), (maxSteps: 10_000, budget: 60) };

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int bestSolvedCount = 0;
        int solvedGen = -1;
        double solveTime = 0;
        double phaseStart = 0;

        foreach (var (maxSteps, phaseBudget) in phases)
        {
            gpu.MaxSteps = maxSteps;
            phaseStart = sw.Elapsed.TotalSeconds;

            while (sw.Elapsed.TotalSeconds - phaseStart < phaseBudget)
            {
                var sp = population.AllSpecies[0];
                var (fitness, allSolved) = gpu.EvaluateAllPositions(sp.Topology, sp.Individuals, seed: gen);
                float genBest = float.MinValue;
                for (int i = 0; i < sp.Individuals.Count; i++)
                {
                    var ind = sp.Individuals[i];
                    ind.Fitness = fitness[i];
                    sp.Individuals[i] = ind;
                    if (fitness[i] > genBest) genBest = fitness[i];
                }
                if (genBest > bestEver) bestEver = genBest;
                if (allSolved > bestSolvedCount) bestSolvedCount = allSolved;

                if (allSolved > 0 && solvedGen < 0 && maxSteps == 10_000)
                {
                    solvedGen = gen;
                    solveTime = sw.Elapsed.TotalSeconds;
                    return new RunResult(solvedGen, solveTime, bestEver, gen,
                        gen / sw.Elapsed.TotalSeconds, sw.Elapsed.TotalSeconds, bestSolvedCount);
                }

                evolver.StepGeneration(population);
                gen++;
            }
        }

        double wall = sw.Elapsed.TotalSeconds;
        return new RunResult(solvedGen, solveTime, bestEver, gen, gen / wall, wall, bestSolvedCount);
    }

    private RunResult RunCurriculumExperiment2(
        GPUDoublePoleEvaluator gpu, int optPop, int seed, float[][] positions)
    {
        int ctx = 2;
        gpu.IncludeVelocity = false;
        gpu.ContextSize = ctx;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;
        gpu.SetStartingPositions(positions);

        var topology = BuildDenseTopology(ctx, new[] { 6 });
        var config = new EvolutionConfig
        {
            SpeciesCount = 1, IndividualsPerSpecies = optPop, MinSpeciesCount = 1,
            Elites = 10, TournamentSize = 5, ParentPoolPercentage = 0.5f,
        };
        config.MutationRates.WeightJitterStdDev = 0.15f;

        var evolver = new Evolver(seed: seed);
        var population = evolver.InitializePopulation(config, topology);

        var phases = new[] { (maxSteps: 1_000, budget: 60), (maxSteps: 10_000, budget: 60) };

        var sw = Stopwatch.StartNew();
        int gen = 0;
        float bestEver = 0f;
        int bestSolvedCount = 0;
        int solvedGen = -1;
        double solveTime = 0;
        double phaseStart = 0;

        foreach (var (maxSteps, phaseBudget) in phases)
        {
            gpu.MaxSteps = maxSteps;
            phaseStart = sw.Elapsed.TotalSeconds;

            while (sw.Elapsed.TotalSeconds - phaseStart < phaseBudget)
            {
                var sp = population.AllSpecies[0];
                var (fitness, allSolved) = gpu.EvaluateAllPositions(sp.Topology, sp.Individuals, seed: gen);
                float genBest = float.MinValue;
                for (int i = 0; i < sp.Individuals.Count; i++)
                {
                    var ind = sp.Individuals[i];
                    ind.Fitness = fitness[i];
                    sp.Individuals[i] = ind;
                    if (fitness[i] > genBest) genBest = fitness[i];
                }
                if (genBest > bestEver) bestEver = genBest;
                if (allSolved > bestSolvedCount) bestSolvedCount = allSolved;

                if (allSolved > 0 && solvedGen < 0 && maxSteps >= 10_000)
                {
                    solvedGen = gen;
                    solveTime = sw.Elapsed.TotalSeconds;
                    return new RunResult(solvedGen, solveTime, bestEver, gen,
                        gen / sw.Elapsed.TotalSeconds, sw.Elapsed.TotalSeconds, bestSolvedCount);
                }

                evolver.StepGeneration(population);
                gen++;
            }
        }

        double wall = sw.Elapsed.TotalSeconds;
        return new RunResult(solvedGen, solveTime, bestEver, gen, gen / wall, wall, bestSolvedCount);
    }

    private void Warmup(GPUDoublePoleEvaluator gpu, float[][] positions)
    {
        gpu.MaxSteps = 100;
        gpu.ContextSize = 2;
        gpu.IsJordan = false;
        gpu.UseGruauFitness = false;
        gpu.SetStartingPositions(positions);

        var topology = BuildDenseTopology(2, new[] { 6 });
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 100 };
        var evolver = new Evolver(seed: 0);
        var pop = evolver.InitializePopulation(config, topology);
        gpu.EvaluateAllPositions(pop.AllSpecies[0].Topology, pop.AllSpecies[0].Individuals, seed: 0);
    }

    private static double Median(List<double> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        int n = sorted.Count;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
    }

    private record RunResult(int SolvedGen, double SolveTime, float BestFitness,
        int TotalGens, double GenPerSec, double WallTime, int BestSolvedCount);
}
