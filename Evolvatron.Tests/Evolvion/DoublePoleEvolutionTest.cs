using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class DoublePoleEvolutionTest
{
    private readonly ITestOutputHelper _output;

    public DoublePoleEvolutionTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public void CPUDoublePole_ZeroAction_FailsQuickly()
    {
        var env = new DoublePoleEnvironment(includeVelocity: true);
        env.Reset();

        int steps = 0;
        var actions = new float[] { 0f };
        while (!env.IsTerminal() && steps < 1000)
        {
            env.Step(actions);
            steps++;
        }

        _output.WriteLine($"Zero-action controller survived {steps} steps");
        Assert.True(steps < 500, $"Zero-action should fail quickly, survived {steps}");
    }

    [Fact]
    public void GPU_Matches_CPU_DoublePole()
    {
        var topology = CreateTopology();

        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 10 };
        var population = evolver.InitializePopulation(config, topology);
        var individuals = population.AllSpecies[0].Individuals;

        var cpuEnv = new DoublePoleEnvironment(includeVelocity: true, maxSteps: 1000);
        var cpuEval = new SimpleFitnessEvaluator();
        var cpuFitness = new float[individuals.Count];
        for (int i = 0; i < individuals.Count; i++)
            cpuFitness[i] = cpuEval.Evaluate(individuals[i], topology, cpuEnv, seed: 0);

        using var gpuEval = new GPUDoublePoleEvaluator(maxIndividuals: 100);
        gpuEval.MaxSteps = 1000;
        gpuEval.TicksPerLaunch = 100;
        gpuEval.IncludeVelocity = true;
        var (gpuFitness, _) = gpuEval.EvaluatePopulation(topology, individuals, seed: 0);

        _output.WriteLine($"{"Idx",4} {"CPU",10} {"GPU",10} {"Delta",10}");
        float maxDelta = 0f;
        for (int i = 0; i < individuals.Count; i++)
        {
            float delta = MathF.Abs(cpuFitness[i] - gpuFitness[i]);
            if (delta > maxDelta) maxDelta = delta;
            _output.WriteLine($"{i,4} {cpuFitness[i],10:F0} {gpuFitness[i],10:F0} {delta,10:F0}");
        }

        _output.WriteLine($"\nMax delta: {maxDelta:F0}");
        Assert.True(maxDelta < 50f, $"CPU/GPU mismatch too large: {maxDelta}");
    }

    /// <summary>
    /// GPU evolution: 1 species then 6 species.
    /// With velocity (6 inputs, Markovian). Shared ILGPU context.
    /// </summary>
    [Fact]
    public void GPU_DoublePole_MultiSpecies_Sweep()
    {
        using var gpuEval = new GPUDoublePoleEvaluator();

        // Warmup: trigger ILGPU JIT compilation
        var warmupTopology = CreateTopology();
        var warmupEvolver = new Evolver(seed: 0);
        var warmupConfig = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 10 };
        var warmupPop = warmupEvolver.InitializePopulation(warmupConfig, warmupTopology);
        gpuEval.MaxSteps = 10;
        gpuEval.IncludeVelocity = true;
        gpuEval.EvaluatePopulation(warmupPop.AllSpecies[0].Topology, warmupPop.AllSpecies[0].Individuals, seed: 0);

        foreach (int speciesCount in new[] { 1, 6 })
        {
            RunMultiSpeciesEvolution(gpuEval, speciesCount, popPerSpecies: 2000, maxSteps: 10_000);
        }
    }

    private void RunMultiSpeciesEvolution(GPUDoublePoleEvaluator gpuEval, int speciesCount, int popPerSpecies = 2000, int maxSteps = 10_000)
    {
        int totalPop = speciesCount * popPerSpecies;
        gpuEval.MaxSteps = maxSteps;
        gpuEval.IncludeVelocity = true;

        _output.WriteLine($"\n=== Double Pole (With Velocity) — {speciesCount} species x {popPerSpecies} = {totalPop:N0} total ===");
        _output.WriteLine($"Device: {gpuEval.DeviceName} ({gpuEval.NumMultiprocessors} SMs)");

        var topology = CreateTopology();
        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig
        {
            SpeciesCount = speciesCount,
            IndividualsPerSpecies = popPerSpecies,
            MinSpeciesCount = speciesCount,
            Elites = 5,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
        };
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        const int timeBudgetSeconds = 15;
        int generation = 0;
        float bestEver = 0f;
        int solvedGeneration = -1;

        _output.WriteLine($"{"Gen",6} {"Best",10} {"Mean",10} {"Solved",8} {"Species",8} {"Time(s)",8} {"Gen/s",8}");
        _output.WriteLine(new string('-', 72));

        while (sw.Elapsed.TotalSeconds < timeBudgetSeconds)
        {
            int totalSolved = 0;
            float globalBest = float.MinValue;
            float fitnessSum = 0f;
            int fitnessCount = 0;

            foreach (var species in population.AllSpecies)
            {
                var (fitness, solved) = gpuEval.EvaluatePopulation(
                    species.Topology, species.Individuals, seed: generation);

                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = fitness[i];
                    species.Individuals[i] = ind;
                    fitnessSum += fitness[i];
                    fitnessCount++;
                    if (fitness[i] > globalBest) globalBest = fitness[i];
                }

                totalSolved += solved;
            }

            if (globalBest > bestEver) bestEver = globalBest;
            float globalMean = fitnessSum / fitnessCount;

            if (generation % 5 == 0 || totalSolved > 0 || generation < 3)
            {
                float genPerSec = generation > 0 ? generation / (float)sw.Elapsed.TotalSeconds : 0f;
                _output.WriteLine($"{generation,6} {globalBest,10:F0} {globalMean,10:F0} {totalSolved,8} {population.AllSpecies.Count,8} {sw.Elapsed.TotalSeconds,8:F1} {genPerSec,8:F1}");
            }

            if (totalSolved > 0 && solvedGeneration < 0)
            {
                solvedGeneration = generation;
                _output.WriteLine($"\n*** SOLVED at generation {generation}! ({totalSolved} individuals survived {gpuEval.MaxSteps} steps) ***\n");
            }

            evolver.StepGeneration(population);
            generation++;
        }

        _output.WriteLine($"\n--- SUMMARY ({speciesCount} species) ---");
        _output.WriteLine($"Generations: {generation}");
        _output.WriteLine($"Total evaluations: {(long)generation * totalPop:N0}");
        _output.WriteLine($"Best fitness ever: {bestEver:F0}");
        _output.WriteLine($"Solved generation: {(solvedGeneration >= 0 ? solvedGeneration.ToString() : "NOT SOLVED")}");
        _output.WriteLine($"Wall time: {sw.Elapsed.TotalSeconds:F1}s");

        _output.WriteLine($"\nFinal species topologies:");
        foreach (var species in population.AllSpecies)
        {
            var rows = string.Join("→", species.Topology.RowCounts);
            float specBest = species.Individuals.Max(ind => ind.Fitness);
            _output.WriteLine($"  {rows}  edges={species.Topology.Edges.Count}  best={specBest:F0}");
        }

        Assert.True(bestEver > 100f, $"[{speciesCount} species] Best fitness only {bestEver:F0} — no learning detected");
    }

    /// <summary>
    /// No-velocity double pole: Jordan vs Elman recurrence.
    /// Jordan: action feeds back as extra input. 4→10→1.
    /// Elman: 3 dedicated memory outputs feed back. 6→10→4.
    /// </summary>
    [Fact]
    public void GPU_DoublePole_NoVelocity_Jordan_vs_Elman()
    {
        using var gpuEval = new GPUDoublePoleEvaluator();

        // Warmup JIT with a tiny feedforward run
        var warmupTopo = CreateTopology();
        var warmupEvolver = new Evolver(seed: 0);
        var warmupPop = warmupEvolver.InitializePopulation(
            new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = 10 }, warmupTopo);
        gpuEval.MaxSteps = 10;
        gpuEval.IncludeVelocity = true;
        gpuEval.EvaluatePopulation(warmupPop.AllSpecies[0].Topology, warmupPop.AllSpecies[0].Individuals, seed: 0);

        // --- Jordan: 3 obs + 1 prev action = 4 inputs, 1 output ---
        gpuEval.IncludeVelocity = false;
        gpuEval.ContextSize = 1;
        gpuEval.IsJordan = true;
        var jordanTopo = CreateJordanTopology();
        _output.WriteLine($"Jordan topology: {string.Join("->", jordanTopo.RowCounts)} edges={jordanTopo.Edges.Count}");
        RunRecurrentEvolution(gpuEval, jordanTopo, "Jordan (ctx=1)");

        // --- Elman: 3 obs + 3 context = 6 inputs, 1 action + 3 context = 4 outputs ---
        gpuEval.ContextSize = 3;
        gpuEval.IsJordan = false;
        var elmanTopo = CreateElmanTopology(contextSize: 3);
        _output.WriteLine($"Elman topology: {string.Join("->", elmanTopo.RowCounts)} edges={elmanTopo.Edges.Count}");
        RunRecurrentEvolution(gpuEval, elmanTopo, "Elman (ctx=3)");
    }

    private void RunRecurrentEvolution(GPUDoublePoleEvaluator gpuEval, SpeciesSpec topology, string label)
    {
        const int popPerSpecies = 4000;
        const int maxSteps = 10_000;
        const int timeBudgetSeconds = 30;

        gpuEval.MaxSteps = maxSteps;

        _output.WriteLine($"\n=== DPNV {label} — {popPerSpecies} pop, {maxSteps} max steps ===");

        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig
        {
            SpeciesCount = 1,
            IndividualsPerSpecies = popPerSpecies,
            MinSpeciesCount = 1,
            Elites = 10,
            TournamentSize = 5,
            ParentPoolPercentage = 0.5f,
        };
        var population = evolver.InitializePopulation(config, topology);

        var sw = Stopwatch.StartNew();
        int generation = 0;
        float bestEver = 0f;
        int solvedGeneration = -1;

        _output.WriteLine($"{"Gen",6} {"Best",10} {"Mean",10} {"Solved",8} {"Time(s)",8} {"Gen/s",8}");
        _output.WriteLine(new string('-', 60));

        while (sw.Elapsed.TotalSeconds < timeBudgetSeconds)
        {
            var species = population.AllSpecies[0];
            var (fitness, solved) = gpuEval.EvaluatePopulation(
                species.Topology, species.Individuals, seed: generation);

            float best = float.MinValue;
            float sum = 0f;
            for (int i = 0; i < species.Individuals.Count; i++)
            {
                var ind = species.Individuals[i];
                ind.Fitness = fitness[i];
                species.Individuals[i] = ind;
                sum += fitness[i];
                if (fitness[i] > best) best = fitness[i];
            }

            if (best > bestEver) bestEver = best;
            float mean = sum / species.Individuals.Count;

            if (generation % 10 == 0 || solved > 0 || generation < 3)
            {
                float genPerSec = generation > 0 ? generation / (float)sw.Elapsed.TotalSeconds : 0f;
                _output.WriteLine($"{generation,6} {best,10:F0} {mean,10:F0} {solved,8} {sw.Elapsed.TotalSeconds,8:F1} {genPerSec,8:F1}");
            }

            if (solved > 0 && solvedGeneration < 0)
            {
                solvedGeneration = generation;
                _output.WriteLine($"\n*** SOLVED at generation {generation}! ({solved} individuals survived {maxSteps} steps) ***\n");
            }

            evolver.StepGeneration(population);
            generation++;
        }

        var rows = string.Join("->", population.AllSpecies[0].Topology.RowCounts);
        int edges = population.AllSpecies[0].Topology.Edges.Count;

        _output.WriteLine($"\n--- {label} SUMMARY ---");
        _output.WriteLine($"Generations: {generation}, Evaluations: {(long)generation * popPerSpecies:N0}");
        _output.WriteLine($"Best fitness: {bestEver:F0}");
        _output.WriteLine($"Solved: {(solvedGeneration >= 0 ? $"gen {solvedGeneration}" : "NOT SOLVED")}");
        _output.WriteLine($"Final topology: {rows} edges={edges}");

        Assert.True(bestEver > 100f, $"[{label}] Best fitness only {bestEver:F0} — no learning detected");
    }

    private static SpeciesSpec CreateTopology()
    {
        // 6 -> 10 -> 1 dense (WITH velocity: 6 inputs)
        return new SpeciesBuilder()
            .AddInputRow(6)
            .AddHiddenRow(10,
                ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.ReLU,
                ActivationType.LeakyReLU, ActivationType.Softsign, ActivationType.Sin)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(new Random(42))
            .Build();
    }

    private static SpeciesSpec CreateJordanTopology()
    {
        // Jordan: 4 inputs (3 obs + 1 prev action), 10 hidden, 1 output
        return new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(10,
                ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.ReLU,
                ActivationType.LeakyReLU, ActivationType.Softsign, ActivationType.Sin)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(new Random(42))
            .Build();
    }

    private static SpeciesSpec CreateElmanTopology(int contextSize)
    {
        // Elman: (3 + ctx) inputs, 10 hidden, (1 + ctx) outputs
        int inputSize = 3 + contextSize;
        int outputSize = 1 + contextSize;
        return new SpeciesBuilder()
            .AddInputRow(inputSize)
            .AddHiddenRow(10,
                ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.ReLU,
                ActivationType.LeakyReLU, ActivationType.Softsign, ActivationType.Sin)
            .AddOutputRow(outputSize, ActivationType.Tanh)
            .WithMaxInDegree(10)
            .InitializeDense(new Random(42))
            .Build();
    }
}
