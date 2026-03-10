using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Quick directional tests — fast iteration, not full sweeps.
/// All 5→4→4→3 (Elman ctx=2, 59 params).
///
/// RESULTS (completed):
/// - GA 160sp multi-pos(10): median 258/625 — species don't help with same topology
/// - GA 3sp multi-pos(10): median 271/625 — multi-pos hurts GA (jitter already regularizes)
/// - Pre-activation scaling: CEM -21%, ES dead — all normalization harmful
/// - Real layer norm: CEM -7% — still harmful
/// CONCLUSION: GA and normalization eliminated. CEM is flagship, ES is fallback.
/// </summary>
public class QuickDirectionalTest
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

    /// <summary>
    /// NEAT-scale GA: 160 species x ~100 individuals each.
    /// RESULT: median 258/625 — species with same topology don't help.
    /// Kept for reference. GA eliminated as candidate.
    /// </summary>
    [Fact]
    public void DenseGA_160Species_MultiPos10()
    {
        var topology = DenseTopology.ForDPNV(new[] { 4, 4 }, contextSize: 2);
        using var evaluator = new GPUDenseDoublePoleEvaluator(topology);
        evaluator.MaxSteps = 1_000;
        evaluator.ContextSize = 2;

        var fullGrid = Build625Grid();
        int gpuCapacity = evaluator.OptimalPopulationSize;
        int numSpecies = 160;
        int numTrain = 10;
        int numSeeds = 5;
        double budget = 60;

        int popPerSpecies = gpuCapacity / numSpecies;
        Console.WriteLine($"=== NEAT-Scale GA: {numSpecies} species x {popPerSpecies} individuals ===");
        Console.WriteLine($"Topology: {topology}, total pop: {popPerSpecies * numSpecies}, {numSeeds} seeds, {budget}s");
        Console.WriteLine($"Multi-pos({numTrain}) training, test on full 625 grid\n");

        var grids = new List<int>();
        for (int seed = 0; seed < numSeeds; seed++)
        {
            var trainPositions = SampleTrainingPositions(fullGrid, numTrain, seed * 1000);
            var ga = new DenseGAOptimizer(topology, gpuCapacity, numSpecies, seed)
            {
                JitterStdDev = 0.15f,
                EliteCount = 2,
                TournamentSize = 3,
                ParentPoolFraction = 0.5f,
                StagnationThreshold = 30,
            };
            var rng = new Random(seed);
            var sw = Stopwatch.StartNew();

            evaluator.SetStartingPositions(trainPositions);
            int totalGens = 0;

            for (int gen = 0; gen < 50_000; gen++)
            {
                var paramVectors = ga.GetParamVectors();
                var (fitness, _) = evaluator.EvaluateAllPositions(paramVectors, ga.TotalPopulation);
                ga.Update(fitness);
                ga.StepGeneration(rng);
                ga.ManageSpecies(rng);
                totalGens = gen + 1;
                if (sw.Elapsed.TotalSeconds > budget) break;
            }

            evaluator.SetStartingPositions(fullGrid);
            var (bestParams, _) = ga.GetBest();
            int gridScore = evaluator.EvaluateChampionGridScore(bestParams);
            grids.Add(gridScore);
            Console.WriteLine($"  Seed {seed}: grid={gridScore}/625, gens={totalGens}");
        }

        grids.Sort();
        int median = grids[grids.Count / 2];
        Console.WriteLine($"\n  Median: {median}/625, range {grids.Min()}-{grids.Max()}");
        Console.WriteLine($"  Pass>=200: {grids.Count(g => g >= 200)}/{numSeeds}");
        Console.WriteLine($"  CEM multi-pos(10)=303/625, ES multi-pos(25)=315/625");
    }
}
