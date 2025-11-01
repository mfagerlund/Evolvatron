using Evolvatron.Evolvion;
namespace Evolvatron.Tests.Evolvion;
public class BiasMutationTests
{
    [Fact]
    public void BiasMutation_AppliesJitter()
    {
        var individual = new Individual(5, 3) { Biases = new[] { 1.0f, 2.0f, 3.0f } };
        var originalBiases = (float[])individual.Biases.Clone();
        var random = new Random(42);
        MutationOperators.ApplyBiasJitter(individual, sigma: 0.1f, random);
        bool anyChanged = false;
        for (int i = 0; i < individual.Biases.Length; i++)
            if (MathF.Abs(individual.Biases[i] - originalBiases[i]) > 1e-6f) anyChanged = true;
        Assert.True(anyChanged);
    }

    [Fact]
    public void BiasMutation_AppliesReset()
    {
        var individual = new Individual(5, 3) { Biases = new[] { 1.0f, 1.0f, 1.0f } };
        var random = new Random(42);
        for (int i = 0; i < 10; i++) MutationOperators.ApplyBiasReset(individual, random);
        int changedCount = 0;
        for (int i = 0; i < individual.Biases.Length; i++)
            if (MathF.Abs(individual.Biases[i] - 1.0f) > 0.01f) changedCount++;
        Assert.True(changedCount > 0);
    }

    [Fact]
    public void BiasMutation_AppliesL1Shrink()
    {
        var individual = new Individual(5, 3) { Biases = new[] { 1.0f, -1.0f, 0.5f } };
        MutationOperators.ApplyBiasL1Shrink(individual, 0.1f);
        Assert.Equal(0.9f, individual.Biases[0], precision: 6);
        Assert.Equal(-0.9f, individual.Biases[1], precision: 6);
        Assert.Equal(0.45f, individual.Biases[2], precision: 6);
    }

    [Fact]
    public void BiasesNotNull_AfterCreation()
    {
        var individual = new Individual(edgeCount: 5, nodeCount: 3);
        Assert.NotNull(individual.Biases);
        Assert.Equal(3, individual.Biases.Length);
    }

    [Fact]
    public void BiasesMutate_DuringEvolution()
    {
        var random = new Random(42);
        var spec = new SpeciesBuilder()
            .AddInputRow(4).AddHiddenRow(3, ActivationType.ReLU, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Linear).InitializeSparse(random).Build();
        var population = new List<Individual>();
        for (int i = 0; i < 10; i++)
        {
            var ind = new Individual(spec.Edges.Count, spec.TotalNodes);
            for (int j = 0; j < ind.Biases.Length; j++) ind.Biases[j] = 1.0f;
            population.Add(ind);
        }
        for (int gen = 0; gen < 5; gen++)
            foreach (var ind in population)
                MutationOperators.ApplyBiasJitter(ind, sigma: 0.1f, random);
        float varianceSum = 0;
        for (int bi = 0; bi < population[0].Biases.Length; bi++)
        {
            var vals = new List<float>();
            foreach (var p in population) vals.Add(p.Biases[bi]);
            float mean = 0; foreach (var v in vals) mean += v; mean /= vals.Count;
            float var = 0; foreach (var v in vals) var += (v - mean) * (v - mean); var /= vals.Count;
            varianceSum += var;
        }
        Assert.True(varianceSum > 0.001f);
    }

    [Fact]
    public void BiasL1Shrink_ReducesMagnitude()
    {
        var individual = new Individual(10, 5) { Biases = new float[10] };
        for (int i = 0; i < 10; i++) individual.Biases[i] = 2.0f;
        MutationOperators.ApplyBiasL1Shrink(individual, 0.2f);
        bool allReduced = true;
        for (int i = 0; i < 10; i++)
            if (MathF.Abs(individual.Biases[i] - 1.6f) > 0.01f) allReduced = false;
        Assert.True(allReduced);
    }

    [Fact]
    public void BiasL1Shrink_PreservesSign()
    {
        var individual = new Individual(4, 2) { Biases = new[] { 2.0f, -2.0f, 0.5f, -0.5f } };
        MutationOperators.ApplyBiasL1Shrink(individual, 0.2f);
        Assert.True(individual.Biases[0] > 0);
        Assert.True(individual.Biases[1] < 0);
    }

    [Fact]
    public void BiasL1Shrink_AppliesUniformly()
    {
        var individual = new Individual(3, 2) { Biases = new[] { 1.0f, 2.0f, 3.0f } };
        MutationOperators.ApplyBiasL1Shrink(individual, 0.25f);
        Assert.Equal(0.75f, individual.Biases[0], precision: 6);
        Assert.Equal(1.5f, individual.Biases[1], precision: 6);
        Assert.Equal(2.25f, individual.Biases[2], precision: 6);
    }

    [Fact]
    public void BiasesPreserved_InDeepCopy()
    {
        var original = new Individual(5, 3) { Biases = new[] { 1.0f, 2.0f, 3.0f } };
        var copy = new Individual(original);
        for (int i = 0; i < 3; i++) Assert.Equal(original.Biases[i], copy.Biases[i]);
        copy.Biases[0] = 999.0f;
        Assert.Equal(1.0f, original.Biases[0]);
    }

    [Fact]
    public void BiasJitter_Deterministic_WithSameSeed()
    {
        var ind1 = new Individual(10, 5) { Biases = new float[10] };
        var ind2 = new Individual(10, 5) { Biases = new float[10] };
        for (int i = 0; i < 10; i++) { ind1.Biases[i] = 0.5f; ind2.Biases[i] = 0.5f; }
        MutationOperators.ApplyBiasJitter(ind1, sigma: 0.1f, new Random(42));
        MutationOperators.ApplyBiasJitter(ind2, sigma: 0.1f, new Random(42));
        for (int i = 0; i < 10; i++) Assert.Equal(ind1.Biases[i], ind2.Biases[i]);
    }

    [Fact]
    public void BiasReset_Deterministic_WithSameSeed()
    {
        var ind1 = new Individual(10, 5) { Biases = new float[10] };
        var ind2 = new Individual(10, 5) { Biases = new float[10] };
        for (int i = 0; i < 10; i++) { ind1.Biases[i] = 0.5f; ind2.Biases[i] = 0.5f; }
        MutationOperators.ApplyBiasReset(ind1, new Random(42));
        MutationOperators.ApplyBiasReset(ind2, new Random(42));
        for (int i = 0; i < 10; i++) Assert.Equal(ind1.Biases[i], ind2.Biases[i]);
    }

    [Fact]
    public void BiasReset_ExploresParameterSpace()
    {
        var ind = new Individual(5, 5) { Biases = new float[5] };
        var random = new Random(42);
        var biasValues = new HashSet<float>();
        for (int i = 0; i < 100; i++)
        {
            MutationOperators.ApplyBiasReset(ind, random);
            for (int j = 0; j < ind.Biases.Length; j++) biasValues.Add(ind.Biases[j]);
        }
        Assert.True(biasValues.Count > 20);
        bool hasNeg = false, hasPos = false;
        foreach (var v in biasValues)
        {
            if (v < -0.5f) hasNeg = true;
            if (v > 0.5f) hasPos = true;
        }
        Assert.True(hasNeg && hasPos);
    }
}
