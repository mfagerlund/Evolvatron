namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Population-based GA on flat param vectors (dense NN).
/// Tournament selection + elitism + Gaussian weight jitter.
/// Same interface as IslandOptimizer: generates flat param vectors for GPU evaluation.
/// </summary>
public class DenseGAOptimizer
{
    public DenseTopology Topology { get; }
    public int PopSize { get; }
    public int ParamCount { get; }
    public int Generation { get; private set; }

    // GA parameters
    public int EliteCount { get; set; } = 10;
    public int TournamentSize { get; set; } = 5;
    public float ParentPoolFraction { get; set; } = 0.5f;
    public float JitterStdDev { get; set; } = 0.15f;

    // Double-buffered population
    private float[] _current;
    private float[] _next;
    private float[] _fitnesses;
    private int[] _sortedIndices;

    public DenseGAOptimizer(DenseTopology topology, int popSize, int seed)
    {
        Topology = topology;
        PopSize = popSize;
        ParamCount = topology.TotalParams;

        _current = new float[popSize * ParamCount];
        _next = new float[popSize * ParamCount];
        _fitnesses = new float[popSize];
        _sortedIndices = new int[popSize];

        InitializeGlorot(new Random(seed));
    }

    /// <summary>
    /// Return current population as flat param vectors (ready for GPU).
    /// </summary>
    public float[] GetParamVectors() => _current;

    /// <summary>
    /// Store evaluated fitness values.
    /// </summary>
    public void Update(float[] fitness)
    {
        Array.Copy(fitness, _fitnesses, PopSize);
    }

    /// <summary>
    /// Evolve: elitism + tournament selection + weight jitter.
    /// </summary>
    public void StepGeneration(Random rng)
    {
        // Sort indices by fitness (descending)
        for (int i = 0; i < PopSize; i++) _sortedIndices[i] = i;
        var fit = _fitnesses;
        Array.Sort(_sortedIndices, (a, b) => fit[b].CompareTo(fit[a]));

        int parentPoolSize = Math.Max(EliteCount, (int)(PopSize * ParentPoolFraction));
        int slot = 0;

        // Copy elites unchanged
        for (int e = 0; e < EliteCount && e < PopSize; e++)
        {
            int srcIdx = _sortedIndices[e];
            Array.Copy(_current, srcIdx * ParamCount, _next, slot * ParamCount, ParamCount);
            slot++;
        }

        // Fill remaining with tournament-selected + jittered offspring
        while (slot < PopSize)
        {
            // Tournament selection from parent pool
            int bestIdx = _sortedIndices[rng.Next(parentPoolSize)];
            for (int t = 1; t < TournamentSize; t++)
            {
                int candidate = _sortedIndices[rng.Next(parentPoolSize)];
                if (_fitnesses[candidate] > _fitnesses[bestIdx])
                    bestIdx = candidate;
            }

            // Clone + jitter
            int srcOff = bestIdx * ParamCount;
            int dstOff = slot * ParamCount;
            for (int p = 0; p < ParamCount; p++)
                _next[dstOff + p] = _current[srcOff + p] + SampleGaussian(rng) * JitterStdDev;

            slot++;
        }

        // Swap buffers
        (_current, _next) = (_next, _current);
        Generation++;
    }

    /// <summary>
    /// Get the best individual's param vector.
    /// </summary>
    public (float[] paramVector, float fitness) GetBest()
    {
        int bestIdx = 0;
        for (int i = 1; i < PopSize; i++)
        {
            if (_fitnesses[i] > _fitnesses[bestIdx])
                bestIdx = i;
        }

        var result = new float[ParamCount];
        Array.Copy(_current, bestIdx * ParamCount, result, 0, ParamCount);
        return (result, _fitnesses[bestIdx]);
    }

    private void InitializeGlorot(Random rng)
    {
        for (int ind = 0; ind < PopSize; ind++)
        {
            int offset = ind * ParamCount;
            int wOffset = offset;
            for (int layer = 0; layer < Topology.NumLayers - 1; layer++)
            {
                int fanIn = Topology.LayerSizes[layer];
                int fanOut = Topology.LayerSizes[layer + 1];
                float stddev = MathF.Sqrt(2f / (fanIn + fanOut));
                int count = fanIn * fanOut;
                for (int w = 0; w < count; w++)
                    _current[wOffset + w] = SampleGaussian(rng) * stddev;
                wOffset += count;
            }
            // Biases remain zero
        }
    }

    private static float SampleGaussian(Random rng)
    {
        float u1 = 1f - (float)rng.NextDouble();
        float u2 = (float)rng.NextDouble();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }
}
