namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Multi-species population-based GA on flat param vectors (dense NN).
/// Each species evolves independently via tournament selection + elitism + Gaussian weight jitter.
/// All species share the same topology but explore different weight configurations.
/// GPU evaluates all species in one batch. Stagnant species replaced by best species + perturbation.
/// </summary>
public class DenseGAOptimizer
{
    public DenseTopology Topology { get; }
    public int NumSpecies { get; }
    public int PopPerSpecies { get; }
    public int TotalPopulation { get; }
    public int ParamCount { get; }
    public int Generation { get; private set; }

    // GA parameters
    public int EliteCount { get; set; } = 10;
    public int TournamentSize { get; set; } = 5;
    public float ParentPoolFraction { get; set; } = 0.5f;
    public float JitterStdDev { get; set; } = 0.15f;
    public int StagnationThreshold { get; set; } = 50;
    public float ReinitJitter { get; set; } = 0.3f;

    // Double-buffered population (all species concatenated)
    private float[] _current;
    private float[] _next;
    private float[] _fitnesses;

    // Per-species tracking
    private readonly float[] _speciesBestFitness;
    private readonly int[] _speciesStagnation;

    // Scratch buffer for per-species sorting
    private readonly int[] _sortedIndices;

    public DenseGAOptimizer(DenseTopology topology, int gpuCapacity, int numSpecies, int seed)
    {
        Topology = topology;
        NumSpecies = numSpecies;
        PopPerSpecies = gpuCapacity / numSpecies;
        TotalPopulation = PopPerSpecies * numSpecies;
        ParamCount = topology.TotalParams;

        _current = new float[TotalPopulation * ParamCount];
        _next = new float[TotalPopulation * ParamCount];
        _fitnesses = new float[TotalPopulation];
        _sortedIndices = new int[PopPerSpecies];

        _speciesBestFitness = new float[numSpecies];
        _speciesStagnation = new int[numSpecies];
        for (int s = 0; s < numSpecies; s++)
            _speciesBestFitness[s] = float.NegativeInfinity;

        // Each species gets a different seed for Glorot init
        for (int s = 0; s < numSpecies; s++)
            InitializeGlorot(s, new Random(seed + s * 7919));
    }

    public float[] GetParamVectors() => _current;

    public void Update(float[] fitness)
    {
        Array.Copy(fitness, _fitnesses, TotalPopulation);
    }

    /// <summary>
    /// Evolve each species independently: elitism + tournament selection + weight jitter.
    /// </summary>
    public void StepGeneration(Random rng)
    {
        for (int s = 0; s < NumSpecies; s++)
            EvolveSpecies(s, rng);

        (_current, _next) = (_next, _current);
        Generation++;
    }

    /// <summary>
    /// Replace stagnant species with perturbed copies of the best species.
    /// </summary>
    public void ManageSpecies(Random rng)
    {
        if (NumSpecies <= 1) return;

        // Track stagnation
        for (int s = 0; s < NumSpecies; s++)
        {
            float best = float.NegativeInfinity;
            int baseIdx = s * PopPerSpecies;
            for (int i = 0; i < PopPerSpecies; i++)
            {
                if (_fitnesses[baseIdx + i] > best)
                    best = _fitnesses[baseIdx + i];
            }

            if (best > _speciesBestFitness[s] + 0.01f)
            {
                _speciesBestFitness[s] = best;
                _speciesStagnation[s] = 0;
            }
            else
            {
                _speciesStagnation[s]++;
            }
        }

        // Find best species
        int bestSpecies = 0;
        for (int s = 1; s < NumSpecies; s++)
        {
            if (_speciesBestFitness[s] > _speciesBestFitness[bestSpecies])
                bestSpecies = s;
        }

        // Replace stagnant species
        for (int s = 0; s < NumSpecies; s++)
        {
            if (s == bestSpecies) continue;
            if (_speciesStagnation[s] < StagnationThreshold) continue;

            int srcBase = bestSpecies * PopPerSpecies * ParamCount;
            int dstBase = s * PopPerSpecies * ParamCount;

            // Copy best species' population + add perturbation
            Array.Copy(_current, srcBase, _current, dstBase, PopPerSpecies * ParamCount);
            for (int i = 0; i < PopPerSpecies * ParamCount; i++)
                _current[dstBase + i] += SampleGaussian(rng) * ReinitJitter;

            _speciesBestFitness[s] = float.NegativeInfinity;
            _speciesStagnation[s] = 0;
        }
    }

    public (float[] paramVector, float fitness) GetBest()
    {
        int bestIdx = 0;
        for (int i = 1; i < TotalPopulation; i++)
        {
            if (_fitnesses[i] > _fitnesses[bestIdx])
                bestIdx = i;
        }

        var result = new float[ParamCount];
        Array.Copy(_current, bestIdx * ParamCount, result, 0, ParamCount);
        return (result, _fitnesses[bestIdx]);
    }

    private void EvolveSpecies(int speciesIdx, Random rng)
    {
        int baseIdx = speciesIdx * PopPerSpecies;
        int baseParam = baseIdx * ParamCount;

        // Sort this species' indices by fitness (descending)
        for (int i = 0; i < PopPerSpecies; i++) _sortedIndices[i] = baseIdx + i;
        var fit = _fitnesses;
        Array.Sort(_sortedIndices, (a, b) => fit[b].CompareTo(fit[a]));

        int parentPoolSize = Math.Max(EliteCount, (int)(PopPerSpecies * ParentPoolFraction));
        int slot = 0;

        // Copy elites unchanged
        for (int e = 0; e < EliteCount && e < PopPerSpecies; e++)
        {
            int srcIdx = _sortedIndices[e];
            int dstOff = (baseIdx + slot) * ParamCount;
            Array.Copy(_current, srcIdx * ParamCount, _next, dstOff, ParamCount);
            slot++;
        }

        // Fill remaining with tournament-selected + jittered offspring
        while (slot < PopPerSpecies)
        {
            int bestIdx = _sortedIndices[rng.Next(parentPoolSize)];
            for (int t = 1; t < TournamentSize; t++)
            {
                int candidate = _sortedIndices[rng.Next(parentPoolSize)];
                if (_fitnesses[candidate] > _fitnesses[bestIdx])
                    bestIdx = candidate;
            }

            int srcOff = bestIdx * ParamCount;
            int dstOff = (baseIdx + slot) * ParamCount;
            for (int p = 0; p < ParamCount; p++)
                _next[dstOff + p] = _current[srcOff + p] + SampleGaussian(rng) * JitterStdDev;

            slot++;
        }
    }

    private void InitializeGlorot(int speciesIdx, Random rng)
    {
        int baseIdx = speciesIdx * PopPerSpecies;
        for (int ind = 0; ind < PopPerSpecies; ind++)
        {
            int offset = (baseIdx + ind) * ParamCount;
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
        }
    }

    private static float SampleGaussian(Random rng)
    {
        float u1 = 1f - (float)rng.NextDouble();
        float u2 = (float)rng.NextDouble();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }
}
