namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Orchestrates island-model population optimization.
/// Generates parameter vectors from island distributions,
/// dispatches strategy updates, and manages island lifecycle.
/// </summary>
public class IslandOptimizer
{
    public DenseTopology Topology { get; }
    public IslandConfig Config { get; }
    public List<Island> Islands { get; }
    public int IndividualsPerIsland { get; }
    public int TotalPopulation { get; }
    public int Generation { get; private set; }

    private readonly IUpdateStrategy _strategy;

    public IslandOptimizer(IslandConfig config, DenseTopology topology, int gpuCapacity)
    {
        Config = config;
        Topology = topology;

        // GPU-adaptive sizing
        int islandCount = config.IslandCount;
        if (gpuCapacity < islandCount * config.MinIslandPop)
            islandCount = Math.Max(1, gpuCapacity / config.MinIslandPop);

        // Round per-island pop down to even (required for ES antithetic sampling)
        IndividualsPerIsland = (gpuCapacity / islandCount) & ~1;
        TotalPopulation = islandCount * IndividualsPerIsland;

        _strategy = config.Strategy switch
        {
            UpdateStrategyType.CEM => new CEMStrategy(config),
            UpdateStrategyType.ES => new ESStrategy(config),
            _ => throw new ArgumentException($"Unknown strategy: {config.Strategy}")
        };

        Islands = new List<Island>(islandCount);
        for (int i = 0; i < islandCount; i++)
        {
            var island = new Island(topology.TotalParams, config.InitialSigma);
            InitializeGlorot(island, new Random(42 + i));
            Islands.Add(island);
        }
    }

    /// <summary>
    /// Generate flat parameter vectors for all individuals across all islands.
    /// Layout: [island0_ind0, island0_ind1, ..., island1_ind0, ...] × paramCount.
    /// </summary>
    public float[] GeneratePopulation(Random rng)
    {
        int paramCount = Topology.TotalParams;
        var paramVectors = new float[TotalPopulation * paramCount];

        for (int i = 0; i < Islands.Count; i++)
        {
            int offset = i * IndividualsPerIsland * paramCount;
            var span = paramVectors.AsSpan(offset, IndividualsPerIsland * paramCount);
            _strategy.GenerateSamples(Islands[i], span, IndividualsPerIsland, rng);
        }

        return paramVectors;
    }

    /// <summary>
    /// Update all island distributions from evaluated fitness values.
    /// </summary>
    public void Update(float[] fitnesses, float[] paramVectors)
    {
        int paramCount = Topology.TotalParams;

        for (int i = 0; i < Islands.Count; i++)
        {
            int popOffset = i * IndividualsPerIsland;
            int paramOffset = popOffset * paramCount;

            var islandFitness = new ReadOnlySpan<float>(fitnesses, popOffset, IndividualsPerIsland);
            var islandParams = new ReadOnlySpan<float>(paramVectors, paramOffset, IndividualsPerIsland * paramCount);

            _strategy.Update(Islands[i], islandFitness, islandParams, IndividualsPerIsland);

            // L2 weight decay: shrink mu toward zero
            if (Config.WeightDecay > 0f)
            {
                float factor = 1f - Config.WeightDecay;
                var mu = Islands[i].Mu;
                for (int p = 0; p < paramCount; p++)
                    mu[p] *= factor;
            }

            // Track best fitness for stagnation detection
            float currentBest = float.NegativeInfinity;
            for (int j = 0; j < IndividualsPerIsland; j++)
            {
                if (fitnesses[popOffset + j] > currentBest)
                    currentBest = fitnesses[popOffset + j];
            }

            if (currentBest > Islands[i].BestFitness + 0.01f)
            {
                Islands[i].BestFitness = currentBest;
                Islands[i].StagnationCounter = 0;
            }
            else
            {
                Islands[i].StagnationCounter++;
            }
        }

        Generation++;
    }

    /// <summary>
    /// Replace stagnant islands with perturbed copies of the best island.
    /// </summary>
    public void ManageIslands(Random rng)
    {
        if (Islands.Count <= 1) return;

        int bestIdx = 0;
        for (int i = 1; i < Islands.Count; i++)
        {
            if (Islands[i].BestFitness > Islands[bestIdx].BestFitness)
                bestIdx = i;
        }

        for (int i = 0; i < Islands.Count; i++)
        {
            if (i == bestIdx) continue;
            if (Islands[i].StagnationCounter < Config.StagnationThreshold) continue;

            var island = Islands[i];
            int paramCount = island.Mu.Length;
            Array.Copy(Islands[bestIdx].Mu, island.Mu, paramCount);

            for (int p = 0; p < paramCount; p++)
            {
                island.Mu[p] += Island.SampleGaussian(rng) * Config.ReinitSigma;
                island.Sigma[p] = Config.InitialSigma;
            }

            Array.Clear(island.AdamM);
            Array.Clear(island.AdamV);
            island.AdamT = 0;
            island.BestFitness = float.NegativeInfinity;
            island.StagnationCounter = 0;
        }
    }

    /// <summary>
    /// Warm-start: keep μ vectors, multiply σ by bump factor.
    /// Call when the environment changes.
    /// </summary>
    public void WarmStart(float sigmaBump)
    {
        foreach (var island in Islands)
        {
            for (int p = 0; p < island.Sigma.Length; p++)
                island.Sigma[p] = MathF.Min(island.Sigma[p] * sigmaBump, Config.MaxSigma);
            island.StagnationCounter = 0;
        }
    }

    /// <summary>
    /// Get the best solution across all islands.
    /// </summary>
    public (float[] mu, float bestFitness) GetBestSolution()
    {
        int bestIdx = 0;
        for (int i = 1; i < Islands.Count; i++)
        {
            if (Islands[i].BestFitness > Islands[bestIdx].BestFitness)
                bestIdx = i;
        }
        return ((float[])Islands[bestIdx].Mu.Clone(), Islands[bestIdx].BestFitness);
    }

    /// <summary>
    /// Glorot/Xavier initialization for neural network weights in the μ vector.
    /// Biases initialized to zero.
    /// </summary>
    private void InitializeGlorot(Island island, Random rng)
    {
        int offset = 0;
        for (int layer = 0; layer < Topology.NumLayers - 1; layer++)
        {
            int fanIn = Topology.LayerSizes[layer];
            int fanOut = Topology.LayerSizes[layer + 1];
            float stddev = MathF.Sqrt(2f / (fanIn + fanOut));
            int count = fanIn * fanOut;
            for (int w = 0; w < count; w++)
                island.Mu[offset + w] = Island.SampleGaussian(rng) * stddev;
            offset += count;
        }
        // Biases remain at zero (default)
    }
}
