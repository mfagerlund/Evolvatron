namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration for mutation probabilities
/// </summary>
public class MutationConfig
{
    public float WeightJitter { get; set; } = 0.9f;
    public float WeightReset { get; set; } = 0.05f;
    public float WeightL1Shrink { get; set; } = 0.1f;
    public float ActivationSwap { get; set; } = 0.01f;
    public float EdgeAdd { get; set; } = 0.05f;
    public float EdgeDelete { get; set; } = 0.02f;
    public float EdgeSplit { get; set; } = 0.01f;
    public float NodeParamMutate { get; set; } = 0.2f;

    public float WeightJitterSigma { get; set; } = 0.05f; // Sigma relative to weight value
    public float WeightL1ShrinkFactor { get; set; } = 0.1f; // 10% shrinkage
}

/// <summary>
/// Mutation operators for individuals within a species
/// </summary>
public static class MutationOperators
{
    /// <summary>
    /// Applies all mutation operators to an individual
    /// </summary>
    public static void Mutate(Individual individual, SpeciesSpec spec, MutationConfig config, Random random)
    {
        // Weight mutations
        if (random.NextSingle() < config.WeightJitter)
            ApplyWeightJitter(individual, config.WeightJitterSigma, random);

        if (random.NextSingle() < config.WeightReset)
            ApplyWeightReset(individual, random);

        if (random.NextSingle() < config.WeightL1Shrink)
            ApplyWeightL1Shrink(individual, config.WeightL1ShrinkFactor);

        // Activation mutations
        if (random.NextSingle() < config.ActivationSwap)
            ApplyActivationSwap(individual, spec, random);

        // Node parameter mutations
        if (random.NextSingle() < config.NodeParamMutate)
            ApplyNodeParamMutate(individual, random);

        // Note: EdgeAdd, EdgeDelete, and EdgeSplit require topology changes
        // These would modify the SpeciesSpec and are handled at species level, not here
    }

    /// <summary>
    /// Weight Jitter: Add Gaussian noise to weights (σ = sigma * |weight|)
    /// </summary>
    public static void ApplyWeightJitter(Individual individual, float sigma, Random random)
    {
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            float noise = SampleGaussian(random) * sigma * MathF.Abs(individual.Weights[i]);
            individual.Weights[i] += noise;
        }
    }

    /// <summary>
    /// Weight Reset: Replace random weight with value from U(-1, 1)
    /// </summary>
    public static void ApplyWeightReset(Individual individual, Random random)
    {
        if (individual.Weights.Length == 0) return;

        int index = random.Next(individual.Weights.Length);
        individual.Weights[index] = random.NextSingle() * 2.0f - 1.0f;
    }

    /// <summary>
    /// Weight L1 Shrink: Reduce |w| by shrinkage factor
    /// </summary>
    public static void ApplyWeightL1Shrink(Individual individual, float shrinkFactor)
    {
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] *= (1.0f - shrinkFactor);
        }
    }

    /// <summary>
    /// Bias Jitter: Add Gaussian noise to biases (σ = sigma * |bias|)
    /// </summary>
    public static void ApplyBiasJitter(Individual individual, float sigma, Random random)
    {
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            float noise = SampleGaussian(random) * sigma * MathF.Abs(individual.Biases[i]);
            individual.Biases[i] += noise;
        }
    }

    /// <summary>
    /// Bias Reset: Replace random bias with value from U(-1, 1)
    /// </summary>
    public static void ApplyBiasReset(Individual individual, Random random)
    {
        if (individual.Biases.Length == 0) return;

        int index = random.Next(individual.Biases.Length);
        individual.Biases[index] = random.NextSingle() * 2.0f - 1.0f;
    }

    /// <summary>
    /// Bias L1 Shrink: Reduce |b| by shrinkage factor
    /// </summary>
    public static void ApplyBiasL1Shrink(Individual individual, float shrinkFactor)
    {
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] *= (1.0f - shrinkFactor);
        }
    }

    /// <summary>
    /// Activation Swap: Replace a node's activation with a random allowed activation
    /// </summary>
    public static void ApplyActivationSwap(Individual individual, SpeciesSpec spec, Random random)
    {
        if (individual.Activations.Length <= 1) return; // Skip bias node

        // Pick a random node (excluding bias at index 0)
        int nodeIndex = random.Next(1, individual.Activations.Length);
        int row = spec.GetRowForNode(nodeIndex);

        // Get allowed activations for this row
        var allowedActivations = GetAllowedActivations(spec.AllowedActivationsPerRow[row]);
        if (allowedActivations.Count == 0) return;

        // Pick a random allowed activation
        var newActivation = allowedActivations[random.Next(allowedActivations.Count)];
        individual.Activations[nodeIndex] = newActivation;

        // Update node parameters if needed
        var defaultParams = newActivation.GetDefaultParameters();
        individual.SetNodeParams(nodeIndex, defaultParams);
    }

    /// <summary>
    /// Node Param Mutate: Add Gaussian jitter to node activation parameters
    /// </summary>
    public static void ApplyNodeParamMutate(Individual individual, Random random)
    {
        if (individual.Activations.Length <= 1) return;

        // Pick a random node (excluding bias)
        int nodeIndex = random.Next(1, individual.Activations.Length);
        var activation = individual.Activations[nodeIndex];
        int requiredParams = activation.RequiredParamCount();

        if (requiredParams == 0) return;

        // Mutate the parameters
        var parameters = individual.GetNodeParams(nodeIndex).ToArray();
        for (int i = 0; i < requiredParams; i++)
        {
            float noise = SampleGaussian(random) * 0.1f; // Small jitter
            parameters[i] += noise;

            // Clamp to reasonable range
            parameters[i] = Math.Clamp(parameters[i], -10.0f, 10.0f);
        }

        individual.SetNodeParams(nodeIndex, parameters);
    }

    /// <summary>
    /// Glorot/Xavier uniform weight initialization
    /// </summary>
    public static float GlorotUniform(int fanIn, int fanOut, Random random)
    {
        float limit = MathF.Sqrt(6.0f / (fanIn + fanOut));
        return random.NextSingle() * 2.0f * limit - limit;
    }

    /// <summary>
    /// Initialize all weights using Glorot uniform initialization
    /// </summary>
    public static void InitializeWeights(Individual individual, SpeciesSpec spec, Random random)
    {
        // For simplicity, use a uniform fan-in/fan-out assumption
        // In a real implementation, compute per-edge based on actual connectivity
        int avgFanIn = spec.MaxInDegree / 2;
        int avgFanOut = spec.MaxInDegree / 2;

        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = GlorotUniform(avgFanIn, avgFanOut, random);
        }
    }

    /// <summary>
    /// Samples from standard normal distribution using Box-Muller transform
    /// </summary>
    private static float SampleGaussian(Random random)
    {
        float u1 = random.NextSingle();
        float u2 = random.NextSingle();
        return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
    }

    /// <summary>
    /// Gets list of allowed activations from bitmask
    /// </summary>
    private static List<ActivationType> GetAllowedActivations(uint mask)
    {
        var activations = new List<ActivationType>();
        for (int i = 0; i < 32; i++)
        {
            if ((mask & (1u << i)) != 0)
            {
                if (Enum.IsDefined(typeof(ActivationType), (byte)i))
                {
                    activations.Add((ActivationType)i);
                }
            }
        }
        return activations;
    }
}
