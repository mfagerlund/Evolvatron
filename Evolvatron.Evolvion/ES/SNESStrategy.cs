namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Separable Natural Evolution Strategies (SNES).
/// Uses natural gradient of expected fitness instead of simple elite refitting.
/// More stable sigma adaptation than CEM, handles noisy fitness better.
/// Same O(n) complexity as CEM (diagonal covariance).
///
/// Update rules (Wierstra et al. 2014):
///   z_i = (x_i - mu) / sigma                          (normalized samples)
///   mu  += eta_mu * sigma * sum(w_i * z_i)             (natural gradient on mean)
///   sigma *= exp(eta_sigma/2 * sum(w_i * (z_i^2 - 1))) (log-space sigma update)
///
/// Utilities w_i are rank-based (fitness-shaping) for scale invariance.
///
/// Mirrored sampling (optional): sample antithetic pairs (z, -z) to halve
/// gradient variance. Requires even popSize. (Salimans et al. 2017)
/// </summary>
public class SNESStrategy : IUpdateStrategy
{
    public float EtaMu { get; set; }
    public float EtaSigma { get; set; }
    public float MinSigma { get; set; }
    public float MaxSigma { get; set; }
    public bool MirroredSampling { get; set; }

    private float[]? _zVectors;

    public SNESStrategy(IslandConfig config)
    {
        EtaMu = config.SNESEtaMu;
        EtaSigma = config.SNESEtaSigma;
        MinSigma = config.MinSigma;
        MaxSigma = config.MaxSigma;
        MirroredSampling = config.SNESMirrored;
    }

    public void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng)
    {
        int paramCount = island.Mu.Length;
        _zVectors = new float[popSize * paramCount];

        if (MirroredSampling)
        {
            int numPairs = popSize / 2;
            for (int i = 0; i < numPairs; i++)
            {
                int plusOffset = (2 * i) * paramCount;
                int minusOffset = (2 * i + 1) * paramCount;
                for (int p = 0; p < paramCount; p++)
                {
                    float z = Island.SampleGaussian(rng);
                    _zVectors[plusOffset + p] = z;
                    _zVectors[minusOffset + p] = -z;
                    paramVectors[plusOffset + p] = island.Mu[p] + island.Sigma[p] * z;
                    paramVectors[minusOffset + p] = island.Mu[p] - island.Sigma[p] * z;
                }
            }
        }
        else
        {
            for (int i = 0; i < popSize; i++)
            {
                int offset = i * paramCount;
                for (int p = 0; p < paramCount; p++)
                {
                    float z = Island.SampleGaussian(rng);
                    _zVectors[offset + p] = z;
                    paramVectors[offset + p] = island.Mu[p] + island.Sigma[p] * z;
                }
            }
        }
    }

    public void Update(Island island, ReadOnlySpan<float> fitnesses,
                       ReadOnlySpan<float> paramVectors, int popSize)
    {
        if (_zVectors == null)
            throw new InvalidOperationException("GenerateSamples must be called before Update");

        int paramCount = island.Mu.Length;

        // Compute rank-based utilities (fitness shaping)
        var utilities = ComputeUtilities(fitnesses, popSize);

        // Natural gradient updates
        for (int p = 0; p < paramCount; p++)
        {
            float gradMu = 0f;
            float gradSigma = 0f;

            for (int i = 0; i < popSize; i++)
            {
                float z = _zVectors[i * paramCount + p];
                float u = utilities[i];
                gradMu += u * z;
                gradSigma += u * (z * z - 1f);
            }

            // mu += eta_mu * sigma * gradMu
            island.Mu[p] += EtaMu * island.Sigma[p] * gradMu;

            // sigma *= exp(eta_sigma/2 * gradSigma)
            float logUpdate = EtaSigma * 0.5f * gradSigma;
            logUpdate = MathF.Max(-0.5f, MathF.Min(0.5f, logUpdate)); // clamp for stability
            island.Sigma[p] *= MathF.Exp(logUpdate);
            island.Sigma[p] = MathF.Max(MinSigma, MathF.Min(MaxSigma, island.Sigma[p]));
        }
    }

    /// <summary>
    /// Rank-based fitness shaping (scale-invariant utilities).
    /// u_i = max(0, log(lambda/2 + 1) - log(rank_i)) / sum - 1/lambda
    /// </summary>
    private static float[] ComputeUtilities(ReadOnlySpan<float> fitnesses, int popSize)
    {
        // Sort indices by fitness descending
        var indices = new int[popSize];
        var fitnessArr = new float[popSize];
        for (int i = 0; i < popSize; i++)
        {
            indices[i] = i;
            fitnessArr[i] = fitnesses[i];
        }
        Array.Sort(indices, (a, b) => fitnessArr[b].CompareTo(fitnessArr[a]));

        // Compute raw utilities from ranks
        float logHalf = MathF.Log(popSize * 0.5f + 1f);
        var utilities = new float[popSize];
        float sum = 0f;

        for (int rank = 0; rank < popSize; rank++)
        {
            float raw = MathF.Max(0f, logHalf - MathF.Log(rank + 1f));
            utilities[indices[rank]] = raw;
            sum += raw;
        }

        // Normalize to sum=1, then center by subtracting 1/lambda
        float invLambda = 1f / popSize;
        if (sum > 0f)
        {
            for (int i = 0; i < popSize; i++)
                utilities[i] = utilities[i] / sum - invLambda;
        }

        return utilities;
    }
}
