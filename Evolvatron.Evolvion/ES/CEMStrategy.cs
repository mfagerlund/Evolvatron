namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Cross-Entropy Method: refit distribution to top-K elites.
/// Per-parameter σ adaptation (diagonal approximation of CMA-ES covariance).
/// Simpler than ES, fewer hyperparameters, robust at large population sizes.
/// </summary>
public class CEMStrategy : IUpdateStrategy
{
    public float EliteFraction { get; set; }
    public float MuSmoothing { get; set; }
    public float SigmaSmoothing { get; set; }
    public float MinSigma { get; set; }
    public float MaxSigma { get; set; }

    public CEMStrategy(IslandConfig config)
    {
        EliteFraction = config.CEMEliteFraction;
        MuSmoothing = config.CEMMuSmoothing;
        SigmaSmoothing = config.CEMSigmaSmoothing;
        MinSigma = config.MinSigma;
        MaxSigma = config.MaxSigma;
    }

    public void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng)
    {
        int paramCount = island.Mu.Length;
        for (int i = 0; i < popSize; i++)
        {
            int offset = i * paramCount;
            for (int p = 0; p < paramCount; p++)
                paramVectors[offset + p] = island.Mu[p] + island.Sigma[p] * Island.SampleGaussian(rng);
        }
    }

    public void Update(Island island, ReadOnlySpan<float> fitnesses,
                       ReadOnlySpan<float> paramVectors, int popSize)
    {
        int paramCount = island.Mu.Length;
        int eliteCount = Math.Max(1, (int)(popSize * EliteFraction));

        // Copy fitness to sortable array (ReadOnlySpan can't be captured in lambda)
        var fitnessArr = new float[popSize];
        fitnesses.CopyTo(fitnessArr);

        // Find elite indices (top-K by fitness, descending)
        var indices = new int[popSize];
        for (int i = 0; i < popSize; i++) indices[i] = i;
        Array.Sort(indices, (a, b) => fitnessArr[b].CompareTo(fitnessArr[a]));

        // Refit μ and σ to elites
        for (int p = 0; p < paramCount; p++)
        {
            float sum = 0, sumSq = 0;
            for (int i = 0; i < eliteCount; i++)
            {
                float v = paramVectors[indices[i] * paramCount + p];
                sum += v;
                sumSq += v * v;
            }
            float eliteMean = sum / eliteCount;
            float eliteVar = sumSq / eliteCount - eliteMean * eliteMean;

            island.Mu[p] = (1f - MuSmoothing) * eliteMean + MuSmoothing * island.Mu[p];
            float newSigma = (1f - SigmaSmoothing) * MathF.Sqrt(MathF.Max(0f, eliteVar))
                           + SigmaSmoothing * island.Sigma[p];
            island.Sigma[p] = MathF.Max(MinSigma, MathF.Min(MaxSigma, newSigma));
        }
    }
}
