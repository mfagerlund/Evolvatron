namespace Evolvatron.Evolvion.ES;

/// <summary>
/// OpenAI Evolution Strategies: gradient estimation from antithetic sampling + Adam optimizer.
/// Uses all individuals (not just elites) for gradient estimation.
/// Requires even population sizes (antithetic pairs).
/// </summary>
public class ESStrategy : IUpdateStrategy
{
    public float Sigma { get; set; }
    public float LearningRate { get; set; }
    public float AdamBeta1 { get; set; }
    public float AdamBeta2 { get; set; }

    private float[]? _noiseVectors;

    public ESStrategy(IslandConfig config)
    {
        Sigma = config.ESSigma;
        LearningRate = config.ESLearningRate;
        AdamBeta1 = config.ESAdamBeta1;
        AdamBeta2 = config.ESAdamBeta2;
    }

    public void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng)
    {
        int paramCount = island.Mu.Length;
        int numPairs = popSize / 2;
        _noiseVectors = new float[numPairs * paramCount];

        for (int i = 0; i < numPairs; i++)
        {
            int noiseOffset = i * paramCount;
            int plusOffset = (2 * i) * paramCount;
            int minusOffset = (2 * i + 1) * paramCount;

            for (int p = 0; p < paramCount; p++)
            {
                float eps = Island.SampleGaussian(rng);
                _noiseVectors[noiseOffset + p] = eps;
                paramVectors[plusOffset + p] = island.Mu[p] + Sigma * eps;
                paramVectors[minusOffset + p] = island.Mu[p] - Sigma * eps;
            }
        }
    }

    public void Update(Island island, ReadOnlySpan<float> fitnesses,
                       ReadOnlySpan<float> paramVectors, int popSize)
    {
        if (_noiseVectors == null)
            throw new InvalidOperationException("GenerateSamples must be called before Update");

        int paramCount = island.Mu.Length;
        int numPairs = popSize / 2;

        var gradient = new float[paramCount];
        for (int i = 0; i < numPairs; i++)
        {
            float fPlus = fitnesses[2 * i];
            float fMinus = fitnesses[2 * i + 1];
            float diff = fPlus - fMinus;
            int noiseOffset = i * paramCount;

            for (int p = 0; p < paramCount; p++)
                gradient[p] += diff * _noiseVectors[noiseOffset + p];
        }

        float scale = 1f / (numPairs * Sigma);
        for (int p = 0; p < paramCount; p++)
            gradient[p] *= scale;

        island.AdamUpdate(gradient, LearningRate, AdamBeta1, AdamBeta2);
    }
}
