namespace Evolvatron.Evolvion.ES;

/// <summary>
/// One island in the island-model optimizer.
/// Maintains a distribution N(μ, σ²) over the parameter space.
/// </summary>
public class Island
{
    public float[] Mu;
    public float[] Sigma;
    public float BestFitness;
    public int StagnationCounter;

    // Adam state (ES only)
    public float[] AdamM;
    public float[] AdamV;
    public int AdamT;

    public Island(int paramCount, float initialSigma)
    {
        Mu = new float[paramCount];
        Sigma = new float[paramCount];
        for (int i = 0; i < paramCount; i++)
            Sigma[i] = initialSigma;
        BestFitness = float.NegativeInfinity;

        AdamM = new float[paramCount];
        AdamV = new float[paramCount];
    }

    public void AdamUpdate(ReadOnlySpan<float> gradient, float lr, float beta1, float beta2, float epsilon = 1e-8f)
    {
        AdamT++;
        float bc1 = 1f - MathF.Pow(beta1, AdamT);
        float bc2 = 1f - MathF.Pow(beta2, AdamT);

        for (int p = 0; p < Mu.Length; p++)
        {
            AdamM[p] = beta1 * AdamM[p] + (1f - beta1) * gradient[p];
            AdamV[p] = beta2 * AdamV[p] + (1f - beta2) * gradient[p] * gradient[p];

            float mHat = AdamM[p] / bc1;
            float vHat = AdamV[p] / bc2;

            Mu[p] += lr * mHat / (MathF.Sqrt(vHat) + epsilon);
        }
    }

    public static float SampleGaussian(Random rng)
    {
        float u1 = MathF.Max(1e-10f, rng.NextSingle());
        float u2 = rng.NextSingle();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }
}
