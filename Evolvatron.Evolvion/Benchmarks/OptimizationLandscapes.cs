namespace Evolvatron.Evolvion.Benchmarks;

public static class OptimizationLandscapes
{
    public static float Sphere(float[] x)
    {
        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * x[i];
        }
        return sum;
    }

    public static float Rosenbrock(float[] x)
    {
        float sum = 0f;
        for (int i = 0; i < x.Length - 1; i++)
        {
            float term1 = x[i + 1] - x[i] * x[i];
            float term2 = 1f - x[i];
            sum += 100f * term1 * term1 + term2 * term2;
        }
        return sum;
    }

    public static float Rastrigin(float[] x)
    {
        const float A = 10f;
        float sum = A * x.Length;
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * x[i] - A * MathF.Cos(2f * MathF.PI * x[i]);
        }
        return sum;
    }

    public static float Ackley(float[] x)
    {
        const float a = 20f;
        const float b = 0.2f;
        const float c = 2f * MathF.PI;

        float sumSq = 0f;
        float sumCos = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            sumSq += x[i] * x[i];
            sumCos += MathF.Cos(c * x[i]);
        }

        float n = x.Length;
        return -a * MathF.Exp(-b * MathF.Sqrt(sumSq / n))
               - MathF.Exp(sumCos / n)
               + a + MathF.E;
    }

    public static float Schwefel(float[] x)
    {
        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * MathF.Sin(MathF.Sqrt(MathF.Abs(x[i])));
        }
        return 418.9829f * x.Length - sum;
    }

    public static void ClampToBounds(float[] x, float min, float max)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = Math.Clamp(x[i], min, max);
        }
    }
}
