namespace Evolvatron.Evolvion.Benchmarks;

public enum ObservationType
{
    FullPosition,
    GradientOnly,
    PartialObservability
}

public class LandscapeNavigationTask
{
    public delegate float LandscapeFunction(float[] position);

    private readonly LandscapeFunction landscape;
    private readonly int dimensions;
    private readonly int timesteps;
    private readonly float stepSize;
    private readonly float minBound;
    private readonly float maxBound;
    private readonly ObservationType observationType;
    private readonly Random random;

    public LandscapeNavigationTask(
        LandscapeFunction landscape,
        int dimensions,
        int timesteps,
        float stepSize = 0.1f,
        float minBound = -5f,
        float maxBound = 5f,
        ObservationType observationType = ObservationType.FullPosition,
        int seed = 42)
    {
        this.landscape = landscape;
        this.dimensions = dimensions;
        this.timesteps = timesteps;
        this.stepSize = stepSize;
        this.minBound = minBound;
        this.maxBound = maxBound;
        this.observationType = observationType;
        this.random = new Random(seed);
    }

    public float Evaluate(Func<float[], float[]> policy)
    {
        var position = InitializePosition();
        var inputBuffer = new float[GetObservationSize()];
        var gradientBuffer = new float[dimensions];

        for (int t = 0; t < timesteps; t++)
        {
            GetObservation(position, gradientBuffer, inputBuffer);

            var action = policy(inputBuffer);
            if (action.Length != dimensions)
            {
                throw new ArgumentException($"Policy must output {dimensions} values");
            }

            for (int i = 0; i < dimensions; i++)
            {
                position[i] += action[i] * stepSize;
            }

            OptimizationLandscapes.ClampToBounds(position, minBound, maxBound);
        }

        return -landscape(position);
    }

    private float[] InitializePosition()
    {
        var position = new float[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            position[i] = (float)(random.NextDouble() * (maxBound - minBound) + minBound);
        }
        return position;
    }

    public int GetObservationSize()
    {
        return observationType switch
        {
            ObservationType.FullPosition => dimensions,
            ObservationType.GradientOnly => dimensions,
            ObservationType.PartialObservability => dimensions * 2,
            _ => throw new NotImplementedException()
        };
    }

    public int GetDimensions() => dimensions;
    public int GetTimesteps() => timesteps;
    public float GetStepSize() => stepSize;
    public float GetMinBound() => minBound;
    public float GetMaxBound() => maxBound;
    public ObservationType GetObservationType() => observationType;
    public LandscapeFunction GetLandscapeFunction() => landscape;

    private void GetObservation(float[] position, float[] gradientBuffer, float[] observation)
    {
        switch (observationType)
        {
            case ObservationType.FullPosition:
                Array.Copy(position, observation, dimensions);
                break;

            case ObservationType.GradientOnly:
                ComputeNumericalGradient(position, gradientBuffer);
                Array.Copy(gradientBuffer, observation, dimensions);
                break;

            case ObservationType.PartialObservability:
                Array.Copy(position, observation, dimensions);
                ComputeNumericalGradient(position, gradientBuffer);
                Array.Copy(gradientBuffer, 0, observation, dimensions, dimensions);
                break;
        }
    }

    private void ComputeNumericalGradient(float[] position, float[] gradient)
    {
        const float epsilon = 1e-4f;
        float baseValue = landscape(position);

        for (int i = 0; i < dimensions; i++)
        {
            position[i] += epsilon;
            float perturbedValue = landscape(position);
            position[i] -= epsilon;

            gradient[i] = (perturbedValue - baseValue) / epsilon;
        }
    }
}
