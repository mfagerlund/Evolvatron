namespace Evolvatron.Evolvion.Benchmarks;

public class LandscapeEnvironmentAdapter : IEnvironment
{
    private readonly LandscapeNavigationTask.LandscapeFunction landscape;
    private readonly int dimensions;
    private readonly int maxSteps;
    private readonly float stepSize;
    private readonly float minBound;
    private readonly float maxBound;
    private readonly ObservationType observationType;

    private float[] position;
    private float[] gradientBuffer;
    private int currentStep;
    private Random random;
    private float previousLandscapeValue;

    public LandscapeEnvironmentAdapter(LandscapeNavigationTask task)
    {
        this.landscape = task.GetLandscapeFunction();
        this.dimensions = task.GetDimensions();
        this.maxSteps = task.GetTimesteps();
        this.stepSize = task.GetStepSize();
        this.minBound = task.GetMinBound();
        this.maxBound = task.GetMaxBound();
        this.observationType = task.GetObservationType();

        this.position = new float[dimensions];
        this.gradientBuffer = new float[dimensions];
        this.random = new Random(42);
    }

    public int InputCount => GetObservationSize();
    public int OutputCount => dimensions;
    public int MaxSteps => maxSteps;

    public void Reset(int seed = 0)
    {
        random = new Random(seed);
        for (int i = 0; i < dimensions; i++)
        {
            position[i] = (float)(random.NextDouble() * (maxBound - minBound) + minBound);
        }
        currentStep = 0;
        previousLandscapeValue = landscape(position);
    }

    public void GetObservations(Span<float> observations)
    {
        switch (observationType)
        {
            case ObservationType.FullPosition:
                for (int i = 0; i < dimensions; i++)
                {
                    observations[i] = position[i];
                }
                break;

            case ObservationType.GradientOnly:
                ComputeNumericalGradient(position, gradientBuffer);
                for (int i = 0; i < dimensions; i++)
                {
                    observations[i] = gradientBuffer[i];
                }
                break;

            case ObservationType.PartialObservability:
                for (int i = 0; i < dimensions; i++)
                {
                    observations[i] = position[i];
                }
                ComputeNumericalGradient(position, gradientBuffer);
                for (int i = 0; i < dimensions; i++)
                {
                    observations[dimensions + i] = gradientBuffer[i];
                }
                break;
        }
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        for (int i = 0; i < dimensions; i++)
        {
            position[i] += actions[i] * stepSize;
        }

        OptimizationLandscapes.ClampToBounds(position, minBound, maxBound);
        currentStep++;

        return 0f;
    }

    public bool IsTerminal()
    {
        if (currentStep >= maxSteps)
        {
            return true;
        }
        return false;
    }

    public float GetFinalFitness()
    {
        return -landscape(position);
    }

    private int GetObservationSize()
    {
        return observationType switch
        {
            ObservationType.FullPosition => dimensions,
            ObservationType.GradientOnly => dimensions,
            ObservationType.PartialObservability => dimensions * 2,
            _ => throw new NotImplementedException()
        };
    }

    private void ComputeNumericalGradient(float[] pos, float[] gradient)
    {
        const float epsilon = 1e-4f;
        float baseValue = landscape(pos);

        for (int i = 0; i < dimensions; i++)
        {
            pos[i] += epsilon;
            float perturbedValue = landscape(pos);
            pos[i] -= epsilon;

            gradient[i] = (perturbedValue - baseValue) / epsilon;
        }
    }
}
