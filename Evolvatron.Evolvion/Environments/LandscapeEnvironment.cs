using Evolvatron.Evolvion.Benchmarks;

namespace Evolvatron.Evolvion.Environments;

public class LandscapeEnvironment : IEnvironment
{
    private readonly LandscapeNavigationTask _task;
    private float[] _currentPosition;
    private float[] _gradientBuffer;
    private int _currentStep;
    private Random _random;

    public LandscapeEnvironment(LandscapeNavigationTask task)
    {
        _task = task;
        _currentPosition = new float[task.GetDimensions()];
        _gradientBuffer = new float[task.GetDimensions()];
        _random = new Random(42);
    }

    public int InputCount => _task.GetObservationSize();
    public int OutputCount => _task.GetDimensions();
    public int MaxSteps => _task.GetTimesteps();

    public void Reset(int seed = 0)
    {
        _random = new Random(seed);
        _currentStep = 0;

        for (int i = 0; i < _task.GetDimensions(); i++)
        {
            _currentPosition[i] = (float)(_random.NextDouble() *
                (_task.GetMaxBound() - _task.GetMinBound()) + _task.GetMinBound());
        }
    }

    public void GetObservations(Span<float> observations)
    {
        switch (_task.GetObservationType())
        {
            case ObservationType.FullPosition:
                for (int i = 0; i < _task.GetDimensions(); i++)
                {
                    observations[i] = _currentPosition[i];
                }
                break;

            case ObservationType.GradientOnly:
                ComputeNumericalGradient(_currentPosition, _gradientBuffer);
                for (int i = 0; i < _task.GetDimensions(); i++)
                {
                    observations[i] = _gradientBuffer[i];
                }
                break;

            case ObservationType.PartialObservability:
                for (int i = 0; i < _task.GetDimensions(); i++)
                {
                    observations[i] = _currentPosition[i];
                }
                ComputeNumericalGradient(_currentPosition, _gradientBuffer);
                for (int i = 0; i < _task.GetDimensions(); i++)
                {
                    observations[_task.GetDimensions() + i] = _gradientBuffer[i];
                }
                break;
        }
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        for (int i = 0; i < _task.GetDimensions(); i++)
        {
            _currentPosition[i] += actions[i] * _task.GetStepSize();
        }

        OptimizationLandscapes.ClampToBounds(
            _currentPosition,
            _task.GetMinBound(),
            _task.GetMaxBound());

        _currentStep++;
        return 0f;
    }

    public bool IsTerminal()
    {
        return _currentStep >= MaxSteps;
    }

    public float GetFinalFitness()
    {
        return -_task.GetLandscapeFunction()(_currentPosition);
    }

    private void ComputeNumericalGradient(float[] position, float[] gradient)
    {
        const float epsilon = 1e-4f;
        float baseValue = _task.GetLandscapeFunction()(position);

        for (int i = 0; i < _task.GetDimensions(); i++)
        {
            position[i] += epsilon;
            float perturbedValue = _task.GetLandscapeFunction()(position);
            position[i] -= epsilon;

            gradient[i] = (perturbedValue - baseValue) / epsilon;
        }
    }
}
