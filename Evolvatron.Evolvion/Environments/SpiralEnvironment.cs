namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Two-spiral classification problem.
/// Points are arranged in two interleaved spirals, must classify which spiral each point belongs to.
/// Classic non-linearly separable benchmark for testing neural network capacity.
/// </summary>
public class SpiralEnvironment : IEnvironment
{
    private readonly List<(float x, float y, float label)> _testCases;
    private int _currentCase;
    private float _totalError;

    public int InputCount => 2; // (x, y) coordinates
    public int OutputCount => 1; // Binary classification: 0 or 1
    public int MaxSteps { get; }

    public SpiralEnvironment(int pointsPerSpiral = 50, float noise = 0.0f)
    {
        _testCases = GenerateSpiralPoints(pointsPerSpiral, noise);
        MaxSteps = _testCases.Count;
    }

    public void Reset(int seed = 0)
    {
        _currentCase = 0;
        _totalError = 0f;
    }

    public void GetObservations(Span<float> observations)
    {
        if (_currentCase >= _testCases.Count)
        {
            observations[0] = 0f;
            observations[1] = 0f;
            return;
        }

        var (x, y, _) = _testCases[_currentCase];
        observations[0] = x;
        observations[1] = y;
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        if (_currentCase >= _testCases.Count)
            return 0f;

        var (_, _, expected) = _testCases[_currentCase];
        float output = actions[0];

        // Compute squared error
        float error = (output - expected) * (output - expected);
        _totalError += error;

        _currentCase++;

        // Return fitness at end of episode
        if (_currentCase >= _testCases.Count)
        {
            // Fitness = -average_error (higher is better)
            return -(_totalError / _testCases.Count);
        }

        return 0f;
    }

    public bool IsTerminal()
    {
        return _currentCase >= _testCases.Count;
    }

    private static List<(float x, float y, float label)> GenerateSpiralPoints(int pointsPerSpiral, float noise)
    {
        var points = new List<(float x, float y, float label)>();
        var random = new Random(42); // Fixed seed for reproducibility - all networks evaluated on same problem

        for (int i = 0; i < pointsPerSpiral; i++)
        {
            float t = i * 4.0f * MathF.PI / pointsPerSpiral; // Angle (0 to 4*pi = 2 full rotations)
            float r = t / (4.0f * MathF.PI); // Radius grows linearly

            // Spiral 1 (label = 0, output should be -1 for tanh)
            float x1 = r * MathF.Cos(t) + (noise > 0 ? (float)(random.NextDouble() - 0.5) * noise : 0f);
            float y1 = r * MathF.Sin(t) + (noise > 0 ? (float)(random.NextDouble() - 0.5) * noise : 0f);
            points.Add((x1, y1, -1f));

            // Spiral 2 (label = 1, output should be +1 for tanh)
            float x2 = r * MathF.Cos(t + MathF.PI) + (noise > 0 ? (float)(random.NextDouble() - 0.5) * noise : 0f);
            float y2 = r * MathF.Sin(t + MathF.PI) + (noise > 0 ? (float)(random.NextDouble() - 0.5) * noise : 0f);
            points.Add((x2, y2, 1f));
        }

        return points;
    }

    /// <summary>
    /// Get all test points for visualization purposes.
    /// </summary>
    public IReadOnlyList<(float x, float y, float label)> GetAllPoints() => _testCases;
}
