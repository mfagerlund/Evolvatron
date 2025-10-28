namespace Evolvatron.Evolvion;

/// <summary>
/// Standard interface for fitness evaluation environments.
/// Provides observations to neural networks and returns rewards.
/// </summary>
public interface IEnvironment
{
    /// <summary>
    /// Number of inputs the network receives.
    /// </summary>
    int InputCount { get; }

    /// <summary>
    /// Number of outputs the network produces.
    /// </summary>
    int OutputCount { get; }

    /// <summary>
    /// Maximum number of steps per episode.
    /// </summary>
    int MaxSteps { get; }

    /// <summary>
    /// Reset the environment to initial state with optional seed.
    /// </summary>
    void Reset(int seed = 0);

    /// <summary>
    /// Get current observations for the network.
    /// </summary>
    void GetObservations(Span<float> observations);

    /// <summary>
    /// Step the environment with network actions, return reward.
    /// </summary>
    float Step(ReadOnlySpan<float> actions);

    /// <summary>
    /// Check if the episode is complete.
    /// </summary>
    bool IsTerminal();

    /// <summary>
    /// Get final fitness based on terminal state (optional, defaults to cumulative reward).
    /// Return 0 to use cumulative reward instead.
    /// </summary>
    float GetFinalFitness() => 0f;
}
