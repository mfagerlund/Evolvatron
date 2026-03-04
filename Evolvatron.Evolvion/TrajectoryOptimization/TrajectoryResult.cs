namespace Evolvatron.Evolvion.TrajectoryOptimization;

/// <summary>
/// Result of trajectory optimization: the optimized control sequence and recorded states.
/// </summary>
public sealed class TrajectoryResult
{
    /// <summary>Raw optimizer parameters [2*N]: pairs of (throttle, gimbal).</summary>
    public double[] Controls = Array.Empty<double>();

    /// <summary>Clamped throttle per control step [N].</summary>
    public float[] Throttles = Array.Empty<float>();

    /// <summary>Clamped gimbal per control step [N].</summary>
    public float[] Gimbals = Array.Empty<float>();

    /// <summary>State at each control boundary [N+1] (initial + after each step).</summary>
    public TrajectoryState[] States = Array.Empty<TrajectoryState>();

    /// <summary>Trajectory states at each LM iteration [iterCount][N+1]. Shows optimization progress.</summary>
    public List<IterationSnapshot> IterationSnapshots = new();

    public bool Success;
    public double FinalCost;
    public int Iterations;
    public double ComputationTimeMs;
    public string ConvergenceReason = "";
}

/// <summary>
/// Snapshot of the full trajectory at one LM iteration.
/// </summary>
public sealed class IterationSnapshot
{
    public int Iteration;
    public double Cost;
    public TrajectoryState[] States = Array.Empty<TrajectoryState>();
}

/// <summary>
/// Rocket state at a control boundary during trajectory playback.
/// </summary>
public struct TrajectoryState
{
    public float X, Y;
    public float VelX, VelY;
    public float Angle, AngularVel;
    public float Throttle, Gimbal;
}
