namespace Evolvatron.Core;

/// <summary>
/// Distance (rod) constraint: maintains fixed distance between two particles.
/// C = |p_i - p_j| - RestLength
/// </summary>
public struct Rod
{
    /// <summary>Index of first particle.</summary>
    public int I;

    /// <summary>Index of second particle.</summary>
    public int J;

    /// <summary>Rest (target) length in meters.</summary>
    public float RestLength;

    /// <summary>Constraint compliance (0 = rigid). α = compliance / dt²</summary>
    public float Compliance;

    /// <summary>Lagrange multiplier (accumulated within step, reset each step).</summary>
    public float Lambda;

    public Rod(int i, int j, float restLength, float compliance = 0f)
    {
        I = i;
        J = j;
        RestLength = restLength;
        Compliance = compliance;
        Lambda = 0f;
    }
}

/// <summary>
/// Angle constraint: maintains angle at middle particle j between i-j-k.
/// C = angle(i-j-k) - Theta0
/// </summary>
public struct Angle
{
    /// <summary>Index of first particle (forms one edge from j).</summary>
    public int I;

    /// <summary>Index of middle particle (vertex of angle).</summary>
    public int J;

    /// <summary>Index of third particle (forms second edge from j).</summary>
    public int K;

    /// <summary>Target angle in radians.</summary>
    public float Theta0;

    /// <summary>Constraint compliance (0 = rigid).</summary>
    public float Compliance;

    /// <summary>Lagrange multiplier (accumulated within step).</summary>
    public float Lambda;

    public Angle(int i, int j, int k, float theta0, float compliance = 0f)
    {
        I = i;
        J = j;
        K = k;
        Theta0 = theta0;
        Compliance = compliance;
        Lambda = 0f;
    }
}

/// <summary>
/// Motorized angle constraint: servo that drives angle i-j-k toward a target.
/// Target can be updated each step for active control (e.g., gimbal, throttle).
/// </summary>
public struct MotorAngle
{
    /// <summary>Index of first particle.</summary>
    public int I;

    /// <summary>Index of middle particle (vertex).</summary>
    public int J;

    /// <summary>Index of third particle.</summary>
    public int K;

    /// <summary>Target angle in radians (updated by controller).</summary>
    public float Target;

    /// <summary>Constraint compliance (small for stiff servo, e.g., 1e-6).</summary>
    public float Compliance;

    /// <summary>Lagrange multiplier (accumulated within step).</summary>
    public float Lambda;

    public MotorAngle(int i, int j, int k, float target, float compliance = 1e-6f)
    {
        I = i;
        J = j;
        K = k;
        Target = target;
        Compliance = compliance;
        Lambda = 0f;
    }
}
