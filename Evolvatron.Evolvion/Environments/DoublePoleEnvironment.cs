namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Double pole balancing environment for CPU evaluation.
/// Physics matches Colonel.Core.Playground.DoublePoleCart exactly (RK4 integration).
///
/// Standard hard benchmark: without velocity (3 inputs, 1 output).
/// State: [cartPos, cartVel, pole1Angle, pole1AngVel, pole2Angle, pole2AngVel]
/// Without velocity observations: [cartPos, pole1Angle, pole2Angle] (normalized)
/// Action: force in [-1, 1]
///
/// Terminal when: cart off track (+-2.4m) or either pole angle > 36 degrees.
/// Solved when: survives 100,000 steps.
/// </summary>
public class DoublePoleEnvironment : IEnvironment
{
    private const float Gravity = -9.8f;
    private const float MassCart = 1.0f;
    private const float Length1 = 0.5f;
    private const float MassPole1 = 0.1f;
    private const float Length2 = 0.05f;
    private const float MassPole2 = 0.01f;
    private const float ForceMag = 10.0f;
    private const float TimeDelta = 0.01f;
    private const float Mup = 0.000002f;
    private const float ML1 = Length1 * MassPole1;
    private const float ML2 = Length2 * MassPole2;

    private const float Pi = 3.14159265358979f;
    private const float FourDegrees = Pi / 45f;
    private const float ThirtySixDegrees = Pi / 5f;

    private readonly float _trackLengthHalf;
    private readonly float _poleAngleThreshold;
    private readonly bool _includeVelocity;

    private float _s0, _s1, _s2, _s3, _s4, _s5; // state
    private int _steps;
    private bool _terminated;
    private readonly float[] _jiggleBuffer = new float[100];

    public int InputCount => _includeVelocity ? 6 : 3;
    public int OutputCount => 1;
    public int MaxSteps { get; }

    public DoublePoleEnvironment(
        bool includeVelocity = false,
        int maxSteps = 100_000,
        float trackLength = 4.8f,
        float poleAngleThresholdDegrees = 36f)
    {
        _includeVelocity = includeVelocity;
        MaxSteps = maxSteps;
        _trackLengthHalf = trackLength / 2f;
        _poleAngleThreshold = poleAngleThresholdDegrees * Pi / 180f;
    }

    public void Reset(int seed = 0)
    {
        _s0 = 0f; // cart position
        _s1 = 0f; // cart velocity
        _s2 = FourDegrees; // pole1 angle (4 degrees, matching Colonel)
        _s3 = 0f; // pole1 angular velocity
        _s4 = 0f; // pole2 angle
        _s5 = 0f; // pole2 angular velocity
        _steps = 0;
        _terminated = false;
        Array.Clear(_jiggleBuffer);
    }

    public void GetObservations(Span<float> observations)
    {
        if (_includeVelocity)
        {
            observations[0] = _s0 / _trackLengthHalf;
            observations[1] = _s1 / 5f;
            observations[2] = _s2 / _poleAngleThreshold;
            observations[3] = _s3 / 5f;
            observations[4] = _s4 / _poleAngleThreshold;
            observations[5] = _s5 / 5f;
        }
        else
        {
            observations[0] = _s0 / _trackLengthHalf;
            observations[1] = _s2 / _poleAngleThreshold;
            observations[2] = _s4 / _poleAngleThreshold;
        }
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        if (_terminated) return 0f;

        float action = Math.Clamp(actions[0], -1f, 1f);

        // 2x RK4 per tick (matching Colonel's Tick)
        PerformRK4(action);
        PerformRK4(action);

        _steps++;

        // Jiggle tracking (Gruau's anti-oscillation metric)
        int jiggleIdx = (_steps - 1) % 100;
        _jiggleBuffer[jiggleIdx] = MathF.Abs(_s0) + MathF.Abs(_s1) + MathF.Abs(_s2) + MathF.Abs(_s3);

        bool outOfBounds =
            _s0 < -_trackLengthHalf || _s0 > _trackLengthHalf ||
            _s2 > _poleAngleThreshold || _s2 < -_poleAngleThreshold ||
            _s4 > _poleAngleThreshold || _s4 < -_poleAngleThreshold;

        if (outOfBounds || _steps >= MaxSteps)
            _terminated = true;

        return 1f;
    }

    public bool IsTerminal() => _terminated;

    public float GetFinalFitness() => _steps;

    private void PerformRK4(float action)
    {
        float hh = TimeDelta * 0.5f;
        float h6 = TimeDelta / 6f;

        float d0 = _s1, d2 = _s3, d4 = _s5;
        ComputeAccelerations(action, _s0, _s1, _s2, _s3, _s4, _s5,
            out float d1, out float d3, out float d5);

        float yt0 = _s0 + hh * d0, yt1 = _s1 + hh * d1, yt2 = _s2 + hh * d2;
        float yt3 = _s3 + hh * d3, yt4 = _s4 + hh * d4, yt5 = _s5 + hh * d5;

        float dyt0 = yt1, dyt2 = yt3, dyt4 = yt5;
        ComputeAccelerations(action, yt0, yt1, yt2, yt3, yt4, yt5,
            out float dyt1, out float dyt3, out float dyt5);

        yt0 = _s0 + hh * dyt0; yt1 = _s1 + hh * dyt1; yt2 = _s2 + hh * dyt2;
        yt3 = _s3 + hh * dyt3; yt4 = _s4 + hh * dyt4; yt5 = _s5 + hh * dyt5;

        float dym0 = yt1, dym2 = yt3, dym4 = yt5;
        ComputeAccelerations(action, yt0, yt1, yt2, yt3, yt4, yt5,
            out float dym1, out float dym3, out float dym5);

        yt0 = _s0 + TimeDelta * dym0; yt1 = _s1 + TimeDelta * dym1; yt2 = _s2 + TimeDelta * dym2;
        yt3 = _s3 + TimeDelta * dym3; yt4 = _s4 + TimeDelta * dym4; yt5 = _s5 + TimeDelta * dym5;
        dym0 += dyt0; dym1 += dyt1; dym2 += dyt2;
        dym3 += dyt3; dym4 += dyt4; dym5 += dyt5;

        dyt0 = yt1; dyt2 = yt3; dyt4 = yt5;
        ComputeAccelerations(action, yt0, yt1, yt2, yt3, yt4, yt5,
            out dyt1, out dyt3, out dyt5);

        _s0 += h6 * (d0 + dyt0 + 2f * dym0);
        _s1 += h6 * (d1 + dyt1 + 2f * dym1);
        _s2 += h6 * (d2 + dyt2 + 2f * dym2);
        _s3 += h6 * (d3 + dyt3 + 2f * dym3);
        _s4 += h6 * (d4 + dyt4 + 2f * dym4);
        _s5 += h6 * (d5 + dyt5 + 2f * dym5);
    }

    private static void ComputeAccelerations(
        float action,
        float s0, float s1, float s2, float s3, float s4, float s5,
        out float cartAcc, out float pole1Acc, out float pole2Acc)
    {
        float force = action * ForceMag;
        float costheta1 = MathF.Cos(s2);
        float sintheta1 = MathF.Sin(s2);
        float gsintheta1 = Gravity * sintheta1;
        float costheta2 = MathF.Cos(s4);
        float sintheta2 = MathF.Sin(s4);
        float gsintheta2 = Gravity * sintheta2;

        float temp1 = Mup * s3 / ML1;
        float temp2 = Mup * s5 / ML2;

        float fi1 = ML1 * s3 * s3 * sintheta1 +
                     0.75f * MassPole1 * costheta1 * (temp1 + gsintheta1);
        float fi2 = ML2 * s5 * s5 * sintheta2 +
                     0.75f * MassPole2 * costheta2 * (temp2 + gsintheta2);

        float mi1 = MassPole1 * (1f - 0.75f * costheta1 * costheta1);
        float mi2 = MassPole2 * (1f - 0.75f * costheta2 * costheta2);

        cartAcc = (force + fi1 + fi2) / (mi1 + mi2 + MassCart);
        pole1Acc = -0.75f * (cartAcc * costheta1 + gsintheta1 + temp1) / Length1;
        pole2Acc = -0.75f * (cartAcc * costheta2 + gsintheta2 + temp2) / Length2;
    }
}
