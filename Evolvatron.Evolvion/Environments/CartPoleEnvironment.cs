namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// CartPole (inverted pendulum) environment for neural network training.
/// Classic control problem: Balance a pole on a cart by applying left/right forces.
///
/// State: [cart_pos, cart_vel, pole_angle, pole_angular_vel] (4D)
/// Action: force in [-1, 1] (continuous)
///
/// Episode terminates when:
/// - Pole angle exceeds ±12 degrees
/// - Cart position exceeds ±2.4m
/// - 1000 steps reached (success!)
/// </summary>
public class CartPoleEnvironment : IEnvironment
{
    // Physics constants (from SinglePoleCart.cs)
    private const float GRAVITY = 9.8f;
    private const float MASSCART = 1.0f;
    private const float MASSPOLE = 0.1f;
    private const float TOTAL_MASS = MASSPOLE + MASSCART;
    private const float LENGTH = 0.5f; // Half the pole's length
    private const float POLEMASS_LENGTH = MASSPOLE * LENGTH;
    private const float FORCE_MAG = 10.0f;
    private const float FOURTHIRDS = 4f / 3f;
    private const float TAU = 0.02f; // Timestep (seconds)
    private const float RAIL_LENGTH = 4.8f;
    private const float HRAIL_LENGTH = RAIL_LENGTH / 2f;
    private const float PI = 3.14159f;
    private const float MAX_ANGLE_DEGREES = 12f;
    private const float MAX_ANGLE_RADIANS = MAX_ANGLE_DEGREES * PI / 180f;
    private const float ANGULAR_FRICTION = 0.0001f;

    // State variables
    private float _cartPosition;
    private float _cartSpeed;
    private float _poleAngle;
    private float _poleAngleSpeed;
    private int _steps;
    private bool _terminated;

    // Reward shaping parameters
    private float _angleWeight = 1.0f;
    private float _positionWeight = 0.1f;
    private float _velocityWeight = 0.01f;

    public int InputCount => 4; // [pos, vel, angle, angular_vel]
    public int OutputCount => 1; // force in [-1, 1]
    public int MaxSteps => 1000; // Episode length

    public void Reset(int seed = 0)
    {
        var random = new Random(seed);

        // Small random initial state (similar to GymInit)
        _cartPosition = (float)(random.NextDouble() * 0.1 - 0.05);
        _cartSpeed = (float)(random.NextDouble() * 0.1 - 0.05);
        _poleAngle = (float)(random.NextDouble() * 0.1 - 0.05);
        _poleAngleSpeed = (float)(random.NextDouble() * 0.1 - 0.05);
        _steps = 0;
        _terminated = false;
    }

    public void GetObservations(Span<float> observations)
    {
        observations[0] = _cartPosition / HRAIL_LENGTH; // Normalize to [-1, 1]
        observations[1] = _cartSpeed / 5f; // Rough normalization
        observations[2] = _poleAngle / MAX_ANGLE_RADIANS; // Normalize to [-1, 1] at threshold
        observations[3] = _poleAngleSpeed / 5f; // Rough normalization
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        if (_terminated)
            return 0f;

        _steps++;

        // Apply action (force in [-1, 1])
        float action = Math.Clamp(actions[0], -1f, 1f);
        float force = action * FORCE_MAG;

        // Physics simulation (from SinglePoleCart.Tick)
        float costheta = MathF.Cos(_poleAngle);
        float sintheta = MathF.Sin(_poleAngle);

        float temp = (force + POLEMASS_LENGTH * _poleAngleSpeed * _poleAngleSpeed * sintheta) / TOTAL_MASS;
        float thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));
        float xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

        // Euler integration
        _cartPosition += TAU * _cartSpeed;
        _cartSpeed += TAU * xacc;
        _poleAngle += TAU * _poleAngleSpeed;
        _poleAngleSpeed += TAU * thetaacc;

        // Apply angular friction
        _poleAngleSpeed -= (ANGULAR_FRICTION * _poleAngleSpeed * _poleAngleSpeed) * MathF.Sign(_poleAngleSpeed);

        // Wrap angle to [-PI, PI]
        while (_poleAngle < -PI)
            _poleAngle += 2 * PI;
        while (_poleAngle > PI)
            _poleAngle -= 2 * PI;

        // Check termination conditions
        bool outOfBounds =
            MathF.Abs(_cartPosition) > HRAIL_LENGTH ||
            MathF.Abs(_poleAngle) > MAX_ANGLE_RADIANS;

        if (outOfBounds)
        {
            _terminated = true;
            // Return cumulative reward earned so far (survival time)
            return 0f;
        }

        // Survived another step - give reward
        // Simple sparse reward: +1 per step survived
        return 1f;
    }

    public bool IsTerminal()
    {
        return _terminated || _steps >= MaxSteps;
    }

    /// <summary>
    /// Configure reward shaping weights.
    /// Higher weights emphasize different aspects of performance.
    /// </summary>
    public void SetRewardWeights(float angleWeight, float positionWeight, float velocityWeight)
    {
        _angleWeight = angleWeight;
        _positionWeight = positionWeight;
        _velocityWeight = velocityWeight;
    }
}
