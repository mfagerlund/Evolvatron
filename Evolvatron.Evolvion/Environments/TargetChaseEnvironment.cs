using Evolvatron.Core;

namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Simple spherical rocket chasing targets in 2D space.
/// No rotation/orientation - just direct X/Y thrust control.
///
/// Observations (4D): [dx, dy, vx, vy]
/// Actions (2D): [thrust_x, thrust_y] in [-1, 1]
///
/// Scoring: Hit targets for points, avoid going out of bounds.
/// </summary>
public class TargetChaseEnvironment : IEnvironment
{
    private readonly SimulationConfig _config;
    private Random _random;

    // Rocket state (simple point mass)
    private float _rocketX;
    private float _rocketY;
    private float _rocketVelX;
    private float _rocketVelY;

    // Target state
    private float _targetX;
    private float _targetY;
    private const float TargetRadius = 1.5f;  // Match GPU

    // Arena bounds (symmetric box: -10 to 10)
    public const float ArenaHalfSize = 10f;

    // Episode state
    private int _steps;
    private int _targetsHit;
    private float _cumulativeReward;
    private bool _terminated;

    // Physics parameters
    private readonly float _maxThrust;

    public int InputCount => 4; // dx, dy, vx, vy
    public int OutputCount => 2; // thrust_x, thrust_y
    public int MaxSteps { get; set; } = 300;

    /// <summary>
    /// Gets the number of targets hit in the current episode.
    /// </summary>
    public int TargetsHit => _targetsHit;

    /// <summary>
    /// Gets current target position for visualization.
    /// </summary>
    public (float X, float Y) TargetPosition => (_targetX, _targetY);

    /// <summary>
    /// Gets current throttle for visualization (magnitude of thrust).
    /// </summary>
    public float CurrentThrottle => 0f; // Not applicable for spherical

    public TargetChaseEnvironment(float maxThrust = 30f)
    {
        _maxThrust = maxThrust;
        _random = new Random();

        _config = new SimulationConfig
        {
            Dt = 1f / 60f,
            GravityX = 0f,
            GravityY = 0f,            // No gravity
            GlobalDamping = 0.05f,    // Some velocity damping
        };
    }

    public void Reset(int seed = 0)
    {
        _random = new Random(seed);

        // Spawn rocket at center
        _rocketX = 0f;
        _rocketY = 0f;
        _rocketVelX = 0f;
        _rocketVelY = 0f;

        // Spawn first target
        SpawnNewTarget();

        // Reset episode state
        _steps = 0;
        _targetsHit = 0;
        _cumulativeReward = 0f;
        _terminated = false;
    }

    private void SpawnNewTarget()
    {
        // Spawn target at random position within arena with margin
        float margin = 2f;
        _targetX = (float)(_random.NextDouble() * (ArenaHalfSize * 2 - margin * 2) - ArenaHalfSize + margin);
        _targetY = (float)(_random.NextDouble() * (ArenaHalfSize * 2 - margin * 2) - ArenaHalfSize + margin);
    }

    public void GetObservations(Span<float> observations)
    {
        // Delta to target (normalized to arena size)
        float dx = (_targetX - _rocketX) / 20f;
        float dy = (_targetY - _rocketY) / 20f;

        // Simple observations: dx, dy, vx, vy
        observations[0] = dx;
        observations[1] = dy;
        observations[2] = _rocketVelX / 15f;
        observations[3] = _rocketVelY / 15f;
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        if (_terminated)
            return 0f;

        _steps++;

        // Parse actions - direct X/Y thrust
        float thrustX = Math.Clamp(actions[0], -1f, 1f);
        float thrustY = Math.Clamp(actions[1], -1f, 1f);

        // Apply thrust
        float dt = _config.Dt;
        _rocketVelX += thrustX * _maxThrust * dt;
        _rocketVelY += thrustY * _maxThrust * dt;

        // Apply damping
        float damping = 1f - _config.GlobalDamping;
        _rocketVelX *= damping;
        _rocketVelY *= damping;

        // Clamp velocity
        float maxVel = 15f;
        float speed = MathF.Sqrt(_rocketVelX * _rocketVelX + _rocketVelY * _rocketVelY);
        if (speed > maxVel)
        {
            _rocketVelX *= maxVel / speed;
            _rocketVelY *= maxVel / speed;
        }

        // Update position
        _rocketX += _rocketVelX * dt;
        _rocketY += _rocketVelY * dt;

        float stepReward = 0f;

        // Check target collision
        float dx = _targetX - _rocketX;
        float dy = _targetY - _rocketY;
        float distToTarget = MathF.Sqrt(dx * dx + dy * dy);

        if (distToTarget < TargetRadius) // Hit target!
        {
            _targetsHit++;
            stepReward += 100f; // Match GPU
            SpawnNewTarget();
        }
        else
        {
            // Simple proximity-based reward (match GPU)
            float proximityReward = (14f - distToTarget) * 0.1f;
            stepReward += proximityReward;
        }

        // Time penalty (match GPU)
        stepReward -= 0.1f;

        // Check terminal conditions - out of bounds
        if (MathF.Abs(_rocketX) > ArenaHalfSize || MathF.Abs(_rocketY) > ArenaHalfSize)
        {
            _terminated = true;
            stepReward -= 50f;
        }

        // Max steps
        if (_steps >= MaxSteps)
        {
            _terminated = true;
        }

        _cumulativeReward += stepReward;
        return stepReward;
    }

    public bool IsTerminal() => _terminated;

    public float GetFinalFitness()
    {
        return _cumulativeReward;
    }

    /// <summary>
    /// Gets the current rocket state for visualization.
    /// </summary>
    public void GetRocketState(out float x, out float y, out float vx, out float vy, out float upX, out float upY, out float angle)
    {
        x = _rocketX;
        y = _rocketY;
        vx = _rocketVelX;
        vy = _rocketVelY;
        upX = 0f;
        upY = 1f;
        angle = 0f;
    }

    /// <summary>
    /// Gets the underlying WorldState for visualization (empty for spherical rocket).
    /// </summary>
    public WorldState GetWorld() => new WorldState(1);
}
