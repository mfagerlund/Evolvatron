using Evolvatron.Core;
using Evolvatron.Core.Templates;

namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Simple environment where a rocket chases targets in 2D space.
/// Fast-paced gameplay suitable for evolution visualization.
///
/// Observations (6D): [targetDirX, targetDirY, velX, velY, upX, upY]
/// Actions (2D): [throttle (0-1), gimbal (-1 to 1)]
///
/// Scoring: Hit targets for points, avoid crashing or going out of bounds.
/// </summary>
public class TargetChaseEnvironment : IEnvironment
{
    // Pre-allocated state
    private readonly WorldState _world;
    private readonly SimulationConfig _config;
    private readonly CPUStepper _stepper;
    private Random _random;

    // Rocket state
    private int[] _rocketIndices = Array.Empty<int>();

    // Target state
    private float _targetX;
    private float _targetY;
    private const float TargetRadius = 1.0f;

    // Arena bounds
    public const float ArenaHalfWidth = 12f;
    public const float ArenaHalfHeight = 10f;
    public const float GroundY = -8f;

    // Episode state
    private int _steps;
    private int _targetsHit;
    private float _cumulativeReward;
    private bool _terminated;
    private float _lastThrottle;

    // Physics parameters
    private readonly float _maxThrust;
    private readonly float _maxGimbalTorque;

    public int InputCount => 8; // dir(2) + vel(2) + up(2) + dist + angVel
    public int OutputCount => 2;
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
    /// Gets current throttle for visualization.
    /// </summary>
    public float CurrentThrottle => _lastThrottle;

    public TargetChaseEnvironment(
        float maxThrust = 500f,       // Much stronger thrust!
        float maxGimbalTorque = 150f) // Stronger gimbal
    {
        _maxThrust = maxThrust;
        _maxGimbalTorque = maxGimbalTorque;
        _random = new Random();

        _world = new WorldState(32);
        _config = new SimulationConfig
        {
            Dt = 1f / 120f,
            XpbdIterations = 6,
            Substeps = 1,
            GravityY = -4f, // Even lighter gravity for agile movement
            FrictionMu = 0.8f,
            Restitution = 0.1f,
            GlobalDamping = 0.01f,
            AngularDamping = 0.1f
        };

        _stepper = new CPUStepper();
    }

    public void Reset(int seed = 0)
    {
        _random = new Random(seed);

        _world.Clear();

        // Ground
        _world.Obbs.Add(OBBCollider.AxisAligned(0f, GroundY, ArenaHalfWidth + 2f, 0.5f));

        // Side walls
        _world.Obbs.Add(OBBCollider.AxisAligned(-ArenaHalfWidth - 0.5f, 0f, 0.5f, ArenaHalfHeight + 5f));
        _world.Obbs.Add(OBBCollider.AxisAligned(ArenaHalfWidth + 0.5f, 0f, 0.5f, ArenaHalfHeight + 5f));

        // Ceiling
        _world.Obbs.Add(OBBCollider.AxisAligned(0f, ArenaHalfHeight + 0.5f, ArenaHalfWidth + 2f, 0.5f));

        // Spawn rocket in center area
        float spawnX = (float)(_random.NextDouble() * 4f - 2f);
        float spawnY = 0f;

        // Lighter, smaller rocket for more agility
        _rocketIndices = RigidBodyRocketTemplate.CreateRocket(
            _world,
            centerX: spawnX,
            centerY: spawnY,
            bodyHeight: 1.0f,
            bodyRadius: 0.12f,
            legLength: 0.6f,
            legRadius: 0.06f,
            bodyMass: 3f,   // Lighter!
            legMass: 0.5f);

        // Spawn first target
        SpawnNewTarget();

        // Reset episode state
        _steps = 0;
        _targetsHit = 0;
        _cumulativeReward = 0f;
        _terminated = false;
        _lastThrottle = 0f;
    }

    private void SpawnNewTarget()
    {
        // Spawn target at random position, avoiding rocket's current position
        float rocketX = 0f, rocketY = 0f;
        if (_rocketIndices.Length > 0)
        {
            RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out rocketX, out rocketY);
        }

        // Keep trying until we find a position far enough from rocket
        for (int attempt = 0; attempt < 10; attempt++)
        {
            _targetX = (float)(_random.NextDouble() * (ArenaHalfWidth * 2 - 4) - ArenaHalfWidth + 2);
            _targetY = (float)(_random.NextDouble() * (ArenaHalfHeight + GroundY - 2) + GroundY + 2);

            float dx = _targetX - rocketX;
            float dy = _targetY - rocketY;
            float dist = MathF.Sqrt(dx * dx + dy * dy);

            if (dist > 5f) break; // Good distance from rocket
        }
    }

    public void GetObservations(Span<float> observations)
    {
        if (_rocketIndices.Length == 0)
        {
            observations.Fill(0f);
            return;
        }

        RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out float comX, out float comY);
        RigidBodyRocketTemplate.GetVelocity(_world, _rocketIndices, out float velX, out float velY);
        RigidBodyRocketTemplate.GetUpVector(_world, _rocketIndices, out float upX, out float upY);

        // Get angular velocity from main body
        float angVel = _world.RigidBodies[_rocketIndices[0]].AngularVel;

        // Direction to target (normalized)
        float dx = _targetX - comX;
        float dy = _targetY - comY;
        float dist = MathF.Sqrt(dx * dx + dy * dy);
        if (dist > 0.001f)
        {
            dx /= dist;
            dy /= dist;
        }

        // Observations: target direction, velocity, up vector, distance, angular velocity
        observations[0] = dx;                    // Target dir X
        observations[1] = dy;                    // Target dir Y
        observations[2] = velX / 15f;            // Velocity X (normalized)
        observations[3] = velY / 15f;            // Velocity Y (normalized)
        observations[4] = upX;                   // Up vector X (orientation)
        observations[5] = upY;                   // Up vector Y (orientation)
        observations[6] = dist / 20f;            // Distance to target (normalized)
        observations[7] = angVel / 10f;          // Angular velocity (normalized)
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        if (_terminated)
            return 0f;

        _steps++;

        // Parse actions - scale throttle to be more useful
        float throttle = Math.Clamp((actions[0] + 1f) * 0.5f, 0f, 1f); // Map [-1,1] to [0,1]
        float gimbal = Math.Clamp(actions[1], -1f, 1f);
        _lastThrottle = throttle;

        // Apply controls
        RigidBodyRocketTemplate.ApplyThrust(_world, _rocketIndices, throttle, _maxThrust, _config.Dt);
        RigidBodyRocketTemplate.ApplyGimbal(_world, _rocketIndices, gimbal * _maxGimbalTorque, _config.Dt);

        // Step physics
        _stepper.Step(_world, _config);

        // Get rocket state
        RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out float comX, out float comY);
        RigidBodyRocketTemplate.GetVelocity(_world, _rocketIndices, out float velX, out float velY);
        RigidBodyRocketTemplate.GetUpVector(_world, _rocketIndices, out float upX, out float upY);

        float stepReward = 0f;

        // Check target collision
        float dx = _targetX - comX;
        float dy = _targetY - comY;
        float distToTarget = MathF.Sqrt(dx * dx + dy * dy);

        if (distToTarget < TargetRadius + 0.5f) // Hit target!
        {
            _targetsHit++;
            stepReward += 200f; // Big bonus
            SpawnNewTarget();
        }
        else
        {
            // Reward for getting closer (strong shaping)
            stepReward += (20f - distToTarget) * 0.1f;

            // Reward for pointing toward target
            float dotProduct = dx / distToTarget * upX + dy / distToTarget * upY;
            stepReward += dotProduct * 0.5f;
        }

        // Small time penalty
        stepReward -= 0.05f;

        // Check terminal conditions
        float speed = MathF.Sqrt(velX * velX + velY * velY);
        float angleErr = MathF.Abs(MathF.Atan2(upX, upY));

        // Out of bounds
        if (MathF.Abs(comX) > ArenaHalfWidth || comY < GroundY - 1f || comY > ArenaHalfHeight + 2f)
        {
            _terminated = true;
            stepReward -= 100f;
        }

        // Crashed (flipped over)
        if (angleErr > MathF.PI * 0.7f)
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
        // Fitness = targets hit * big bonus + cumulative reward
        return _targetsHit * 200f + _cumulativeReward;
    }

    /// <summary>
    /// Gets the current rocket state for visualization.
    /// </summary>
    public void GetRocketState(out float x, out float y, out float vx, out float vy, out float upX, out float upY, out float angle)
    {
        if (_rocketIndices.Length == 0)
        {
            x = y = vx = vy = upX = upY = angle = 0f;
            return;
        }

        RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out x, out y);
        RigidBodyRocketTemplate.GetVelocity(_world, _rocketIndices, out vx, out vy);
        RigidBodyRocketTemplate.GetUpVector(_world, _rocketIndices, out upX, out upY);
        angle = _world.RigidBodies[_rocketIndices[0]].Angle;
    }

    /// <summary>
    /// Gets the underlying WorldState for visualization.
    /// </summary>
    public WorldState GetWorld() => _world;
}
