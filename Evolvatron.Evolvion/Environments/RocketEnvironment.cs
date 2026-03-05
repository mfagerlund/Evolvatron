using Evolvatron.Core;
using Evolvatron.Core.Templates;

namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Rocket landing environment that integrates Rigidon physics with Evolvion neural evolution.
/// Uses rigid body rockets with physics-based simulation for RL training.
///
/// Observations (8D): [relPosX, relPosY, velX, velY, upX, upY, gimbal, throttle]
/// Actions (2D): [throttle (0-1), gimbal (-1 to 1)]
///
/// Episode terminates when:
/// - Rocket lands successfully (low velocity, upright, near pad)
/// - Rocket crashes (high velocity, flipped, out of bounds)
/// - Max steps reached
/// </summary>
public class RocketEnvironment : IEnvironment
{
    // Pre-allocated state (reused across episodes)
    private readonly WorldState _world;
    private readonly SimulationConfig _config;
    private readonly CPUStepper _stepper;
    private readonly float[] _observations;

    // Rocket state
    private int[] _rocketIndices = Array.Empty<int>();
    private RewardParams _rewardParams;

    // Episode state
    private float _currentThrottle;
    private float _currentGimbal;
    private float _prevThrottle;
    private float _prevGimbal;
    private int _steps;
    private bool _terminated;
    private bool _landed;
    private float _cumulativeReward;

    // Physics parameters
    private readonly float _maxThrust;
    private readonly float _maxGimbalTorque;
    private readonly bool _singleBody;

    // Scene parameters
    private readonly float _spawnHeight;
    private readonly float _groundY;

    public int InputCount => 8; // RocketObservation dimensions
    public int OutputCount => 2; // throttle, gimbal
    public int MaxSteps { get; set; } = 500;

    // Difficulty knobs (defaults match original easy settings)
    public float SpawnXRange { get; set; } = 2f;         // max distance from pad center
    public float SpawnXMin { get; set; } = 0f;           // min distance from pad (0 = any)
    public float SpawnAngleRange { get; set; } = 0f;     // no initial tilt (radians)
    public float InitialVelXRange { get; set; } = 1f;    // ±1 m/s horizontal
    public float InitialVelYMax { get; set; } = 2f;      // 0 to -2 m/s (downward)

    // Rendering properties
    public float CurrentThrottle => _currentThrottle;
    public float CurrentGimbal => _currentGimbal;
    public float GroundSurfaceY => _groundY + 0.5f;
    public float PadX => _rewardParams.PadX;
    public float PadY => _rewardParams.PadY;
    public float PadHalfWidth => _rewardParams.PadHalfWidth;
    public WorldState World => _world;
    public int[] RocketIndices => _rocketIndices;

    /// <summary>
    /// Creates a rocket environment with default parameters.
    /// </summary>
    /// <param name="singleBody">When true, uses a single capsule body matching GPU physics exactly.
    /// When false, uses 3-body rocket (body + 2 legs with joints).</param>
    public RocketEnvironment(
        float maxThrust = 200f,
        float maxGimbalTorque = 50f,
        float spawnHeight = 15f,
        float groundY = -5f,
        bool singleBody = false)
    {
        _maxThrust = maxThrust;
        _maxGimbalTorque = maxGimbalTorque;
        _spawnHeight = spawnHeight;
        _groundY = groundY;
        _singleBody = singleBody;

        // Pre-allocate world and config (reused across episodes)
        _world = new WorldState(64); // Small initial capacity, will grow if needed
        _config = new SimulationConfig
        {
            Dt = 1f / 120f, // 120 Hz physics for faster simulation
            XpbdIterations = 8,
            Substeps = 1,
            GravityY = -9.81f,
            FrictionMu = 0.8f,
            Restitution = 0.0f, // No bounce for landing
            GlobalDamping = 0.02f,
            AngularDamping = 0.1f
        };

        _stepper = new CPUStepper();
        _observations = new float[InputCount];

        _rewardParams = RewardParams.Default();
        _rewardParams.PadY = groundY + 0.5f; // Slightly above ground
    }

    /// <summary>
    /// Resets the environment to initial state for a new episode.
    /// Efficiently reuses WorldState by clearing and reinitializing.
    /// </summary>
    public void Reset(int seed = 0)
    {
        var random = new Random(seed);

        // Clear world state (reuses allocated arrays)
        _world.Clear();

        // Add ground collider
        _world.Obbs.Add(OBBCollider.AxisAligned(0f, _groundY, 30f, 0.5f));

        // Random spawn position
        float spawnX;
        if (SpawnXMin > 0f)
        {
            float side = random.NextDouble() < 0.5 ? -1f : 1f;
            spawnX = side * (SpawnXMin + (float)(random.NextDouble() * (SpawnXRange - SpawnXMin)));
        }
        else
        {
            spawnX = (float)(random.NextDouble() * SpawnXRange * 2 - SpawnXRange);
        }
        float spawnY = _spawnHeight + (float)(random.NextDouble() * 3f);
        float spawnAngle = (float)(random.NextDouble() * SpawnAngleRange * 2 - SpawnAngleRange);

        // Random initial velocity
        float initialVelX = (float)(random.NextDouble() * InitialVelXRange * 2 - InitialVelXRange);
        float initialVelY = (float)(random.NextDouble() * -InitialVelYMax);

        if (_singleBody)
        {
            // Single capsule body matching GPU physics exactly
            // Total mass = body(8) + legs(1.5+1.5) = 11 kg
            float bodyHeight = 1.5f;
            float bodyRadius = 0.2f;
            float halfLength = bodyHeight * 0.5f;
            float centerY = spawnY + halfLength; // Match GPU: baseY = spawnHeight + halfLength

            int bodyIdx = RigidBodyFactory.CreateCapsule(
                _world, spawnX, centerY, halfLength, bodyRadius,
                mass: 11f, angle: MathF.PI / 2f + spawnAngle);
            _rocketIndices = new[] { bodyIdx };

            var rb = _world.RigidBodies[bodyIdx];
            rb.VelX = initialVelX;
            rb.VelY = initialVelY;
            _world.RigidBodies[bodyIdx] = rb;
        }
        else
        {
            // Full 3-body rocket with legs and joints
            _rocketIndices = RigidBodyRocketTemplate.CreateRocket(
                _world,
                centerX: spawnX,
                centerY: spawnY,
                bodyHeight: 1.5f,
                bodyRadius: 0.2f,
                legLength: 1.0f,
                legRadius: 0.1f,
                bodyMass: 8f,
                legMass: 1.5f);

            // Apply initial tilt by rotating all bodies around spawn base point
            float cosT = MathF.Cos(spawnAngle);
            float sinT = MathF.Sin(spawnAngle);
            for (int i = 0; i < _rocketIndices.Length; i++)
            {
                var rb = _world.RigidBodies[_rocketIndices[i]];
                float dx = rb.X - spawnX;
                float dy = rb.Y - spawnY;
                rb.X = spawnX + dx * cosT - dy * sinT;
                rb.Y = spawnY + dx * sinT + dy * cosT;
                rb.Angle += spawnAngle;
                rb.VelX = initialVelX;
                rb.VelY = initialVelY;
                _world.RigidBodies[_rocketIndices[i]] = rb;
            }
        }

        // Reset episode state
        _currentThrottle = 0f;
        _currentGimbal = 0f;
        _prevThrottle = 0f;
        _prevGimbal = 0f;
        _steps = 0;
        _terminated = false;
        _landed = false;
        _cumulativeReward = 0f;
    }

    /// <summary>
    /// Gets current observations for the neural network.
    /// </summary>
    public void GetObservations(Span<float> observations)
    {
        if (_rocketIndices.Length == 0)
        {
            observations.Fill(0f);
            return;
        }

        // Get rocket state using rigid body template methods
        RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out float comX, out float comY);
        RigidBodyRocketTemplate.GetVelocity(_world, _rocketIndices, out float velX, out float velY);
        RigidBodyRocketTemplate.GetUpVector(_world, _rocketIndices, out float upX, out float upY);

        // Normalize observations for neural network input
        // Position relative to pad, normalized by typical range
        observations[0] = (comX - _rewardParams.PadX) / 20f;
        observations[1] = (comY - _rewardParams.PadY) / 20f;

        // Velocity normalized
        observations[2] = velX / 10f;
        observations[3] = velY / 10f;

        // Up vector (already normalized)
        observations[4] = upX;
        observations[5] = upY;

        // Current control inputs
        observations[6] = _currentGimbal;
        observations[7] = _currentThrottle;
    }

    /// <summary>
    /// Steps the environment with neural network actions.
    /// </summary>
    /// <param name="actions">Actions: [throttle (clamped to 0-1), gimbal (clamped to -1,1)]</param>
    /// <returns>Step reward</returns>
    public float Step(ReadOnlySpan<float> actions)
    {
        if (_terminated)
            return 0f;

        _steps++;

        // Store previous controls for reward computation
        _prevThrottle = _currentThrottle;
        _prevGimbal = _currentGimbal;

        // Parse and clamp actions
        _currentThrottle = Math.Clamp(actions[0], 0f, 1f);
        _currentGimbal = Math.Clamp(actions[1], -1f, 1f);

        // Apply controls to rocket (pass dt for correct force application)
        RigidBodyRocketTemplate.ApplyThrust(_world, _rocketIndices, _currentThrottle, _maxThrust, _config.Dt);
        RigidBodyRocketTemplate.ApplyGimbal(_world, _rocketIndices, _currentGimbal * _maxGimbalTorque, _config.Dt);

        // Step physics
        _stepper.Step(_world, _config);

        // Compute reward using rigid body state
        float stepReward = ComputeReward(out bool terminal, out float terminalReward);

        if (terminal)
        {
            _terminated = true;
            stepReward += terminalReward;
        }

        // Check max steps
        if (_steps >= MaxSteps)
        {
            _terminated = true;
        }

        _cumulativeReward += stepReward;
        return stepReward;
    }

    /// <summary>
    /// Checks if the episode is complete.
    /// </summary>
    public bool IsTerminal()
    {
        return _terminated;
    }

    /// <summary>
    /// Returns terminal-state fitness for evolutionary optimization.
    /// Evaluates how good the final state is rather than accumulating per-step penalties.
    /// This prevents the "die quickly" degenerate strategy that cumulative rewards create.
    /// </summary>
    public float GetFinalFitness()
    {
        if (_rocketIndices.Length == 0)
            return -100f;

        RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out float comX, out float comY);
        RigidBodyRocketTemplate.GetVelocity(_world, _rocketIndices, out float velX, out float velY);
        RigidBodyRocketTemplate.GetUpVector(_world, _rocketIndices, out float upX, out float upY);

        float errX = comX - _rewardParams.PadX;
        float errY = comY - _rewardParams.PadY;
        float distToPad = MathF.Sqrt(errX * errX + errY * errY);
        float speed = MathF.Sqrt(velX * velX + velY * velY);
        float angleErr = MathF.Abs(MathF.Atan2(upX, upY));
        float survivalFrac = (float)_steps / MaxSteps;

        // Terminal-state fitness components:
        // Survival bonus: staying alive is always better (max +20)
        float fitness = 20f * survivalFrac;

        // Closeness bonus: being near the pad at termination (max +20)
        const float maxDist = 30f;
        fitness += 20f * MathF.Max(0f, 1f - distToPad / maxDist);

        // Velocity penalty: high terminal speed is bad (max -5)
        const float maxSpeed = 20f;
        fitness -= 5f * MathF.Min(1f, speed / maxSpeed);

        // Angle penalty: being tilted at termination (max -5)
        fitness -= 5f * MathF.Min(1f, angleErr / MathF.PI);

        // Landing bonus: dominates everything
        if (_landed)
            fitness += 200f;

        return fitness;
    }

    public bool Landed => _landed;

    /// <summary>
    /// Computes step reward and terminal conditions using rigid body state.
    /// Based on RewardModel but adapted for rigid body rockets.
    /// </summary>
    private float ComputeReward(out bool terminal, out float terminalReward)
    {
        terminal = false;
        terminalReward = 0f;

        // Get rocket state
        RigidBodyRocketTemplate.GetCenterOfMass(_world, _rocketIndices, out float comX, out float comY);
        RigidBodyRocketTemplate.GetVelocity(_world, _rocketIndices, out float velX, out float velY);
        RigidBodyRocketTemplate.GetUpVector(_world, _rocketIndices, out float upX, out float upY);

        // Position error relative to pad
        float errX = comX - _rewardParams.PadX;
        float errY = comY - _rewardParams.PadY;
        float positionError = MathF.Sqrt(errX * errX + errY * errY);

        // Angle error (want upright: ux=0, uy=1)
        float angleErr = MathF.Abs(MathF.Atan2(upX, upY));

        // Control effort (penalize large changes)
        float dThrottle = _currentThrottle - _prevThrottle;
        float dGimbal = _currentGimbal - _prevGimbal;
        float controlPenalty = dThrottle * dThrottle + dGimbal * dGimbal;

        // Step reward (negative penalties + alive bonus)
        float stepReward = -_rewardParams.K_Position * positionError
                         - _rewardParams.K_VelocityX * MathF.Abs(velX)
                         - _rewardParams.K_VelocityY * MathF.Abs(velY)
                         - _rewardParams.K_Angle * angleErr * angleErr
                         - _rewardParams.K_Control * controlPenalty
                         + _rewardParams.K_Alive;

        // Terminal conditions

        // 1. Success: inside pad zone, low velocity, upright
        bool nearPad = MathF.Abs(errX) < _rewardParams.PadHalfWidth &&
                       MathF.Abs(errY) < 2f;
        bool lowVelocity = MathF.Abs(velX) < _rewardParams.MaxLandingVelocity &&
                           MathF.Abs(velY) < _rewardParams.MaxLandingVelocity;
        bool upright = angleErr < _rewardParams.MaxLandingAngle;

        if (nearPad && lowVelocity && upright)
        {
            terminal = true;
            _landed = true;
            terminalReward = _rewardParams.R_Land;
            return stepReward;
        }

        // 2. Crash: high impact velocity or extreme angle
        bool highImpact = MathF.Abs(velY) > 15f || MathF.Abs(velX) > 10f;
        bool flipped = angleErr > MathF.PI * 0.4f;

        if (highImpact || flipped)
        {
            terminal = true;
            terminalReward = _rewardParams.R_Crash;
            return stepReward;
        }

        // 3. Out of bounds
        if (positionError > 50f || comY < _groundY - 10f || comY > _spawnHeight + 30f)
        {
            terminal = true;
            terminalReward = _rewardParams.R_Crash;
            return stepReward;
        }

        return stepReward;
    }

    /// <summary>
    /// Gets the current rocket state for debugging/visualization.
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
    /// Configure reward parameters for different training objectives.
    /// </summary>
    public void SetRewardParams(RewardParams rparams)
    {
        _rewardParams = rparams;
    }
}
