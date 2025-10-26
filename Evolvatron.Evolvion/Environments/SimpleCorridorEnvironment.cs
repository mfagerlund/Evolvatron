using System.Numerics;

namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Simplified "Follow the Corridor" environment for evolutionary training.
/// Navigate a 2D car through a procedurally-generated winding corridor.
///
/// This is a simplified version of Colonel.Tests FollowTheCorridor that:
/// - Uses procedural track generation (sine wave)
/// - Has no external dependencies (no Godot, no SVG)
/// - Implements simple raycast collision detection
///
/// State: 9 distance sensors (raycasts at different angles)
/// Action: [steering, throttle] both in [-1, 1]
///
/// Episode terminates when:
/// - Car hits a wall (failure, reward = 0)
/// - Car reaches all checkpoints (success, reward = 1.0+)
/// - MaxSteps reached (timeout)
/// </summary>
public class SimpleCorridorEnvironment : IEnvironment
{
    private const float CAR_RADIUS = 2f;
    private const float MAX_SPEED = 10f;
    private const float MAX_SENSOR_RANGE = 50f;
    private const float CORRIDOR_WIDTH = 15f;
    private const float CHECKPOINT_RADIUS = 5f;

    // Track geometry
    private List<(Vector2 leftStart, Vector2 leftEnd, Vector2 rightStart, Vector2 rightEnd)> _wallSegments = new();
    private List<Vector2> _checkpoints = new();

    // Car state
    private Vector2 _position;
    private float _heading; // radians
    private float _speed;
    private int _checkpointIndex;
    private int _step;
    private bool _crashed;

    // Sensor configuration: 9 sensors at different angles
    private static readonly float[] SensorAngles = { -60, -45, -30, -15, 0, 15, 30, 45, 60 };

    public int InputCount => 9; // 9 distance sensors
    public int OutputCount => 2; // steering + throttle
    public int MaxSteps => 320;

    public void Reset(int seed = 0)
    {
        GenerateProceduralTrack(seed);
        _position = new Vector2(0, 0);
        _heading = 0; // Start facing right
        _speed = 0;
        _checkpointIndex = 0;
        _step = 0;
        _crashed = false;
    }

    private void GenerateProceduralTrack(int seed)
    {
        var random = new Random(seed);
        _wallSegments.Clear();
        _checkpoints.Clear();

        // Generate sine wave corridor
        int segmentCount = 40;
        float segmentLength = 5f;

        for (int i = 0; i < segmentCount; i++)
        {
            float x1 = i * segmentLength;
            float x2 = (i + 1) * segmentLength;

            // Sine wave with some randomness
            float y1 = 30f * MathF.Sin(x1 / 20f) + (float)(random.NextDouble() - 0.5) * 5f;
            float y2 = 30f * MathF.Sin(x2 / 20f) + (float)(random.NextDouble() - 0.5) * 5f;

            // Create wall segments (left and right)
            _wallSegments.Add((
                leftStart: new Vector2(x1, y1 - CORRIDOR_WIDTH),
                leftEnd: new Vector2(x2, y2 - CORRIDOR_WIDTH),
                rightStart: new Vector2(x1, y1 + CORRIDOR_WIDTH),
                rightEnd: new Vector2(x2, y2 + CORRIDOR_WIDTH)
            ));

            // Place checkpoint at midpoint
            _checkpoints.Add(new Vector2((x1 + x2) / 2, (y1 + y2) / 2));
        }
    }

    public void GetObservations(Span<float> observations)
    {
        // Cast 9 rays at different angles relative to car heading
        for (int i = 0; i < SensorAngles.Length; i++)
        {
            float angleRad = SensorAngles[i] * MathF.PI / 180f;
            float rayAngle = _heading + angleRad;
            float distance = CastRay(_position, rayAngle, MAX_SENSOR_RANGE);

            // Normalize: 0 = far away, 1 = very close
            observations[i] = 1f - (distance / MAX_SENSOR_RANGE);
        }
    }

    private float CastRay(Vector2 origin, float angle, float maxRange)
    {
        Vector2 direction = new Vector2(MathF.Cos(angle), MathF.Sin(angle));
        Vector2 rayEnd = origin + direction * maxRange;

        float minDistance = maxRange;

        // Check intersection with all wall segments
        foreach (var (leftStart, leftEnd, rightStart, rightEnd) in _wallSegments)
        {
            // Check left wall
            if (LineIntersection(origin, rayEnd, leftStart, leftEnd, out float t1))
            {
                float dist = t1 * maxRange;
                if (dist < minDistance) minDistance = dist;
            }

            // Check right wall
            if (LineIntersection(origin, rayEnd, rightStart, rightEnd, out float t2))
            {
                float dist = t2 * maxRange;
                if (dist < minDistance) minDistance = dist;
            }
        }

        return minDistance;
    }

    private bool LineIntersection(Vector2 p1, Vector2 p2, Vector2 p3, Vector2 p4, out float t)
    {
        Vector2 s1 = p2 - p1;
        Vector2 s2 = p4 - p3;

        float denom = Cross2D(s1, s2);
        if (MathF.Abs(denom) < 1e-6f)
        {
            t = 0;
            return false;
        }

        float s = Cross2D(p3 - p1, s1) / denom;
        t = Cross2D(p3 - p1, s2) / denom;

        return t >= 0 && t <= 1 && s >= 0 && s <= 1;
    }

    private float Cross2D(Vector2 a, Vector2 b) => a.X * b.Y - a.Y * b.X;

    public float Step(ReadOnlySpan<float> actions)
    {
        if (_crashed)
            return 0f;

        _step++;

        // Extract actions
        float steering = Math.Clamp(actions[0], -1f, 1f);
        float throttle = Math.Clamp(actions[1], -1f, 1f);

        // Update heading based on steering (more effective at higher speeds)
        float steeringEffect = steering * 0.08f * (_speed / MAX_SPEED);
        _heading += steeringEffect;

        // Update speed based on throttle
        if (throttle > 0)
        {
            _speed += throttle * 1.5f; // Acceleration
        }
        else
        {
            _speed += throttle * 3f; // Braking
        }
        _speed = Math.Clamp(_speed, 0f, MAX_SPEED);

        // Update position
        Vector2 velocity = new Vector2(MathF.Cos(_heading), MathF.Sin(_heading)) * _speed;
        _position += velocity * 0.1f; // dt = 0.1

        // Check for wall collision
        float distanceToWall = CastRay(_position, _heading, CAR_RADIUS * 1.5f);
        if (distanceToWall < CAR_RADIUS)
        {
            _crashed = true;
            // Penalty for crashing
            return -0.5f;
        }

        // Check for checkpoint collection
        float reward = 0f;
        while (_checkpointIndex < _checkpoints.Count)
        {
            Vector2 checkpoint = _checkpoints[_checkpointIndex];
            float distance = Vector2.Distance(_position, checkpoint);

            if (distance < CHECKPOINT_RADIUS)
            {
                // Reward for reaching checkpoint
                reward += 1f / _checkpoints.Count;
                _checkpointIndex++;
            }
            else
            {
                break; // Must reach checkpoints in order
            }
        }

        // Small time penalty to encourage speed
        reward -= 0.001f;

        // Bonus for finishing
        if (_checkpointIndex >= _checkpoints.Count)
        {
            reward += 0.1f;
        }

        return reward;
    }

    public bool IsTerminal()
    {
        return _crashed ||
               _step >= MaxSteps ||
               _checkpointIndex >= _checkpoints.Count;
    }

    /// <summary>
    /// Get current progress for debugging/visualization.
    /// Returns (checkpoints_collected / total_checkpoints).
    /// </summary>
    public float GetProgress()
    {
        return _checkpointIndex / (float)_checkpoints.Count;
    }

    /// <summary>
    /// Get current car state for visualization.
    /// </summary>
    public (Vector2 position, float heading, float speed, int step) GetCarState()
    {
        return (_position, _heading, _speed, _step);
    }
}
