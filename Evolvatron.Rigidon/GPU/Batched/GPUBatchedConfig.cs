namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// Configuration for batched GPU simulation of N parallel worlds.
/// </summary>
public struct GPUBatchedWorldConfig
{
    /// <summary>Number of parallel simulations (worlds).</summary>
    public int WorldCount;

    /// <summary>Rigid bodies per world (e.g., 2 for rocket body + engine).</summary>
    public int RigidBodiesPerWorld;

    /// <summary>Geoms per world (collision shapes per rigid body).</summary>
    public int GeomsPerWorld;

    /// <summary>Revolute joints per world.</summary>
    public int JointsPerWorld;

    /// <summary>Static colliders (shared across all worlds - arena walls).</summary>
    public int SharedColliderCount;

    /// <summary>Max targets (breadcrumbs) per world.</summary>
    public int TargetsPerWorld;

    /// <summary>Max contacts that can be detected per world per step.</summary>
    public int MaxContactsPerWorld;

    // Helper properties for computing array sizes
    public readonly int TotalRigidBodies => WorldCount * RigidBodiesPerWorld;
    public readonly int TotalGeoms => WorldCount * GeomsPerWorld;
    public readonly int TotalJoints => WorldCount * JointsPerWorld;
    public readonly int TotalTargets => WorldCount * TargetsPerWorld;
    public readonly int TotalContacts => WorldCount * MaxContactsPerWorld;

    // Index computation helpers
    public readonly int GetRigidBodyIndex(int worldIdx, int localRbIdx) => worldIdx * RigidBodiesPerWorld + localRbIdx;
    public readonly int GetGeomIndex(int worldIdx, int localGeomIdx) => worldIdx * GeomsPerWorld + localGeomIdx;
    public readonly int GetJointIndex(int worldIdx, int localJointIdx) => worldIdx * JointsPerWorld + localJointIdx;
    public readonly int GetTargetIndex(int worldIdx, int localTargetIdx) => worldIdx * TargetsPerWorld + localTargetIdx;
    public readonly int GetContactIndex(int worldIdx, int localContactIdx) => worldIdx * MaxContactsPerWorld + localContactIdx;

    // Reverse: get worldIdx from global index
    public readonly int GetWorldFromRigidBodyIndex(int globalIdx) => globalIdx / RigidBodiesPerWorld;
    public readonly int GetWorldFromGeomIndex(int globalIdx) => globalIdx / GeomsPerWorld;
    public readonly int GetWorldFromTargetIndex(int globalIdx) => globalIdx / TargetsPerWorld;
    public readonly int GetWorldFromContactIndex(int globalIdx) => globalIdx / MaxContactsPerWorld;

    /// <summary>
    /// Factory for common rocket chase configuration.
    /// </summary>
    /// <param name="worldCount">Number of parallel worlds to simulate.</param>
    /// <param name="targetsPerWorld">Maximum targets (breadcrumbs) per world.</param>
    /// <returns>Configuration suitable for rocket chase scenarios.</returns>
    public static GPUBatchedWorldConfig ForRocketChase(int worldCount, int targetsPerWorld = 10)
    {
        return new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = 3,  // body + 2 legs
            GeomsPerWorld = 8,        // multiple collision circles
            JointsPerWorld = 2,       // leg joints
            SharedColliderCount = 4,  // arena walls (floor, ceiling, sides)
            TargetsPerWorld = targetsPerWorld,
            MaxContactsPerWorld = 32  // generous contact budget
        };
    }
}

/// <summary>
/// Configuration for the batched environment (rewards, observations, actions).
/// </summary>
public struct GPUBatchedEnvironmentConfig
{
    /// <summary>Number of observation values per world.</summary>
    public int ObservationsPerWorld;

    /// <summary>Number of action values per world.</summary>
    public int ActionsPerWorld;

    /// <summary>Maximum simulation steps per episode before timeout.</summary>
    public int MaxStepsPerEpisode;

    /// <summary>Arena left boundary.</summary>
    public float ArenaMinX;

    /// <summary>Arena right boundary.</summary>
    public float ArenaMaxX;

    /// <summary>Arena bottom boundary.</summary>
    public float ArenaMinY;

    /// <summary>Arena top boundary.</summary>
    public float ArenaMaxY;

    /// <summary>Radius within which a target is considered "hit".</summary>
    public float TargetRadius;

    /// <summary>Reward for hitting a target.</summary>
    public float TargetHitReward;

    /// <summary>Penalty for crashing (high impact collision).</summary>
    public float CrashPenalty;

    /// <summary>Penalty for going out of bounds.</summary>
    public float OutOfBoundsPenalty;

    /// <summary>Small penalty per time step to encourage efficiency.</summary>
    public float TimeStepPenalty;

    // World count stored for total calculations
    private int _worldCount;

    /// <summary>Number of parallel worlds (set via factory method).</summary>
    public int WorldCount
    {
        readonly get => _worldCount;
        set => _worldCount = value;
    }

    /// <summary>Total observation buffer size across all worlds.</summary>
    public readonly int TotalObservations => _worldCount * ObservationsPerWorld;

    /// <summary>Total action buffer size across all worlds.</summary>
    public readonly int TotalActions => _worldCount * ActionsPerWorld;

    /// <summary>
    /// Factory for target chase environment configuration.
    /// </summary>
    /// <param name="worldCount">Number of parallel worlds.</param>
    /// <returns>Configuration suitable for target chase training.</returns>
    public static GPUBatchedEnvironmentConfig ForTargetChase(int worldCount)
    {
        return new GPUBatchedEnvironmentConfig
        {
            _worldCount = worldCount,
            ObservationsPerWorld = 4,  // Simple: dx, dy, vx, vy
            ActionsPerWorld = 2,       // thrust_x, thrust_y
            MaxStepsPerEpisode = 300,
            ArenaMinX = -10f,
            ArenaMaxX = 10f,
            ArenaMinY = -10f,          // Symmetric arena (no gravity)
            ArenaMaxY = 10f,
            TargetRadius = 1.5f,       // Medium targets
            TargetHitReward = 100f,    // Good reward for hitting
            CrashPenalty = 0f,         // No crash (spherical, can't flip)
            OutOfBoundsPenalty = -50f,
            TimeStepPenalty = -0.1f
        };
    }
}
