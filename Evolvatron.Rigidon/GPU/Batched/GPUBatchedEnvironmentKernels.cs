using ILGPU;
using ILGPU.Algorithms;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// ILGPU kernels for batched environment logic (targets, rewards, observations, actions).
/// These kernels handle the game/RL environment aspects that run alongside physics.
///
/// Key insight: Environment state is per-world while physics operates on per-body/geom granularity.
/// Target indices: globalTargetIdx = worldIdx * targetsPerWorld + localTargetIdx
/// </summary>
public static class GPUBatchedEnvironmentKernels
{
    private const float Epsilon = 1e-9f;
    private const float Pi = 3.14159265358979f;

    #region Target Management

    /// <summary>
    /// Initialize targets with random positions for all worlds.
    /// Uses a simple LCG PRNG for deterministic randomness on GPU.
    ///
    /// One thread per target across all worlds.
    /// </summary>
    public static void BatchedInitializeTargetsKernel(
        Index1D globalTargetIdx,
        ArrayView<float> targetPositions,    // [x,y] pairs, size = worldCount * targetsPerWorld * 2
        ArrayView<byte> targetActive,        // 1 = active, 0 = collected
        int targetsPerWorld,
        int baseSeed,
        float arenaMinX,
        float arenaMaxX,
        float arenaMinY,
        float arenaMaxY,
        float margin)
    {
        int worldIdx = globalTargetIdx / targetsPerWorld;
        int localTargetIdx = globalTargetIdx % targetsPerWorld;

        // Only first target is active initially
        targetActive[globalTargetIdx] = (byte)(localTargetIdx == 0 ? 1 : 0);

        // LCG random number generator
        // seed = baseSeed + worldIdx * 10000 + localTargetIdx * 100
        uint seed = (uint)(baseSeed + worldIdx * 10000 + localTargetIdx * 100);
        seed = LCGNext(seed);
        float randX = LCGToFloat(seed);
        seed = LCGNext(seed);
        float randY = LCGToFloat(seed);

        // Map to arena bounds with margin
        float minX = arenaMinX + margin;
        float maxX = arenaMaxX - margin;
        float minY = arenaMinY + margin;
        float maxY = arenaMaxY - margin;

        float x = minX + randX * (maxX - minX);
        float y = minY + randY * (maxY - minY);

        int posIdx = globalTargetIdx * 2;
        targetPositions[posIdx] = x;
        targetPositions[posIdx + 1] = y;
    }

    /// <summary>
    /// Spawn a new target at a random position for a specific world.
    /// Called after a target is collected to spawn the next one.
    /// One thread per world.
    /// </summary>
    public static void BatchedSpawnNextTargetKernel(
        Index1D worldIdx,
        ArrayView<float> targetPositions,
        ArrayView<byte> targetActive,
        ArrayView<int> targetsCollected,
        int targetsPerWorld,
        int baseSeed,
        float arenaMinX,
        float arenaMaxX,
        float arenaMinY,
        float arenaMaxY,
        float margin,
        float rocketX,                        // Avoid spawning too close to rocket
        float rocketY,
        float minDistanceFromRocket)
    {
        int collected = targetsCollected[worldIdx];

        // If all targets for this world are used up, don't spawn more
        if (collected >= targetsPerWorld) return;

        int localTargetIdx = collected;  // Next target to activate
        int globalTargetIdx = worldIdx * targetsPerWorld + localTargetIdx;

        // Generate random position
        uint seed = (uint)(baseSeed + worldIdx * 10000 + localTargetIdx * 100 + collected * 7);

        float x, y, dist;
        int attempts = 0;
        const int maxAttempts = 10;

        do
        {
            seed = LCGNext(seed);
            float randX = LCGToFloat(seed);
            seed = LCGNext(seed);
            float randY = LCGToFloat(seed);

            float minX = arenaMinX + margin;
            float maxX = arenaMaxX - margin;
            float minY = arenaMinY + margin;
            float maxY = arenaMaxY - margin;

            x = minX + randX * (maxX - minX);
            y = minY + randY * (maxY - minY);

            float dx = x - rocketX;
            float dy = y - rocketY;
            dist = XMath.Sqrt(dx * dx + dy * dy);
            attempts++;
        }
        while (dist < minDistanceFromRocket && attempts < maxAttempts);

        int posIdx = globalTargetIdx * 2;
        targetPositions[posIdx] = x;
        targetPositions[posIdx + 1] = y;
        targetActive[globalTargetIdx] = 1;
    }

    /// <summary>
    /// Check for target collisions and collect targets.
    /// One thread per world - checks the currently active target.
    /// </summary>
    public static void BatchedCheckTargetCollisionsKernel(
        Index1D worldIdx,
        ArrayView<float> targetPositions,
        ArrayView<byte> targetActive,
        ArrayView<int> targetsCollected,
        ArrayView<float> cumulativeRewards,
        ArrayView<byte> isTerminal,
        int targetsPerWorld,
        float rocketX,
        float rocketY,
        float targetHitRadius,
        float targetHitReward)
    {
        // Skip if world is terminal
        if (isTerminal[worldIdx] != 0) return;

        // Find the currently active target for this world
        int baseTargetIdx = worldIdx * targetsPerWorld;
        int collected = targetsCollected[worldIdx];

        if (collected >= targetsPerWorld) return;  // All targets collected

        int activeTargetIdx = baseTargetIdx + collected;
        if (targetActive[activeTargetIdx] == 0) return;

        // Get target position
        int posIdx = activeTargetIdx * 2;
        float tx = targetPositions[posIdx];
        float ty = targetPositions[posIdx + 1];

        // Check collision
        float dx = tx - rocketX;
        float dy = ty - rocketY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        if (dist < targetHitRadius)
        {
            // Collect target
            targetActive[activeTargetIdx] = 0;
            targetsCollected[worldIdx] = collected + 1;
            cumulativeRewards[worldIdx] += targetHitReward;
        }
    }

    /// <summary>
    /// Check for target collisions reading rocket position from bodies array.
    /// This version works for batched evaluation where each world has its own rocket.
    /// One thread per world.
    /// </summary>
    public static void BatchedCheckTargetCollisionsFromBodiesKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> targetPositions,
        ArrayView<byte> targetActive,
        ArrayView<int> targetsCollected,
        ArrayView<float> cumulativeRewards,
        ArrayView<byte> isTerminal,
        int bodiesPerWorld,
        int primaryBodyLocalIdx,
        int targetsPerWorld,
        float targetHitRadius,
        float targetHitReward)
    {
        // Skip if world is terminal
        if (isTerminal[worldIdx] != 0) return;

        // Get rocket position from primary body
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[bodyGlobalIdx];
        float rocketX = body.X;
        float rocketY = body.Y;

        // Find the currently active target for this world
        int baseTargetIdx = worldIdx * targetsPerWorld;
        int collected = targetsCollected[worldIdx];

        if (collected >= targetsPerWorld) return;  // All targets collected

        int activeTargetIdx = baseTargetIdx + collected;
        if (targetActive[activeTargetIdx] == 0) return;

        // Get target position
        int posIdx = activeTargetIdx * 2;
        float tx = targetPositions[posIdx];
        float ty = targetPositions[posIdx + 1];

        // Check collision
        float dx = tx - rocketX;
        float dy = ty - rocketY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        if (dist < targetHitRadius)
        {
            // Collect target
            targetActive[activeTargetIdx] = 0;
            int newCollected = collected + 1;
            targetsCollected[worldIdx] = newCollected;
            cumulativeRewards[worldIdx] += targetHitReward;

            // Activate next target if available
            if (newCollected < targetsPerWorld)
            {
                int nextTargetIdx = baseTargetIdx + newCollected;
                targetActive[nextTargetIdx] = 1;
            }
        }
    }

    #endregion

    #region Observations

    /// <summary>
    /// Extract observations for neural network input.
    /// One thread per world.
    ///
    /// Observation layout (8 values):
    /// [0] targetDirX - normalized direction to target
    /// [1] targetDirY
    /// [2] velX - rocket velocity (normalized)
    /// [3] velY
    /// [4] upX - rocket up vector
    /// [5] upY
    /// [6] distToTarget - distance (normalized)
    /// [7] angularVel - angular velocity (normalized)
    /// </summary>
    public static void BatchedGetObservationsKernel(
        Index1D worldIdx,
        ArrayView<float> observations,
        ArrayView<float> targetPositions,
        ArrayView<byte> targetActive,
        ArrayView<int> targetsCollected,
        ArrayView<byte> isTerminal,
        int observationsPerWorld,
        int targetsPerWorld,
        float rocketX,
        float rocketY,
        float rocketVelX,
        float rocketVelY,
        float rocketAngle,
        float rocketAngularVel,
        float velocityNormalization,
        float distanceNormalization,
        float angularVelNormalization)
    {
        int obsBase = worldIdx * observationsPerWorld;

        // If terminal, return zeros
        if (isTerminal[worldIdx] != 0)
        {
            for (int i = 0; i < observationsPerWorld; i++)
            {
                observations[obsBase + i] = 0f;
            }
            return;
        }

        // Up vector from angle (computed once)
        float cos = XMath.Cos(rocketAngle);
        float sin = XMath.Sin(rocketAngle);
        float upX = -sin;
        float upY = cos;

        // Find active target
        int collected = targetsCollected[worldIdx];
        if (collected >= targetsPerWorld)
        {
            // No more targets - default observations
            observations[obsBase + 0] = 0f;
            observations[obsBase + 1] = 0f;
            observations[obsBase + 2] = rocketVelX / velocityNormalization;
            observations[obsBase + 3] = rocketVelY / velocityNormalization;
            observations[obsBase + 4] = upX;
            observations[obsBase + 5] = upY;
            observations[obsBase + 6] = 0f;
            observations[obsBase + 7] = rocketAngularVel / angularVelNormalization;
            return;
        }

        int activeTargetIdx = worldIdx * targetsPerWorld + collected;
        int posIdx = activeTargetIdx * 2;
        float tx = targetPositions[posIdx];
        float ty = targetPositions[posIdx + 1];

        // Direction to target (normalized)
        float dx = tx - rocketX;
        float dy = ty - rocketY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        float dirX = 0f, dirY = 0f;
        if (dist > Epsilon)
        {
            dirX = dx / dist;
            dirY = dy / dist;
        }

        // Write observations
        observations[obsBase + 0] = dirX;
        observations[obsBase + 1] = dirY;
        observations[obsBase + 2] = rocketVelX / velocityNormalization;
        observations[obsBase + 3] = rocketVelY / velocityNormalization;
        observations[obsBase + 4] = upX;
        observations[obsBase + 5] = upY;
        observations[obsBase + 6] = dist / distanceNormalization;
        observations[obsBase + 7] = rocketAngularVel / angularVelNormalization;
    }

    /// <summary>
    /// Batched version that reads rocket state from rigid body arrays.
    /// Simple spherical rocket observations:
    /// [0] dx - direction to target X (normalized to arena size)
    /// [1] dy - direction to target Y (normalized to arena size)
    /// [2] vx - velocity X (normalized)
    /// [3] vy - velocity Y (normalized)
    /// </summary>
    public static void BatchedGetObservationsFromBodiesKernel(
        Index1D worldIdx,
        ArrayView<float> observations,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> targetPositions,
        ArrayView<int> targetsCollected,
        ArrayView<byte> isTerminal,
        int observationsPerWorld,
        int bodiesPerWorld,
        int targetsPerWorld,
        int primaryBodyLocalIdx,
        float velocityNormalization,
        float distanceNormalization,
        float angularVelNormalization)
    {
        int obsBase = worldIdx * observationsPerWorld;

        // If terminal, return zeros
        if (isTerminal[worldIdx] != 0)
        {
            for (int i = 0; i < observationsPerWorld; i++)
            {
                observations[obsBase + i] = 0f;
            }
            return;
        }

        // Get rocket state from primary body
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[bodyGlobalIdx];

        float rocketX = body.X;
        float rocketY = body.Y;
        float rocketVelX = body.VelX;
        float rocketVelY = body.VelY;

        // Find active target
        int collected = targetsCollected[worldIdx];
        float tx = 0f, ty = 0f;

        if (collected < targetsPerWorld)
        {
            int activeTargetIdx = worldIdx * targetsPerWorld + collected;
            int posIdx = activeTargetIdx * 2;
            tx = targetPositions[posIdx];
            ty = targetPositions[posIdx + 1];
        }

        // Delta to target (normalized to arena size ~20 units)
        float dx = (tx - rocketX) / distanceNormalization;
        float dy = (ty - rocketY) / distanceNormalization;

        // Simple observations: dx, dy, vx, vy
        observations[obsBase + 0] = dx;
        observations[obsBase + 1] = dy;
        observations[obsBase + 2] = rocketVelX / velocityNormalization;
        observations[obsBase + 3] = rocketVelY / velocityNormalization;
    }

    #endregion

    #region Actions

    /// <summary>
    /// Apply actions as direct X/Y thrust to spherical rockets.
    /// Actions: [thrust_x, thrust_y] in [-1,1] range, scaled by maxThrust.
    ///
    /// One thread per world.
    /// </summary>
    public static void BatchedApplyActionsKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> actions,
        ArrayView<byte> isTerminal,
        int actionsPerWorld,
        int bodiesPerWorld,
        int primaryBodyLocalIdx,
        float maxThrust,
        float maxGimbalTorque,  // Unused for spherical rocket
        float dt)
    {
        // Skip if terminal
        if (isTerminal[worldIdx] != 0) return;

        int actionBase = worldIdx * actionsPerWorld;
        float thrustX = actions[actionBase + 0];
        float thrustY = actions[actionBase + 1];

        // Clamp actions to [-1, 1]
        if (thrustX < -1f) thrustX = -1f;
        if (thrustX > 1f) thrustX = 1f;
        if (thrustY < -1f) thrustY = -1f;
        if (thrustY > 1f) thrustY = 1f;

        // Get primary body
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[bodyGlobalIdx];

        // Apply direct X/Y thrust
        float fx = thrustX * maxThrust;
        float fy = thrustY * maxThrust;

        body.VelX += fx * body.InvMass * dt;
        body.VelY += fy * body.InvMass * dt;

        bodies[bodyGlobalIdx] = body;
    }

    #endregion

    #region Terminal Conditions and Rewards

    /// <summary>
    /// Check terminal conditions for all worlds.
    /// Terminal if: out of bounds or max steps reached.
    /// (Spherical rockets can't crash/flip)
    ///
    /// One thread per world.
    /// </summary>
    public static void BatchedCheckTerminalConditionsKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<int> stepCounters,
        ArrayView<byte> isTerminal,
        ArrayView<float> cumulativeRewards,
        int bodiesPerWorld,
        int primaryBodyLocalIdx,
        int maxSteps,
        float arenaMinX,
        float arenaMaxX,
        float arenaMinY,
        float arenaMaxY,
        float crashAngleThreshold,        // Unused for spherical rocket
        float outOfBoundsPenalty,
        float crashPenalty)               // Unused for spherical rocket
    {
        // Skip if already terminal
        if (isTerminal[worldIdx] != 0) return;

        // Increment step counter
        int steps = stepCounters[worldIdx] + 1;
        stepCounters[worldIdx] = steps;

        // Get rocket state
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[bodyGlobalIdx];

        float x = body.X;
        float y = body.Y;

        // Check out of bounds
        if (x < arenaMinX || x > arenaMaxX || y < arenaMinY || y > arenaMaxY)
        {
            isTerminal[worldIdx] = 1;
            cumulativeRewards[worldIdx] += outOfBoundsPenalty;
            return;
        }

        // Check max steps
        if (steps >= maxSteps)
        {
            isTerminal[worldIdx] = 1;
        }
    }

    /// <summary>
    /// Compute shaping reward for approaching target.
    /// Simple distance-based reward for spherical rockets.
    /// One thread per world.
    /// </summary>
    public static void BatchedComputeShapingRewardKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> targetPositions,
        ArrayView<int> targetsCollected,
        ArrayView<float> cumulativeRewards,
        ArrayView<byte> isTerminal,
        int bodiesPerWorld,
        int targetsPerWorld,
        int primaryBodyLocalIdx,
        float distanceRewardScale,        // e.g., 0.1
        float orientationRewardScale,     // Unused for spherical rocket
        float timeStepPenalty)            // e.g., -0.1
    {
        // Skip if terminal
        if (isTerminal[worldIdx] != 0) return;

        // Get rocket state
        int bodyGlobalIdx = worldIdx * bodiesPerWorld + primaryBodyLocalIdx;
        var body = bodies[bodyGlobalIdx];

        float rocketX = body.X;
        float rocketY = body.Y;

        // Find active target
        int collected = targetsCollected[worldIdx];
        if (collected >= targetsPerWorld)
        {
            // No targets left - just time penalty
            cumulativeRewards[worldIdx] += timeStepPenalty;
            return;
        }

        int activeTargetIdx = worldIdx * targetsPerWorld + collected;
        int posIdx = activeTargetIdx * 2;
        float tx = targetPositions[posIdx];
        float ty = targetPositions[posIdx + 1];

        // Distance to target
        float dx = tx - rocketX;
        float dy = ty - rocketY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        // Simple proximity-based reward: closer = better
        // Max distance in arena is ~28 (diagonal of 20x20)
        // Reward range: -0.4 to +2.4 per step based on distance
        float proximityReward = (14f - dist) * distanceRewardScale;

        // Total step reward: proximity - time penalty
        float stepReward = proximityReward + timeStepPenalty;
        cumulativeRewards[worldIdx] += stepReward;
    }

    /// <summary>
    /// Compute final fitness values from cumulative rewards and targets collected.
    /// One thread per world.
    /// </summary>
    public static void BatchedComputeFitnessKernel(
        Index1D worldIdx,
        ArrayView<float> fitnessValues,
        ArrayView<float> cumulativeRewards,
        ArrayView<int> targetsCollected,
        float targetBonusMultiplier)      // e.g., 200
    {
        float bonus = targetsCollected[worldIdx] * targetBonusMultiplier;
        fitnessValues[worldIdx] = bonus + cumulativeRewards[worldIdx];
    }

    #endregion

    #region Episode Reset

    /// <summary>
    /// Reset environment state for worlds that need resetting.
    /// One thread per world.
    /// </summary>
    public static void BatchedResetEnvironmentKernel(
        Index1D worldIdx,
        ArrayView<int> resetFlags,        // 1 = reset this world
        ArrayView<float> cumulativeRewards,
        ArrayView<int> stepCounters,
        ArrayView<byte> isTerminal,
        ArrayView<int> targetsCollected,
        ArrayView<float> fitnessValues)
    {
        if (resetFlags[worldIdx] == 0) return;

        cumulativeRewards[worldIdx] = 0f;
        stepCounters[worldIdx] = 0;
        isTerminal[worldIdx] = 0;
        targetsCollected[worldIdx] = 0;
        fitnessValues[worldIdx] = 0f;
    }

    /// <summary>
    /// Reset target active flags for worlds that need resetting.
    /// One thread per target.
    /// </summary>
    public static void BatchedResetTargetsKernel(
        Index1D globalTargetIdx,
        ArrayView<byte> targetActive,
        ArrayView<int> resetFlags,
        int targetsPerWorld)
    {
        int worldIdx = globalTargetIdx / targetsPerWorld;
        int localTargetIdx = globalTargetIdx % targetsPerWorld;

        if (resetFlags[worldIdx] == 0) return;

        // Only first target is active after reset
        targetActive[globalTargetIdx] = (byte)(localTargetIdx == 0 ? 1 : 0);
    }

    #endregion

    #region Utility: LCG Random

    /// <summary>
    /// Linear Congruential Generator step.
    /// Simple, fast PRNG suitable for GPU.
    /// </summary>
    private static uint LCGNext(uint seed)
    {
        // Parameters from Numerical Recipes
        return seed * 1664525u + 1013904223u;
    }

    /// <summary>
    /// Convert LCG state to float in [0,1).
    /// </summary>
    private static float LCGToFloat(uint seed)
    {
        // Divide by 2^32 to get [0,1)
        return seed / 4294967296f;
    }

    #endregion
}
