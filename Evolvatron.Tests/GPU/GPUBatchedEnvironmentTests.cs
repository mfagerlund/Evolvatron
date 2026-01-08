using Xunit;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace Evolvatron.Tests.GPU;

/// <summary>
/// Tests for the GPU batched environment kernels.
/// These tests verify target management, observations, actions, terminal conditions,
/// and fitness computation for parallel world simulations.
/// </summary>
public class GPUBatchedEnvironmentTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    // Compiled kernel delegates
    private readonly Action<Index1D, ArrayView<float>, ArrayView<byte>, int, int, float, float, float, float, float> _initTargetsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, float, float, float, float> _checkTargetCollisionsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<int>, ArrayView<byte>, int, int, int, int, float, float, float> _getObservationsFromBodiesKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<byte>, int, int, int, float, float, float> _applyActionsKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<int>, ArrayView<byte>, ArrayView<float>, int, int, int, float, float, float, float, float, float, float> _checkTerminalConditionsKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, int, int, float, float, float> _computeShapingRewardKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, float> _computeFitnessKernel;

    public GPUBatchedEnvironmentTests()
    {
        _context = Context.CreateDefault();
        _accelerator = _context.CreateCPUAccelerator(0);

        // Compile kernels
        _initTargetsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<byte>, int, int, float, float, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedInitializeTargetsKernel);

        _checkTargetCollisionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, float, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedCheckTargetCollisionsKernel);

        _getObservationsFromBodiesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<int>, ArrayView<byte>, int, int, int, int, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedGetObservationsFromBodiesKernel);

        _applyActionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<byte>, int, int, int, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedApplyActionsKernel);

        _checkTerminalConditionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<int>, ArrayView<byte>, ArrayView<float>, int, int, int, float, float, float, float, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedCheckTerminalConditionsKernel);

        _computeShapingRewardKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, int, int, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedComputeShapingRewardKernel);

        _computeFitnessKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, float>(
            GPUBatchedEnvironmentKernels.BatchedComputeFitnessKernel);
    }

    #region Target Initialization Tests

    [Fact]
    public void InitializeTargets_ProducesDifferentPositionsPerWorld()
    {
        // Arrange
        const int worldCount = 5;
        const int targetsPerWorld = 3;
        int totalTargets = worldCount * targetsPerWorld;

        using var targetPositions = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive = _accelerator.Allocate1D<byte>(totalTargets);

        // Act
        // Parameters: targetPositions, targetActive, targetsPerWorld, baseSeed, arenaMinX, arenaMaxX, arenaMinY, arenaMaxY, margin
        _initTargetsKernel(
            totalTargets,
            targetPositions.View,
            targetActive.View,
            targetsPerWorld,
            12345,      // baseSeed
            -10f,       // arenaMinX
            10f,        // arenaMaxX
            -5f,        // arenaMinY
            15f,        // arenaMaxY
            1f);        // margin
        _accelerator.Synchronize();

        // Assert
        var positions = targetPositions.GetAsArray1D();
        var active = targetActive.GetAsArray1D();

        // Verify positions are within bounds (with margin)
        for (int i = 0; i < totalTargets; i++)
        {
            float x = positions[i * 2];
            float y = positions[i * 2 + 1];
            Assert.InRange(x, -9f, 9f);   // minX + margin to maxX - margin
            Assert.InRange(y, -4f, 14f);  // minY + margin to maxY - margin
        }

        // Verify different worlds get different positions for their first target
        float world0Target0X = positions[0];
        float world1Target0X = positions[targetsPerWorld * 2];  // World 1, target 0
        float world2Target0X = positions[targetsPerWorld * 2 * 2];  // World 2, target 0

        // With LCG seeded differently per world, positions should differ
        Assert.NotEqual(world0Target0X, world1Target0X);
        Assert.NotEqual(world1Target0X, world2Target0X);
    }

    [Fact]
    public void InitializeTargets_OnlyFirstTargetIsActive()
    {
        // Arrange
        const int worldCount = 3;
        const int targetsPerWorld = 5;
        int totalTargets = worldCount * targetsPerWorld;

        using var targetPositions = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive = _accelerator.Allocate1D<byte>(totalTargets);

        // Act
        _initTargetsKernel(
            totalTargets,
            targetPositions.View,
            targetActive.View,
            targetsPerWorld,
            42,         // baseSeed
            -10f,       // arenaMinX
            10f,        // arenaMaxX
            -5f,        // arenaMinY
            15f,        // arenaMaxY
            0.5f);      // margin
        _accelerator.Synchronize();

        // Assert
        var active = targetActive.GetAsArray1D();

        for (int w = 0; w < worldCount; w++)
        {
            for (int t = 0; t < targetsPerWorld; t++)
            {
                int globalIdx = w * targetsPerWorld + t;
                if (t == 0)
                {
                    Assert.Equal(1, active[globalIdx]); // First target active
                }
                else
                {
                    Assert.Equal(0, active[globalIdx]); // Others inactive
                }
            }
        }
    }

    [Fact]
    public void InitializeTargets_DeterministicWithSameSeed()
    {
        // Arrange
        const int worldCount = 4;
        const int targetsPerWorld = 3;
        int totalTargets = worldCount * targetsPerWorld;

        using var targetPositions1 = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive1 = _accelerator.Allocate1D<byte>(totalTargets);
        using var targetPositions2 = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive2 = _accelerator.Allocate1D<byte>(totalTargets);

        // Act - run twice with same seed
        _initTargetsKernel(
            totalTargets, targetPositions1.View, targetActive1.View,
            targetsPerWorld, 99999, -10f, 10f, -5f, 15f, 1f);
        _accelerator.Synchronize();

        _initTargetsKernel(
            totalTargets, targetPositions2.View, targetActive2.View,
            targetsPerWorld, 99999, -10f, 10f, -5f, 15f, 1f);
        _accelerator.Synchronize();

        // Assert - positions should be identical
        var pos1 = targetPositions1.GetAsArray1D();
        var pos2 = targetPositions2.GetAsArray1D();

        for (int i = 0; i < pos1.Length; i++)
        {
            Assert.Equal(pos1[i], pos2[i], 6);
        }
    }

    #endregion

    #region Target Collision Tests

    [Fact]
    public void CheckTargetCollisions_CollectsTargetWhenRocketIsClose()
    {
        // Arrange
        const int worldCount = 2;
        const int targetsPerWorld = 3;
        int totalTargets = worldCount * targetsPerWorld;

        using var targetPositions = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive = _accelerator.Allocate1D<byte>(totalTargets);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // Set target at (5, 5) for world 0
        var positions = new float[totalTargets * 2];
        positions[0] = 5f;  // World 0, target 0, x
        positions[1] = 5f;  // World 0, target 0, y
        positions[targetsPerWorld * 2] = 10f;  // World 1, target 0, x
        positions[targetsPerWorld * 2 + 1] = 10f;  // World 1, target 0, y
        targetPositions.CopyFromCPU(positions);

        var active = new byte[totalTargets];
        active[0] = 1;  // World 0, target 0 active
        active[targetsPerWorld] = 1;  // World 1, target 0 active
        targetActive.CopyFromCPU(active);

        targetsCollected.MemSetToZero();
        cumulativeRewards.MemSetToZero();
        isTerminal.MemSetToZero();

        // Act - rocket at (5.3, 5) should hit target at (5,5) with radius 0.5
        // Parameters: targetPositions, targetActive, targetsCollected, cumulativeRewards, isTerminal,
        //             targetsPerWorld, rocketX, rocketY, targetHitRadius, targetHitReward
        _checkTargetCollisionsKernel(
            worldCount,
            targetPositions.View,
            targetActive.View,
            targetsCollected.View,
            cumulativeRewards.View,
            isTerminal.View,
            targetsPerWorld,
            5.3f,       // rocketX
            5f,         // rocketY
            0.5f,       // targetHitRadius
            100f);      // targetHitReward
        _accelerator.Synchronize();

        // Assert
        var collected = targetsCollected.GetAsArray1D();
        var rewards = cumulativeRewards.GetAsArray1D();
        var activeAfter = targetActive.GetAsArray1D();

        // World 0 should have collected target (rocket at 5.3, target at 5, distance 0.3 < 0.5)
        Assert.Equal(1, collected[0]);
        Assert.Equal(100f, rewards[0], 3);
        Assert.Equal(0, activeAfter[0]); // Target now inactive

        // World 1 should not have collected (target at 10,10, distance > 0.5)
        Assert.Equal(0, collected[1]);
        Assert.Equal(0f, rewards[1], 3);
        Assert.Equal(1, activeAfter[targetsPerWorld]); // Still active
    }

    [Fact]
    public void CheckTargetCollisions_SkipsTerminalWorlds()
    {
        // Arrange
        const int worldCount = 2;
        const int targetsPerWorld = 2;
        int totalTargets = worldCount * targetsPerWorld;

        using var targetPositions = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive = _accelerator.Allocate1D<byte>(totalTargets);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // Place target right at rocket position for both worlds
        var positions = new float[totalTargets * 2];
        positions[0] = 0f; positions[1] = 0f;  // World 0
        positions[4] = 0f; positions[5] = 0f;  // World 1
        targetPositions.CopyFromCPU(positions);

        var active = new byte[totalTargets];
        active[0] = 1;
        active[targetsPerWorld] = 1;
        targetActive.CopyFromCPU(active);

        // Mark world 1 as terminal
        var terminal = new byte[] { 0, 1 };
        isTerminal.CopyFromCPU(terminal);

        targetsCollected.MemSetToZero();
        cumulativeRewards.MemSetToZero();

        // Act
        _checkTargetCollisionsKernel(
            worldCount,
            targetPositions.View,
            targetActive.View,
            targetsCollected.View,
            cumulativeRewards.View,
            isTerminal.View,
            targetsPerWorld,
            0f,     // rocketX
            0f,     // rocketY
            1f,     // targetHitRadius
            50f);   // targetHitReward
        _accelerator.Synchronize();

        // Assert
        var collected = targetsCollected.GetAsArray1D();
        var rewards = cumulativeRewards.GetAsArray1D();

        Assert.Equal(1, collected[0]);  // World 0 collected
        Assert.Equal(50f, rewards[0]);

        Assert.Equal(0, collected[1]);  // World 1 skipped (terminal)
        Assert.Equal(0f, rewards[1]);
    }

    [Fact]
    public void CheckTargetCollisions_AddsRewardCorrectly()
    {
        // Arrange
        const int worldCount = 1;
        const int targetsPerWorld = 5;
        int totalTargets = worldCount * targetsPerWorld;

        using var targetPositions = _accelerator.Allocate1D<float>(totalTargets * 2);
        using var targetActive = _accelerator.Allocate1D<byte>(totalTargets);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // Pre-existing reward
        cumulativeRewards.CopyFromCPU(new float[] { 25f });

        // Target at origin
        var positions = new float[totalTargets * 2];
        positions[0] = 0f;
        positions[1] = 0f;
        targetPositions.CopyFromCPU(positions);

        var active = new byte[totalTargets];
        active[0] = 1;
        targetActive.CopyFromCPU(active);

        targetsCollected.MemSetToZero();
        isTerminal.MemSetToZero();

        // Act
        _checkTargetCollisionsKernel(
            worldCount,
            targetPositions.View,
            targetActive.View,
            targetsCollected.View,
            cumulativeRewards.View,
            isTerminal.View,
            targetsPerWorld,
            0f,     // rocketX
            0f,     // rocketY
            1f,     // targetHitRadius
            75f);   // targetHitReward
        _accelerator.Synchronize();

        // Assert - reward should be added to existing
        var rewards = cumulativeRewards.GetAsArray1D();
        Assert.Equal(100f, rewards[0], 3);  // 25 + 75
    }

    #endregion

    #region Observations Tests

    [Fact]
    public void GetObservationsFromBodies_ProducesValidNormalizedValues()
    {
        // Arrange
        const int worldCount = 2;
        const int observationsPerWorld = 8;
        const int bodiesPerWorld = 3;
        const int targetsPerWorld = 2;

        using var observations = _accelerator.Allocate1D<float>(worldCount * observationsPerWorld);
        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var targetPositions = _accelerator.Allocate1D<float>(worldCount * targetsPerWorld * 2);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // Set up bodies - primary body (index 0) at origin, pointing up
        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        for (int w = 0; w < worldCount; w++)
        {
            bodyData[w * bodiesPerWorld] = new GPURigidBody
            {
                X = 0f,
                Y = 0f,
                Angle = 0f,  // Pointing up (cos(0)=1, sin(0)=0)
                VelX = 5f,
                VelY = -3f,
                AngularVel = 0.5f
            };
        }
        bodies.CopyFromCPU(bodyData);

        // Target at (10, 10) for all worlds
        var targetPos = new float[worldCount * targetsPerWorld * 2];
        targetPos[0] = 10f;  // World 0, target 0, x
        targetPos[1] = 10f;  // World 0, target 0, y
        targetPos[4] = 10f;  // World 1, target 0, x
        targetPos[5] = 10f;  // World 1, target 0, y
        targetPositions.CopyFromCPU(targetPos);

        targetsCollected.MemSetToZero();
        isTerminal.MemSetToZero();

        // Act
        // Parameters: observations, bodies, targetPositions, targetsCollected, isTerminal,
        //             observationsPerWorld, bodiesPerWorld, targetsPerWorld, primaryBodyLocalIdx,
        //             velocityNormalization, distanceNormalization, angularVelNormalization
        _getObservationsFromBodiesKernel(
            worldCount,
            observations.View,
            bodies.View,
            targetPositions.View,
            targetsCollected.View,
            isTerminal.View,
            observationsPerWorld,
            bodiesPerWorld,
            targetsPerWorld,
            0,      // primaryBodyLocalIdx
            10f,    // velocityNormalization
            20f,    // distanceNormalization
            2f);    // angularVelNormalization
        _accelerator.Synchronize();

        // Assert
        var obs = observations.GetAsArray1D();

        for (int w = 0; w < worldCount; w++)
        {
            int baseIdx = w * observationsPerWorld;

            // Direction to target should be normalized (length ~1)
            float dirX = obs[baseIdx + 0];
            float dirY = obs[baseIdx + 1];
            float dirLen = MathF.Sqrt(dirX * dirX + dirY * dirY);
            Assert.InRange(dirLen, 0.99f, 1.01f);

            // Velocity should be normalized by 10
            float velX = obs[baseIdx + 2];
            float velY = obs[baseIdx + 3];
            Assert.Equal(0.5f, velX, 3);   // 5/10
            Assert.Equal(-0.3f, velY, 3);  // -3/10

            // Up vector for angle=0: (-sin(0), cos(0)) = (0, 1)
            float upX = obs[baseIdx + 4];
            float upY = obs[baseIdx + 5];
            Assert.Equal(0f, upX, 3);
            Assert.Equal(1f, upY, 3);

            // Distance to target: sqrt(10^2 + 10^2) / 20 = 14.14 / 20 = 0.707
            float normDist = obs[baseIdx + 6];
            Assert.InRange(normDist, 0.7f, 0.72f);

            // Angular velocity: 0.5 / 2 = 0.25
            float angVel = obs[baseIdx + 7];
            Assert.Equal(0.25f, angVel, 3);
        }
    }

    [Fact]
    public void GetObservationsFromBodies_ReturnsZerosForTerminalWorlds()
    {
        // Arrange
        const int worldCount = 2;
        const int observationsPerWorld = 8;
        const int bodiesPerWorld = 3;
        const int targetsPerWorld = 2;

        using var observations = _accelerator.Allocate1D<float>(worldCount * observationsPerWorld);
        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var targetPositions = _accelerator.Allocate1D<float>(worldCount * targetsPerWorld * 2);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // Set up bodies with non-zero state
        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        for (int w = 0; w < worldCount; w++)
        {
            bodyData[w * bodiesPerWorld] = new GPURigidBody
            {
                X = 5f, Y = 5f, VelX = 2f, VelY = 3f, AngularVel = 1f
            };
        }
        bodies.CopyFromCPU(bodyData);

        // Mark world 1 as terminal
        isTerminal.CopyFromCPU(new byte[] { 0, 1 });
        targetsCollected.MemSetToZero();

        var targetPos = new float[worldCount * targetsPerWorld * 2];
        targetPos[0] = 10f; targetPos[1] = 10f;
        targetPositions.CopyFromCPU(targetPos);

        // Act
        _getObservationsFromBodiesKernel(
            worldCount,
            observations.View,
            bodies.View,
            targetPositions.View,
            targetsCollected.View,
            isTerminal.View,
            observationsPerWorld,
            bodiesPerWorld,
            targetsPerWorld,
            0,      // primaryBodyLocalIdx
            10f,    // velocityNormalization
            20f,    // distanceNormalization
            2f);    // angularVelNormalization
        _accelerator.Synchronize();

        // Assert
        var obs = observations.GetAsArray1D();

        // World 0 should have non-zero observations
        bool hasNonZeroWorld0 = false;
        for (int i = 0; i < observationsPerWorld; i++)
        {
            if (obs[i] != 0f) hasNonZeroWorld0 = true;
        }
        Assert.True(hasNonZeroWorld0);

        // World 1 (terminal) should have all zeros
        for (int i = observationsPerWorld; i < observationsPerWorld * 2; i++)
        {
            Assert.Equal(0f, obs[i]);
        }
    }

    #endregion

    #region Actions Tests

    [Fact]
    public void ApplyActions_AppliesThrustAndGimbalCorrectly()
    {
        // Arrange
        const int worldCount = 2;
        const int actionsPerWorld = 2;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var actions = _accelerator.Allocate1D<float>(worldCount * actionsPerWorld);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // Set up bodies with invMass and invInertia
        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        for (int w = 0; w < worldCount; w++)
        {
            bodyData[w * bodiesPerWorld] = new GPURigidBody
            {
                X = 0f, Y = 0f,
                Angle = 0f,  // Pointing up
                VelX = 0f, VelY = 0f,
                AngularVel = 0f,
                InvMass = 1f,      // 1 kg
                InvInertia = 1f    // 1 kg*m^2
            };
        }
        bodies.CopyFromCPU(bodyData);

        // Set actions: throttle=1 (maps from [-1,1] -> [0,1], so input 1 -> throttle 1)
        // gimbal=0.5
        var actionData = new float[worldCount * actionsPerWorld];
        actionData[0] = 1f;    // World 0 throttle (input) -> maps to 1.0
        actionData[1] = 0.5f;  // World 0 gimbal
        actionData[2] = -1f;   // World 1 throttle (input) -> maps to 0.0
        actionData[3] = -0.5f; // World 1 gimbal
        actions.CopyFromCPU(actionData);

        isTerminal.MemSetToZero();

        float dt = 1f / 60f;
        float maxThrust = 100f;
        float maxGimbalTorque = 10f;

        // Act
        // Parameters: bodies, actions, isTerminal, actionsPerWorld, bodiesPerWorld, primaryBodyLocalIdx,
        //             maxThrust, maxGimbalTorque, dt
        _applyActionsKernel(
            worldCount,
            bodies.View,
            actions.View,
            isTerminal.View,
            actionsPerWorld,
            bodiesPerWorld,
            0,              // primaryBodyLocalIdx
            maxThrust,
            maxGimbalTorque,
            dt);
        _accelerator.Synchronize();

        // Assert
        var updatedBodies = bodies.GetAsArray1D();

        // World 0: full throttle pointing up -> velY should increase
        // dVelY = thrust * invMass * dt = 100 * 1 * (1/60) = 1.67
        Assert.True(updatedBodies[0].VelY > 1.6f);
        Assert.Equal(0f, updatedBodies[0].VelX, 3);  // No lateral thrust

        // World 0: gimbal 0.5 -> angular velocity change
        // dAngVel = 0.5 * 10 * 1 * dt = 0.083
        Assert.True(updatedBodies[0].AngularVel > 0.08f);

        // World 1: zero throttle (input -1 -> 0) -> no velocity change
        Assert.Equal(0f, updatedBodies[bodiesPerWorld].VelY, 3);

        // World 1: gimbal -0.5 -> negative angular velocity
        Assert.True(updatedBodies[bodiesPerWorld].AngularVel < -0.08f);
    }

    [Fact]
    public void ApplyActions_SkipsTerminalWorlds()
    {
        // Arrange
        const int worldCount = 2;
        const int actionsPerWorld = 2;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var actions = _accelerator.Allocate1D<float>(worldCount * actionsPerWorld);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        for (int w = 0; w < worldCount; w++)
        {
            bodyData[w * bodiesPerWorld] = new GPURigidBody
            {
                VelX = 0f, VelY = 0f, AngularVel = 0f,
                InvMass = 1f, InvInertia = 1f
            };
        }
        bodies.CopyFromCPU(bodyData);

        var actionData = new float[worldCount * actionsPerWorld];
        actionData[0] = 1f; actionData[1] = 1f;  // World 0
        actionData[2] = 1f; actionData[3] = 1f;  // World 1
        actions.CopyFromCPU(actionData);

        // Mark world 1 as terminal
        isTerminal.CopyFromCPU(new byte[] { 0, 1 });

        // Act
        _applyActionsKernel(
            worldCount,
            bodies.View,
            actions.View,
            isTerminal.View,
            actionsPerWorld,
            bodiesPerWorld,
            0,              // primaryBodyLocalIdx
            100f,           // maxThrust
            10f,            // maxGimbalTorque
            1f / 60f);      // dt
        _accelerator.Synchronize();

        // Assert
        var updatedBodies = bodies.GetAsArray1D();

        // World 0 should have changed
        Assert.True(updatedBodies[0].VelY > 0f || updatedBodies[0].AngularVel > 0f);

        // World 1 (terminal) should be unchanged
        Assert.Equal(0f, updatedBodies[bodiesPerWorld].VelY);
        Assert.Equal(0f, updatedBodies[bodiesPerWorld].AngularVel);
    }

    [Fact]
    public void ApplyActions_ClampsThrustAndGimbal()
    {
        // Arrange
        const int worldCount = 1;
        const int actionsPerWorld = 2;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var actions = _accelerator.Allocate1D<float>(worldCount * actionsPerWorld);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        var bodyData = new GPURigidBody[bodiesPerWorld];
        bodyData[0] = new GPURigidBody
        {
            Angle = 0f,
            VelX = 0f, VelY = 0f, AngularVel = 0f,
            InvMass = 1f, InvInertia = 1f
        };
        bodies.CopyFromCPU(bodyData);

        // Out of range actions
        actions.CopyFromCPU(new float[] { 5f, 10f });  // Should be clamped
        isTerminal.MemSetToZero();

        // Act
        _applyActionsKernel(
            worldCount,
            bodies.View,
            actions.View,
            isTerminal.View,
            actionsPerWorld,
            bodiesPerWorld,
            0,              // primaryBodyLocalIdx
            100f,           // maxThrust
            10f,            // maxGimbalTorque
            1f / 60f);      // dt
        _accelerator.Synchronize();

        // Assert - values should be clamped
        var updatedBodies = bodies.GetAsArray1D();

        // Throttle 5 -> clamp to 1, then map to [0,1] -> 1.0 -> thrust = 100
        // VelY += 100 * 1 * (1/60) = 1.67
        Assert.InRange(updatedBodies[0].VelY, 1.6f, 1.7f);

        // Gimbal 10 -> clamp to 1 -> torque = 10
        // AngVel += 10 * 1 * (1/60) = 0.167
        Assert.InRange(updatedBodies[0].AngularVel, 0.16f, 0.17f);
    }

    #endregion

    #region Terminal Conditions Tests

    [Fact]
    public void CheckTerminalConditions_DetectsOutOfBounds()
    {
        // Arrange
        const int worldCount = 3;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var stepCounters = _accelerator.Allocate1D<int>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);

        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        // World 0: in bounds
        bodyData[0] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        // World 1: out of bounds (X too low)
        bodyData[bodiesPerWorld] = new GPURigidBody { X = -15f, Y = 0f, Angle = 0f };
        // World 2: out of bounds (Y too high)
        bodyData[2 * bodiesPerWorld] = new GPURigidBody { X = 0f, Y = 15f, Angle = 0f };
        bodies.CopyFromCPU(bodyData);

        stepCounters.MemSetToZero();
        isTerminal.MemSetToZero();
        cumulativeRewards.MemSetToZero();

        // Act
        // Parameters: bodies, stepCounters, isTerminal, cumulativeRewards,
        //             bodiesPerWorld, primaryBodyLocalIdx, maxSteps,
        //             arenaMinX, arenaMaxX, arenaMinY, arenaMaxY,
        //             crashAngleThreshold, outOfBoundsPenalty, crashPenalty
        _checkTerminalConditionsKernel(
            worldCount,
            bodies.View,
            stepCounters.View,
            isTerminal.View,
            cumulativeRewards.View,
            bodiesPerWorld,
            0,                      // primaryBodyLocalIdx
            1000,                   // maxSteps
            -10f,                   // arenaMinX
            10f,                    // arenaMaxX
            -5f,                    // arenaMinY
            10f,                    // arenaMaxY
            MathF.PI * 0.7f,        // crashAngleThreshold
            -50f,                   // outOfBoundsPenalty
            -100f);                 // crashPenalty
        _accelerator.Synchronize();

        // Assert
        var terminal = isTerminal.GetAsArray1D();
        var rewards = cumulativeRewards.GetAsArray1D();

        Assert.Equal(0, terminal[0]);  // World 0 still active
        Assert.Equal(1, terminal[1]);  // World 1 out of bounds
        Assert.Equal(1, terminal[2]);  // World 2 out of bounds

        Assert.Equal(0f, rewards[0]);   // No penalty
        Assert.Equal(-50f, rewards[1]); // Out of bounds penalty
        Assert.Equal(-50f, rewards[2]); // Out of bounds penalty
    }

    [Fact]
    public void CheckTerminalConditions_DetectsCrash()
    {
        // Arrange
        const int worldCount = 2;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var stepCounters = _accelerator.Allocate1D<int>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);

        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        // World 0: upright (angle = 0)
        bodyData[0] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        // World 1: flipped (angle = PI, pointing down)
        bodyData[bodiesPerWorld] = new GPURigidBody { X = 0f, Y = 0f, Angle = MathF.PI };
        bodies.CopyFromCPU(bodyData);

        stepCounters.MemSetToZero();
        isTerminal.MemSetToZero();
        cumulativeRewards.MemSetToZero();

        // Act - crash threshold at 70% of PI (~126 degrees)
        _checkTerminalConditionsKernel(
            worldCount,
            bodies.View,
            stepCounters.View,
            isTerminal.View,
            cumulativeRewards.View,
            bodiesPerWorld,
            0,                      // primaryBodyLocalIdx
            1000,                   // maxSteps
            -10f,                   // arenaMinX
            10f,                    // arenaMaxX
            -5f,                    // arenaMinY
            10f,                    // arenaMaxY
            MathF.PI * 0.7f,        // crashAngleThreshold
            -50f,                   // outOfBoundsPenalty
            -100f);                 // crashPenalty
        _accelerator.Synchronize();

        // Assert
        var terminal = isTerminal.GetAsArray1D();
        var rewards = cumulativeRewards.GetAsArray1D();

        Assert.Equal(0, terminal[0]);    // World 0 upright
        Assert.Equal(1, terminal[1]);    // World 1 crashed

        Assert.Equal(0f, rewards[0]);
        Assert.Equal(-100f, rewards[1]); // Crash penalty
    }

    [Fact]
    public void CheckTerminalConditions_DetectsMaxSteps()
    {
        // Arrange
        const int worldCount = 2;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var stepCounters = _accelerator.Allocate1D<int>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);

        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        bodyData[0] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        bodyData[bodiesPerWorld] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        bodies.CopyFromCPU(bodyData);

        // Set step counters: world 0 at 99, world 1 at 0
        stepCounters.CopyFromCPU(new int[] { 99, 0 });
        isTerminal.MemSetToZero();
        cumulativeRewards.MemSetToZero();

        // Act - max steps = 100, so after incrementing world 0 will hit limit
        _checkTerminalConditionsKernel(
            worldCount,
            bodies.View,
            stepCounters.View,
            isTerminal.View,
            cumulativeRewards.View,
            bodiesPerWorld,
            0,                      // primaryBodyLocalIdx
            100,                    // maxSteps
            -10f,                   // arenaMinX
            10f,                    // arenaMaxX
            -5f,                    // arenaMinY
            10f,                    // arenaMaxY
            MathF.PI * 0.7f,        // crashAngleThreshold
            -50f,                   // outOfBoundsPenalty
            -100f);                 // crashPenalty
        _accelerator.Synchronize();

        // Assert
        var terminal = isTerminal.GetAsArray1D();
        var rewards = cumulativeRewards.GetAsArray1D();
        var steps = stepCounters.GetAsArray1D();

        Assert.Equal(1, terminal[0]);  // World 0 hit max steps
        Assert.Equal(0, terminal[1]);  // World 1 still running

        Assert.Equal(100, steps[0]);
        Assert.Equal(1, steps[1]);

        // No penalty for timeout
        Assert.Equal(0f, rewards[0]);
        Assert.Equal(0f, rewards[1]);
    }

    [Fact]
    public void CheckTerminalConditions_SkipsAlreadyTerminalWorlds()
    {
        // Arrange
        const int worldCount = 2;
        const int bodiesPerWorld = 3;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var stepCounters = _accelerator.Allocate1D<int>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);

        // Both out of bounds, but world 1 already terminal
        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        bodyData[0] = new GPURigidBody { X = -15f, Y = 0f, Angle = 0f };
        bodyData[bodiesPerWorld] = new GPURigidBody { X = -15f, Y = 0f, Angle = 0f };
        bodies.CopyFromCPU(bodyData);

        stepCounters.CopyFromCPU(new int[] { 5, 10 });
        isTerminal.CopyFromCPU(new byte[] { 0, 1 });  // World 1 already terminal
        cumulativeRewards.MemSetToZero();

        // Act
        _checkTerminalConditionsKernel(
            worldCount,
            bodies.View,
            stepCounters.View,
            isTerminal.View,
            cumulativeRewards.View,
            bodiesPerWorld,
            0,                      // primaryBodyLocalIdx
            1000,                   // maxSteps
            -10f,                   // arenaMinX
            10f,                    // arenaMaxX
            -5f,                    // arenaMinY
            10f,                    // arenaMaxY
            MathF.PI * 0.7f,        // crashAngleThreshold
            -50f,                   // outOfBoundsPenalty
            -100f);                 // crashPenalty
        _accelerator.Synchronize();

        // Assert
        var steps = stepCounters.GetAsArray1D();
        var rewards = cumulativeRewards.GetAsArray1D();

        Assert.Equal(6, steps[0]);   // World 0 incremented
        Assert.Equal(10, steps[1]);  // World 1 not incremented (was terminal)

        Assert.Equal(-50f, rewards[0]);  // World 0 got penalty
        Assert.Equal(0f, rewards[1]);    // World 1 no new penalty
    }

    #endregion

    #region Fitness Computation Tests

    [Fact]
    public void ComputeFitness_CombinesTargetsAndRewards()
    {
        // Arrange
        const int worldCount = 3;

        using var fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);

        cumulativeRewards.CopyFromCPU(new float[] { 50f, -25f, 100f });
        targetsCollected.CopyFromCPU(new int[] { 2, 0, 5 });

        // Act
        // Parameters: fitnessValues, cumulativeRewards, targetsCollected, targetBonusMultiplier
        _computeFitnessKernel(
            worldCount,
            fitnessValues.View,
            cumulativeRewards.View,
            targetsCollected.View,
            200f);  // targetBonusMultiplier
        _accelerator.Synchronize();

        // Assert
        var fitness = fitnessValues.GetAsArray1D();

        // World 0: 2 * 200 + 50 = 450
        Assert.Equal(450f, fitness[0], 3);

        // World 1: 0 * 200 + (-25) = -25
        Assert.Equal(-25f, fitness[1], 3);

        // World 2: 5 * 200 + 100 = 1100
        Assert.Equal(1100f, fitness[2], 3);
    }

    [Fact]
    public void ComputeFitness_HandlesZeroTargetsAndRewards()
    {
        // Arrange
        const int worldCount = 2;

        using var fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);

        cumulativeRewards.MemSetToZero();
        targetsCollected.MemSetToZero();

        // Act
        _computeFitnessKernel(
            worldCount,
            fitnessValues.View,
            cumulativeRewards.View,
            targetsCollected.View,
            100f);  // targetBonusMultiplier
        _accelerator.Synchronize();

        // Assert
        var fitness = fitnessValues.GetAsArray1D();

        Assert.Equal(0f, fitness[0]);
        Assert.Equal(0f, fitness[1]);
    }

    #endregion

    #region Shaping Reward Tests

    [Fact]
    public void ComputeShapingReward_RewardsProximityToTarget()
    {
        // Arrange
        const int worldCount = 2;
        const int bodiesPerWorld = 3;
        const int targetsPerWorld = 2;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var targetPositions = _accelerator.Allocate1D<float>(worldCount * targetsPerWorld * 2);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        // World 0: close to target (dist=5)
        // World 1: far from target (dist=15)
        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        bodyData[0] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };  // Pointing up
        bodyData[bodiesPerWorld] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        bodies.CopyFromCPU(bodyData);

        var targetPos = new float[worldCount * targetsPerWorld * 2];
        // World 0 target at (3, 4) -> dist = 5
        targetPos[0] = 3f; targetPos[1] = 4f;
        // World 1 target at (9, 12) -> dist = 15
        targetPos[4] = 9f; targetPos[5] = 12f;
        targetPositions.CopyFromCPU(targetPos);

        targetsCollected.MemSetToZero();
        cumulativeRewards.MemSetToZero();
        isTerminal.MemSetToZero();

        // Act
        // Parameters: bodies, targetPositions, targetsCollected, cumulativeRewards, isTerminal,
        //             bodiesPerWorld, targetsPerWorld, primaryBodyLocalIdx,
        //             distanceRewardScale, orientationRewardScale, timeStepPenalty
        _computeShapingRewardKernel(
            worldCount,
            bodies.View,
            targetPositions.View,
            targetsCollected.View,
            cumulativeRewards.View,
            isTerminal.View,
            bodiesPerWorld,
            targetsPerWorld,
            0,      // primaryBodyLocalIdx
            0.1f,   // distanceRewardScale
            0.5f,   // orientationRewardScale
            -0.05f);// timeStepPenalty
        _accelerator.Synchronize();

        // Assert
        var rewards = cumulativeRewards.GetAsArray1D();

        // World 0 closer should have higher reward than World 1
        Assert.True(rewards[0] > rewards[1],
            $"Closer world should have higher reward. World0: {rewards[0]}, World1: {rewards[1]}");
    }

    [Fact]
    public void ComputeShapingReward_RewardsPointingAtTarget()
    {
        // Arrange
        const int worldCount = 2;
        const int bodiesPerWorld = 3;
        const int targetsPerWorld = 2;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var targetPositions = _accelerator.Allocate1D<float>(worldCount * targetsPerWorld * 2);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        // World 0: pointing up (angle=0), target is above
        bodyData[0] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        // World 1: pointing down (angle=PI), target is above
        bodyData[bodiesPerWorld] = new GPURigidBody { X = 0f, Y = 0f, Angle = MathF.PI };
        bodies.CopyFromCPU(bodyData);

        // Target above at (0, 10) for both
        var targetPos = new float[worldCount * targetsPerWorld * 2];
        targetPos[0] = 0f; targetPos[1] = 10f;
        targetPos[4] = 0f; targetPos[5] = 10f;
        targetPositions.CopyFromCPU(targetPos);

        targetsCollected.MemSetToZero();
        cumulativeRewards.MemSetToZero();
        isTerminal.MemSetToZero();

        // Act
        _computeShapingRewardKernel(
            worldCount,
            bodies.View,
            targetPositions.View,
            targetsCollected.View,
            cumulativeRewards.View,
            isTerminal.View,
            bodiesPerWorld,
            targetsPerWorld,
            0,      // primaryBodyLocalIdx
            0.0f,   // distanceRewardScale (disable distance reward)
            1.0f,   // orientationRewardScale (only orientation reward)
            0f);    // timeStepPenalty
        _accelerator.Synchronize();

        // Assert
        var rewards = cumulativeRewards.GetAsArray1D();

        // World 0 pointing at target should have positive orientation reward
        // World 1 pointing away should have negative orientation reward
        Assert.True(rewards[0] > rewards[1],
            $"World pointing at target should have higher reward. World0: {rewards[0]}, World1: {rewards[1]}");
        Assert.True(rewards[0] > 0f);
        Assert.True(rewards[1] < 0f);
    }

    [Fact]
    public void ComputeShapingReward_SkipsTerminalWorlds()
    {
        // Arrange
        const int worldCount = 2;
        const int bodiesPerWorld = 3;
        const int targetsPerWorld = 2;

        using var bodies = _accelerator.Allocate1D<GPURigidBody>(worldCount * bodiesPerWorld);
        using var targetPositions = _accelerator.Allocate1D<float>(worldCount * targetsPerWorld * 2);
        using var targetsCollected = _accelerator.Allocate1D<int>(worldCount);
        using var cumulativeRewards = _accelerator.Allocate1D<float>(worldCount);
        using var isTerminal = _accelerator.Allocate1D<byte>(worldCount);

        var bodyData = new GPURigidBody[worldCount * bodiesPerWorld];
        bodyData[0] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        bodyData[bodiesPerWorld] = new GPURigidBody { X = 0f, Y = 0f, Angle = 0f };
        bodies.CopyFromCPU(bodyData);

        var targetPos = new float[worldCount * targetsPerWorld * 2];
        targetPos[0] = 5f; targetPos[1] = 5f;
        targetPos[4] = 5f; targetPos[5] = 5f;
        targetPositions.CopyFromCPU(targetPos);

        targetsCollected.MemSetToZero();
        cumulativeRewards.MemSetToZero();
        isTerminal.CopyFromCPU(new byte[] { 0, 1 });  // World 1 terminal

        // Act
        _computeShapingRewardKernel(
            worldCount,
            bodies.View,
            targetPositions.View,
            targetsCollected.View,
            cumulativeRewards.View,
            isTerminal.View,
            bodiesPerWorld,
            targetsPerWorld,
            0,      // primaryBodyLocalIdx
            0.1f,   // distanceRewardScale
            0.5f,   // orientationRewardScale
            -0.1f); // timeStepPenalty
        _accelerator.Synchronize();

        // Assert
        var rewards = cumulativeRewards.GetAsArray1D();

        Assert.NotEqual(0f, rewards[0]);  // World 0 should have reward
        Assert.Equal(0f, rewards[1]);     // World 1 (terminal) should be unchanged
    }

    #endregion

    public void Dispose()
    {
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
