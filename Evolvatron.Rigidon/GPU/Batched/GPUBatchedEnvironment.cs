using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// High-level orchestration class for batched GPU environment simulation.
/// Ties together physics (GPUBatchedWorldState) and environment (GPUBatchedEnvironmentState)
/// for easy use in evolutionary/RL training loops.
///
/// This class provides a simple interface for:
/// - Resetting all worlds with new random targets
/// - Extracting observations for neural network input
/// - Applying neural network actions to rockets
/// - Stepping the simulation and computing rewards
/// - Checking terminal conditions
/// - Retrieving final fitness values
///
/// Typical usage:
/// <code>
/// using var env = new GPUBatchedEnvironment(accelerator, worldConfig, envConfig);
/// env.UploadRocketTemplate(bodies, geoms, joints);
/// env.UploadSharedColliders(walls);
///
/// env.Reset(seed);
/// while (!env.AllTerminal())
/// {
///     var observations = env.GetObservations();
///     // Run neural network on observations
///     env.UploadActions(neuralNetOutputs);
///     env.ApplyActions();
///     env.PhysicsStep(simConfig);
///     env.EnvironmentStep();
/// }
/// var fitness = env.GetFitness();
/// </code>
/// </summary>
public class GPUBatchedEnvironment : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly GPUBatchedWorldConfig _worldConfig;
    private readonly GPUBatchedEnvironmentConfig _envConfig;
    private bool _disposed;

    // State managers
    public GPUBatchedWorldState WorldState { get; }
    public GPUBatchedEnvironmentState EnvironmentState { get; }

    // Physics stepper
    public GPUBatchedStepper Stepper { get; }

    // Environment kernels
    private Action<Index1D, ArrayView<float>, ArrayView<byte>, int, int, float, float, float, float, float>
        _initializeTargetsKernel = null!;

    private Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<byte>,
        ArrayView<int>, ArrayView<float>>
        _resetEnvironmentKernel = null!;

    private Action<Index1D, ArrayView<byte>, ArrayView<int>, int>
        _resetTargetsKernel = null!;

    private Action<Index1D, ArrayView<float>, ArrayView<GPURigidBody>, ArrayView<float>,
        ArrayView<int>, ArrayView<byte>, int, int, int, int, float, float, float>
        _getObservationsKernel = null!;

    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<byte>,
        int, int, int, float, float, float>
        _applyActionsKernel = null!;

    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<int>, ArrayView<byte>,
        ArrayView<float>, int, int, int, float, float, float, float, float, float, float>
        _checkTerminalConditionsKernel = null!;

    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<int>,
        ArrayView<float>, ArrayView<byte>, int, int, int, float, float, float>
        _computeShapingRewardKernel = null!;

    private Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, float>
        _computeFitnessKernel = null!;

    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<byte>,
        ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, int, int, float, float>
        _checkTargetCollisionsKernel = null!;

    // Cached host arrays for download
    private float[]? _observationsCache;
    private float[]? _fitnessCache;
    private byte[]? _terminalCache;
    private int[]? _resetFlagsBuffer;

    /// <summary>
    /// Creates a new batched GPU environment.
    /// </summary>
    /// <param name="accelerator">The ILGPU accelerator to use for kernel execution.</param>
    /// <param name="worldConfig">Configuration for physics world (bodies, geoms, contacts).</param>
    /// <param name="envConfig">Configuration for environment (observations, actions, rewards).</param>
    public GPUBatchedEnvironment(
        Accelerator accelerator,
        GPUBatchedWorldConfig worldConfig,
        GPUBatchedEnvironmentConfig envConfig)
    {
        _accelerator = accelerator;
        _worldConfig = worldConfig;
        _envConfig = envConfig;

        // Ensure environment config world count matches world config
        if (_envConfig.WorldCount != worldConfig.WorldCount)
        {
            _envConfig.WorldCount = worldConfig.WorldCount;
        }

        // Create state managers
        WorldState = new GPUBatchedWorldState(accelerator, worldConfig);
        EnvironmentState = new GPUBatchedEnvironmentState(
            accelerator,
            worldConfig.WorldCount,
            worldConfig.TargetsPerWorld,
            envConfig.ObservationsPerWorld,
            envConfig.ActionsPerWorld);

        // Create physics stepper
        Stepper = new GPUBatchedStepper(accelerator);

        // Load environment kernels
        LoadKernels();

        // Allocate host caches
        _observationsCache = new float[envConfig.TotalObservations];
        _fitnessCache = new float[worldConfig.WorldCount];
        _terminalCache = new byte[worldConfig.WorldCount];
        _resetFlagsBuffer = new int[worldConfig.WorldCount];
    }

    private void LoadKernels()
    {
        // Initialize targets kernel
        _initializeTargetsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<byte>, int, int, float, float, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedInitializeTargetsKernel);

        // Reset environment kernel
        _resetEnvironmentKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<byte>,
            ArrayView<int>, ArrayView<float>>(
            GPUBatchedEnvironmentKernels.BatchedResetEnvironmentKernel);

        // Reset targets kernel
        _resetTargetsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, int>(
            GPUBatchedEnvironmentKernels.BatchedResetTargetsKernel);

        // Get observations from bodies kernel
        _getObservationsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<GPURigidBody>, ArrayView<float>,
            ArrayView<int>, ArrayView<byte>, int, int, int, int, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedGetObservationsFromBodiesKernel);

        // Apply actions kernel
        _applyActionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<byte>,
            int, int, int, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedApplyActionsKernel);

        // Check target collisions kernel (wrapper needed for per-world rocket position)
        // Note: The original kernel requires per-world rocket position, so we need a modified approach
        // For now, we'll use a kernel that reads rocket position from bodies array
        // This will require adding a new kernel or modifying the existing one

        // Check terminal conditions kernel
        _checkTerminalConditionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<int>, ArrayView<byte>,
            ArrayView<float>, int, int, int, float, float, float, float, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedCheckTerminalConditionsKernel);

        // Compute shaping reward kernel
        _computeShapingRewardKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<int>,
            ArrayView<float>, ArrayView<byte>, int, int, int, float, float, float>(
            GPUBatchedEnvironmentKernels.BatchedComputeShapingRewardKernel);

        // Compute fitness kernel
        _computeFitnessKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, float>(
            GPUBatchedEnvironmentKernels.BatchedComputeFitnessKernel);

        // Target collision kernel (reads rocket pos from bodies array)
        _checkTargetCollisionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<float>, ArrayView<byte>,
            ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, int, int, float, float>(
            GPUBatchedEnvironmentKernels.BatchedCheckTargetCollisionsFromBodiesKernel);
    }

    #region Setup Methods

    /// <summary>
    /// Upload a rocket template to all worlds (same initial state).
    /// </summary>
    public void UploadRocketTemplate(
        GPURigidBody[] templateBodies,
        GPURigidBodyGeom[] templateGeoms,
        GPURevoluteJoint[] templateJoints)
    {
        WorldState.UploadRocketTemplate(templateBodies, templateGeoms, templateJoints);
    }

    /// <summary>
    /// Upload shared static colliders (arena walls).
    /// </summary>
    public void UploadSharedColliders(GPUOBBCollider[] colliders)
    {
        WorldState.UploadSharedColliders(colliders);
    }

    #endregion

    #region Episode Control

    /// <summary>
    /// Reset all worlds with new random targets.
    /// </summary>
    /// <param name="baseSeed">Base seed for random number generation. Each world gets a different derived seed.</param>
    public void Reset(int baseSeed)
    {
        int worldCount = _worldConfig.WorldCount;
        int targetsPerWorld = _worldConfig.TargetsPerWorld;
        int totalTargets = worldCount * targetsPerWorld;

        // Reset environment state
        EnvironmentState.ResetAll();

        // Initialize targets with random positions
        float margin = 2.0f; // Margin from arena edges
        _initializeTargetsKernel(
            totalTargets,
            EnvironmentState.TargetPositions.View,
            EnvironmentState.TargetActive.View,
            targetsPerWorld,
            baseSeed,
            _envConfig.ArenaMinX,
            _envConfig.ArenaMaxX,
            _envConfig.ArenaMinY,
            _envConfig.ArenaMaxY,
            margin);

        // Clear contact counts
        WorldState.ClearContactCounts();

        _accelerator.Synchronize();
    }

    /// <summary>
    /// Reset specific worlds that have terminated.
    /// Useful for continuous training where we want to restart only finished episodes.
    /// </summary>
    /// <param name="baseSeed">Base seed for new random targets.</param>
    /// <param name="resetFlags">Array of flags indicating which worlds to reset (1=reset, 0=keep).</param>
    public void ResetTerminatedWorlds(int baseSeed, int[] resetFlags)
    {
        if (resetFlags.Length != _worldConfig.WorldCount)
        {
            throw new ArgumentException($"resetFlags length ({resetFlags.Length}) must match world count ({_worldConfig.WorldCount})");
        }

        // Upload reset flags to temporary buffer
        using var resetFlagsBuffer = _accelerator.Allocate1D<int>(resetFlags.Length);
        resetFlagsBuffer.CopyFromCPU(resetFlags);

        int worldCount = _worldConfig.WorldCount;
        int targetsPerWorld = _worldConfig.TargetsPerWorld;
        int totalTargets = worldCount * targetsPerWorld;

        // Reset environment state for flagged worlds
        _resetEnvironmentKernel(
            worldCount,
            resetFlagsBuffer.View,
            EnvironmentState.CumulativeRewards.View,
            EnvironmentState.StepCounters.View,
            EnvironmentState.IsTerminal.View,
            EnvironmentState.TargetsCollected.View,
            EnvironmentState.FitnessValues.View);

        // Reset target active flags for flagged worlds
        _resetTargetsKernel(
            totalTargets,
            EnvironmentState.TargetActive.View,
            resetFlagsBuffer.View,
            targetsPerWorld);

        // Re-initialize targets for reset worlds
        float margin = 2.0f;
        _initializeTargetsKernel(
            totalTargets,
            EnvironmentState.TargetPositions.View,
            EnvironmentState.TargetActive.View,
            targetsPerWorld,
            baseSeed,
            _envConfig.ArenaMinX,
            _envConfig.ArenaMaxX,
            _envConfig.ArenaMinY,
            _envConfig.ArenaMaxY,
            margin);

        _accelerator.Synchronize();
    }

    #endregion

    #region Observation/Action Interface

    /// <summary>
    /// Extract observations for all worlds.
    /// Returns a view of the GPU observations buffer. Call Synchronize() first if needed.
    /// </summary>
    /// <returns>ArrayView to observations buffer on GPU.</returns>
    public ArrayView<float> GetObservationsView()
    {
        int worldCount = _worldConfig.WorldCount;
        int bodiesPerWorld = _worldConfig.RigidBodiesPerWorld;
        int targetsPerWorld = _worldConfig.TargetsPerWorld;
        int observationsPerWorld = _envConfig.ObservationsPerWorld;
        int primaryBodyLocalIdx = 0; // Main rocket body is at index 0

        // Default normalization values
        float velocityNormalization = 20f;
        float distanceNormalization = 30f;
        float angularVelNormalization = 10f;

        _getObservationsKernel(
            worldCount,
            EnvironmentState.Observations.View,
            WorldState.RigidBodies.View,
            EnvironmentState.TargetPositions.View,
            EnvironmentState.TargetsCollected.View,
            EnvironmentState.IsTerminal.View,
            observationsPerWorld,
            bodiesPerWorld,
            targetsPerWorld,
            primaryBodyLocalIdx,
            velocityNormalization,
            distanceNormalization,
            angularVelNormalization);

        return EnvironmentState.Observations.View;
    }

    /// <summary>
    /// Extract observations and download to host memory.
    /// </summary>
    /// <returns>Host array containing observations for all worlds.</returns>
    public float[] GetObservations()
    {
        GetObservationsView();
        _accelerator.Synchronize();

        EnvironmentState.Observations.CopyToCPU(_observationsCache!);
        return _observationsCache!;
    }

    /// <summary>
    /// Upload actions from host to GPU actions buffer.
    /// </summary>
    /// <param name="actions">Actions array with layout [world0_action0, world0_action1, world1_action0, ...]</param>
    public void UploadActions(float[] actions)
    {
        if (actions.Length != _envConfig.TotalActions)
        {
            throw new ArgumentException(
                $"Actions length ({actions.Length}) must match total actions ({_envConfig.TotalActions})");
        }

        EnvironmentState.Actions.CopyFromCPU(actions);
    }

    /// <summary>
    /// Get direct view of the actions buffer for GPU-to-GPU transfers.
    /// </summary>
    public ArrayView<float> ActionsView => EnvironmentState.Actions.View;

    /// <summary>
    /// Apply actions from the actions buffer to rocket bodies.
    /// </summary>
    /// <param name="maxThrust">Maximum thrust force (Newtons).</param>
    /// <param name="maxGimbalTorque">Maximum gimbal torque (N*m).</param>
    /// <param name="dt">Timestep for impulse calculation.</param>
    public void ApplyActions(float maxThrust = 100f, float maxGimbalTorque = 20f, float dt = 1f / 240f)
    {
        int worldCount = _worldConfig.WorldCount;
        int bodiesPerWorld = _worldConfig.RigidBodiesPerWorld;
        int actionsPerWorld = _envConfig.ActionsPerWorld;
        int primaryBodyLocalIdx = 0;

        _applyActionsKernel(
            worldCount,
            WorldState.RigidBodies.View,
            EnvironmentState.Actions.View,
            EnvironmentState.IsTerminal.View,
            actionsPerWorld,
            bodiesPerWorld,
            primaryBodyLocalIdx,
            maxThrust,
            maxGimbalTorque,
            dt);
    }

    #endregion

    #region Simulation Step

    /// <summary>
    /// Step physics simulation for all worlds.
    /// </summary>
    /// <param name="simConfig">Simulation configuration (timestep, iterations, etc.).</param>
    public void PhysicsStep(SimulationConfig simConfig)
    {
        Stepper.Step(WorldState, simConfig);
    }

    /// <summary>
    /// Step environment logic: check collisions, compute rewards, check terminal conditions.
    /// Call this after PhysicsStep.
    /// </summary>
    /// <param name="distanceRewardScale">Scale factor for distance-to-target reward shaping.</param>
    /// <param name="orientationRewardScale">Scale factor for orientation reward shaping.</param>
    public void EnvironmentStep(
        float distanceRewardScale = 0.1f,
        float orientationRewardScale = 0.5f)
    {
        int worldCount = _worldConfig.WorldCount;
        int bodiesPerWorld = _worldConfig.RigidBodiesPerWorld;
        int targetsPerWorld = _worldConfig.TargetsPerWorld;
        int primaryBodyLocalIdx = 0;
        float crashAngleThreshold = 3.14159265f * 0.99f; // ~178 degrees - almost never crash

        // Check target collisions (before terminal check so we collect targets first)
        _checkTargetCollisionsKernel(
            worldCount,
            WorldState.RigidBodies.View,
            EnvironmentState.TargetPositions.View,
            EnvironmentState.TargetActive.View,
            EnvironmentState.TargetsCollected.View,
            EnvironmentState.CumulativeRewards.View,
            EnvironmentState.IsTerminal.View,
            bodiesPerWorld,
            primaryBodyLocalIdx,
            targetsPerWorld,
            _envConfig.TargetRadius,
            _envConfig.TargetHitReward);

        // Check terminal conditions (out of bounds, crashed, max steps)
        _checkTerminalConditionsKernel(
            worldCount,
            WorldState.RigidBodies.View,
            EnvironmentState.StepCounters.View,
            EnvironmentState.IsTerminal.View,
            EnvironmentState.CumulativeRewards.View,
            bodiesPerWorld,
            primaryBodyLocalIdx,
            _envConfig.MaxStepsPerEpisode,
            _envConfig.ArenaMinX,
            _envConfig.ArenaMaxX,
            _envConfig.ArenaMinY,
            _envConfig.ArenaMaxY,
            crashAngleThreshold,
            _envConfig.OutOfBoundsPenalty,
            _envConfig.CrashPenalty);

        // Compute shaping rewards (distance to target, orientation)
        _computeShapingRewardKernel(
            worldCount,
            WorldState.RigidBodies.View,
            EnvironmentState.TargetPositions.View,
            EnvironmentState.TargetsCollected.View,
            EnvironmentState.CumulativeRewards.View,
            EnvironmentState.IsTerminal.View,
            bodiesPerWorld,
            targetsPerWorld,
            primaryBodyLocalIdx,
            distanceRewardScale,
            orientationRewardScale,
            _envConfig.TimeStepPenalty);
    }

    /// <summary>
    /// Combined step: apply actions, physics step, environment step.
    /// Convenience method for typical training loop.
    /// </summary>
    public void Step(SimulationConfig simConfig, float maxThrust = 100f, float maxGimbalTorque = 20f)
    {
        ApplyActions(maxThrust, maxGimbalTorque, simConfig.Dt);
        PhysicsStep(simConfig);
        EnvironmentStep();
    }

    #endregion

    #region Terminal/Fitness Queries

    /// <summary>
    /// Check if all worlds have terminated.
    /// </summary>
    public bool AllTerminal()
    {
        _accelerator.Synchronize();
        return EnvironmentState.AllTerminal();
    }

    /// <summary>
    /// Download terminal flags for all worlds.
    /// </summary>
    public byte[] GetTerminalFlags()
    {
        _accelerator.Synchronize();
        EnvironmentState.IsTerminal.CopyToCPU(_terminalCache!);
        return _terminalCache!;
    }

    /// <summary>
    /// Compute final fitness values from cumulative rewards and targets collected.
    /// Call this after all episodes have terminated.
    /// </summary>
    /// <param name="targetBonusMultiplier">Bonus multiplier per target collected.</param>
    public void ComputeFitness(float targetBonusMultiplier = 200f)
    {
        _computeFitnessKernel(
            _worldConfig.WorldCount,
            EnvironmentState.FitnessValues.View,
            EnvironmentState.CumulativeRewards.View,
            EnvironmentState.TargetsCollected.View,
            targetBonusMultiplier);
    }

    /// <summary>
    /// Download final fitness values for all worlds.
    /// Call ComputeFitness() first if you want accurate final values.
    /// </summary>
    public float[] GetFitness()
    {
        _accelerator.Synchronize();
        EnvironmentState.FitnessValues.CopyToCPU(_fitnessCache!);
        return _fitnessCache!;
    }

    /// <summary>
    /// Get targets collected count for all worlds.
    /// </summary>
    public int[] GetTargetsCollected()
    {
        _accelerator.Synchronize();
        return EnvironmentState.DownloadTargetsCollected();
    }

    #endregion

    #region Direct Access

    /// <summary>
    /// Download all rigid bodies (for debugging/visualization).
    /// </summary>
    public GPURigidBody[] DownloadAllBodies()
    {
        _accelerator.Synchronize();
        return WorldState.DownloadAllBodies();
    }

    /// <summary>
    /// Download rigid body state for a specific world.
    /// </summary>
    public GPURigidBody[] DownloadWorldBodies(int worldIdx)
    {
        _accelerator.Synchronize();
        return WorldState.DownloadWorldBodies(worldIdx);
    }

    /// <summary>
    /// Synchronize GPU operations. Call before downloading data.
    /// </summary>
    public void Synchronize()
    {
        _accelerator.Synchronize();
    }

    /// <summary>
    /// World configuration.
    /// </summary>
    public GPUBatchedWorldConfig WorldConfig => _worldConfig;

    /// <summary>
    /// Environment configuration.
    /// </summary>
    public GPUBatchedEnvironmentConfig EnvironmentConfig => _envConfig;

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Stepper?.Dispose();
        WorldState?.Dispose();
        EnvironmentState?.Dispose();
    }
}
