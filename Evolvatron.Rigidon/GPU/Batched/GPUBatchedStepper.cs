using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// GPU-accelerated physics stepper for N parallel worlds.
/// Each Step call advances all worlds by one timestep.
///
/// This stepper is designed for batch training of RL agents where
/// many identical simulations run in parallel with different actions.
///
/// Key differences from single-world GPUStepper:
/// - All worlds share static colliders (arena walls)
/// - All arrays are batched: [world0_body0, world0_body1, ..., world1_body0, ...]
/// - Contact detection is parallelized across all (geom, collider) pairs
/// </summary>
public class GPUBatchedStepper : IDisposable
{
    private readonly Accelerator _accelerator;
    private bool _disposed;

    // Physics kernels
    private Action<Index1D, ArrayView<GPURigidBody>, int, float, float, float> _applyGravityKernel = null!;
    private Action<Index1D, ArrayView<GPURigidBody>> _savePrevPositionsKernel = null!;
    private Action<Index1D, ArrayView<GPURigidBody>, float> _integrateKernel = null!;
    private Action<Index1D, ArrayView<GPURigidBodyGeom>, ArrayView<GPURigidBody>, int, int> _updateGeomPositionsKernel = null!;
    private Action<Index1D, ArrayView<GPURigidBody>, float, float, float> _dampVelocitiesKernel = null!;
    private Action<Index1D, ArrayView<GPURigidBody>, float, float, float> _velocityStabilizationKernel = null!;

    // Contact management kernels
    private Action<Index1D, ArrayView<int>> _clearContactCountsKernel = null!;

    // Contact detection kernels (one for each collider type)
    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>,
        ArrayView<GPUOBBCollider>, ArrayView<GPUContactConstraint>, ArrayView<int>,
        int, int, int, int, int, float, float, float> _detectOBBContactsKernel = null!;

    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>,
        ArrayView<GPUCircleCollider>, ArrayView<GPUContactConstraint>, ArrayView<int>,
        int, int, int, int, int, float, float, float> _detectCircleContactsKernel = null!;

    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>,
        ArrayView<GPUCapsuleCollider>, ArrayView<GPUContactConstraint>, ArrayView<int>,
        int, int, int, int, int, float, float, float> _detectCapsuleContactsKernel = null!;

    // Contact solving kernel
    private Action<Index1D, ArrayView<GPUContactConstraint>, ArrayView<GPURigidBody>,
        ArrayView<int>, int, int, int> _solveContactVelocitiesKernel = null!;

    // Contact warm-starting kernels
    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUContactConstraint>,
        ArrayView<GPUCachedContactImpulse>, ArrayView<int>,
        int, int, int, int> _applyWarmStartKernel = null!;
    private Action<Index1D, ArrayView<GPUContactConstraint>,
        ArrayView<GPUCachedContactImpulse>, ArrayView<int>,
        int, int> _storeToCacheKernel = null!;

    // Joint solving kernels
    private Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURevoluteJoint>,
        ArrayView<GPUJointConstraint>, float> _initializeJointConstraintsKernel = null!;
    // Per-world Gauss-Seidel joint solvers (no atomics, sequential within each world)
    private Action<Index1D, ArrayView<GPUJointConstraint>, ArrayView<GPURigidBody>,
        int, float> _solveJointVelocitiesKernel = null!;
    private Action<Index1D, ArrayView<GPUJointConstraint>, ArrayView<GPURigidBody>,
        int> _solveJointPositionsKernel = null!;

    /// <summary>
    /// Creates a new batched GPU stepper using the provided accelerator.
    /// </summary>
    /// <param name="accelerator">The ILGPU accelerator to use for kernel execution.</param>
    public GPUBatchedStepper(Accelerator accelerator)
    {
        _accelerator = accelerator;
        LoadKernels();
    }

    private void LoadKernels()
    {
        // Load physics kernels
        _applyGravityKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, int, float, float, float>(
            GPUBatchedPhysicsKernels.BatchedApplyGravityKernel);

        _savePrevPositionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>>(
            GPUBatchedPhysicsKernels.BatchedSavePreviousPositionsKernel);

        _integrateKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, float>(
            GPUBatchedPhysicsKernels.BatchedIntegrateKernel);

        _updateGeomPositionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBodyGeom>, ArrayView<GPURigidBody>, int, int>(
            GPUBatchedPhysicsKernels.BatchedUpdateGeomPositionsKernel);

        _dampVelocitiesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, float, float, float>(
            GPUBatchedPhysicsKernels.BatchedDampVelocitiesKernel);

        _velocityStabilizationKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, float, float, float>(
            GPUBatchedPhysicsKernels.BatchedVelocityStabilizationKernel);

        // Load contact management kernels
        _clearContactCountsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>>(
            GPUBatchedContactKernels.BatchedClearContactCountsKernel);

        // Load contact detection kernels
        _detectOBBContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>,
            ArrayView<GPUOBBCollider>, ArrayView<GPUContactConstraint>, ArrayView<int>,
            int, int, int, int, int, float, float, float>(
            GPUBatchedContactKernels.BatchedDetectOBBContactsKernel);

        _detectCircleContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>,
            ArrayView<GPUCircleCollider>, ArrayView<GPUContactConstraint>, ArrayView<int>,
            int, int, int, int, int, float, float, float>(
            GPUBatchedContactKernels.BatchedDetectCircleContactsKernel);

        _detectCapsuleContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>,
            ArrayView<GPUCapsuleCollider>, ArrayView<GPUContactConstraint>, ArrayView<int>,
            int, int, int, int, int, float, float, float>(
            GPUBatchedContactKernels.BatchedDetectCapsuleContactsKernel);

        // Load contact solver kernel
        _solveContactVelocitiesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPUContactConstraint>, ArrayView<GPURigidBody>,
            ArrayView<int>, int, int, int>(
            GPUBatchedContactKernels.BatchedSolveContactVelocitiesKernel);

        // Load warm-starting kernels
        _applyWarmStartKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<GPUContactConstraint>,
            ArrayView<GPUCachedContactImpulse>, ArrayView<int>,
            int, int, int, int>(
            GPUBatchedContactKernels.BatchedApplyWarmStartKernel);

        _storeToCacheKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPUContactConstraint>,
            ArrayView<GPUCachedContactImpulse>, ArrayView<int>,
            int, int>(
            GPUBatchedContactKernels.BatchedStoreToCacheKernel);

        // Load joint solver kernels
        // Init uses non-batched kernel (joints have global body indices from evaluator)
        _initializeJointConstraintsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<GPURevoluteJoint>,
            ArrayView<GPUJointConstraint>, float>(
            GPURigidBodyJointKernels.InitializeJointConstraintsKernel);

        // Per-world Gauss-Seidel solvers (no atomics, sequential within each world)
        _solveJointVelocitiesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPUJointConstraint>, ArrayView<GPURigidBody>,
            int, float>(
            GPUBatchedJointKernels.BatchedSolveJointVelocitiesPerWorldKernel);

        _solveJointPositionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPUJointConstraint>, ArrayView<GPURigidBody>,
            int>(
            GPUBatchedJointKernels.BatchedSolveJointPositionsPerWorldKernel);
    }

    /// <summary>
    /// Step all N worlds by one physics timestep.
    /// </summary>
    /// <param name="worldState">The batched GPU world state containing all bodies, geoms, and contacts.</param>
    /// <param name="simConfig">The simulation configuration (timestep, iterations, physics parameters).</param>
    public void Step(GPUBatchedWorldState worldState, SimulationConfig simConfig)
    {
        StepNoSync(worldState, simConfig);
        _accelerator.Synchronize();
    }

    /// <summary>
    /// Step all N worlds by one physics timestep without synchronizing.
    /// Kernels are enqueued on the default stream which guarantees execution order.
    /// Only sync when CPU needs to read GPU memory (e.g., downloading fitness values).
    /// </summary>
    public void StepNoSync(GPUBatchedWorldState worldState, SimulationConfig simConfig)
    {
        var config = worldState.Config;
        float dt = simConfig.Dt;
        float invDt = 1f / dt;
        int iterations = simConfig.XpbdIterations;

        for (int substep = 0; substep < simConfig.Substeps; substep++)
        {
            SubStep(worldState, config, simConfig, dt, invDt, iterations);
        }
    }

    private void SubStep(
        GPUBatchedWorldState worldState,
        GPUBatchedWorldConfig config,
        SimulationConfig simConfig,
        float dt,
        float invDt,
        int iterations)
    {
        // 1. Apply gravity to all rigid bodies
        if (config.TotalRigidBodies > 0)
        {
            _applyGravityKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                config.RigidBodiesPerWorld,
                simConfig.GravityX,
                simConfig.GravityY,
                dt);
        }

        // 2. Save previous positions for velocity stabilization
        if (config.TotalRigidBodies > 0)
        {
            _savePrevPositionsKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View);
        }

        // 3. Integrate positions from velocities
        if (config.TotalRigidBodies > 0)
        {
            _integrateKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                dt);
        }

        // 4. Update geom world positions from rigid body transforms
        if (config.TotalGeoms > 0)
        {
            _updateGeomPositionsKernel(
                config.TotalGeoms,
                worldState.Geoms.View,
                worldState.RigidBodies.View,
                config.GeomsPerWorld,
                config.RigidBodiesPerWorld);
        }

        // 5. Clear contact counts for all worlds
        _clearContactCountsKernel(
            config.WorldCount,
            worldState.ContactCounts.View);

        // 6. Detect contacts (geoms vs shared OBB colliders)
        // Each thread handles one (geom, collider) pair
        int obbPairsPerWorld = config.GeomsPerWorld * config.SharedColliderCount;
        int totalOBBPairs = config.WorldCount * obbPairsPerWorld;

        if (totalOBBPairs > 0 && config.SharedColliderCount > 0)
        {
            _detectOBBContactsKernel(
                totalOBBPairs,
                worldState.RigidBodies.View,
                worldState.Geoms.View,
                worldState.SharedOBBColliders.View,
                worldState.ContactConstraints.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.RigidBodiesPerWorld,
                config.GeomsPerWorld,
                config.SharedColliderCount,
                config.MaxContactsPerWorld,
                simConfig.FrictionMu,
                simConfig.Restitution,
                dt);
        }

        // 7. Apply warm-starting from previous frame's cached impulses
        if (config.TotalContacts > 0)
        {
            _applyWarmStartKernel(
                config.TotalContacts,
                worldState.RigidBodies.View,
                worldState.ContactConstraints.View,
                worldState.ContactCache.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.RigidBodiesPerWorld,
                config.MaxContactsPerWorld,
                config.MaxContactsPerWorld); // cacheSize = maxContactsPerWorld
        }

        // 8. Initialize joint constraints (precompute effective masses)
        if (config.TotalJoints > 0)
        {
            _initializeJointConstraintsKernel(
                config.TotalJoints,
                worldState.RigidBodies.View,
                worldState.Joints.View,
                worldState.JointConstraints.View,
                dt);
        }

        // 9. Solve constraints iteratively (sequential impulse method)
        for (int iter = 0; iter < iterations; iter++)
        {
            // Solve contact velocity constraints (one thread per world, Gauss-Seidel)
            if (config.TotalContacts > 0)
            {
                _solveContactVelocitiesKernel(
                    config.WorldCount,
                    worldState.ContactConstraints.View,
                    worldState.RigidBodies.View,
                    worldState.ContactCounts.View,
                    config.WorldCount,
                    config.RigidBodiesPerWorld,
                    config.MaxContactsPerWorld);
            }

            // Solve joint velocity constraints (one thread per world, Gauss-Seidel)
            if (config.TotalJoints > 0)
            {
                _solveJointVelocitiesKernel(
                    config.WorldCount,
                    worldState.JointConstraints.View,
                    worldState.RigidBodies.View,
                    config.JointsPerWorld,
                    dt);
            }
        }

        // 10. Solve joint position constraints (one thread per world, Gauss-Seidel)
        if (config.TotalJoints > 0)
        {
            _solveJointPositionsKernel(
                config.WorldCount,
                worldState.JointConstraints.View,
                worldState.RigidBodies.View,
                config.JointsPerWorld);
        }

        // 11. Store contact impulses to cache for next frame's warm-starting
        if (config.TotalContacts > 0)
        {
            _storeToCacheKernel(
                config.TotalContacts,
                worldState.ContactConstraints.View,
                worldState.ContactCache.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.MaxContactsPerWorld);
        }

        // NOTE: No velocity stabilization for rigid bodies!
        // Velocity stabilization (v = (pos-prevPos)/dt) is for XPBD particle systems where
        // constraints modify positions. The impulse-based rigid body solver works directly
        // on velocities, so stabilization would overwrite the solved velocities.
        // CPU CPUStepper.cs line 105: "Rigid body velocity stabilization not needed with impulse solver"

        // 12. Apply velocity damping
        if (config.TotalRigidBodies > 0 && (simConfig.GlobalDamping > 0f || simConfig.AngularDamping > 0f))
        {
            _dampVelocitiesKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                simConfig.GlobalDamping,
                simConfig.AngularDamping,
                dt);
        }
    }

    /// <summary>
    /// Step with additional circle colliders (beyond the shared OBBs).
    /// Useful when scenes have mixed collider types.
    /// </summary>
    public void StepWithCircleColliders(
        GPUBatchedWorldState worldState,
        SimulationConfig simConfig,
        MemoryBuffer1D<GPUCircleCollider, Stride1D.Dense> circleColliders,
        int circleCount)
    {
        var config = worldState.Config;
        float dt = simConfig.Dt;
        float invDt = 1f / dt;
        int iterations = simConfig.XpbdIterations;

        for (int substep = 0; substep < simConfig.Substeps; substep++)
        {
            SubStepWithCircles(worldState, config, simConfig, dt, invDt, iterations, circleColliders, circleCount);
        }

        _accelerator.Synchronize();
    }

    private void SubStepWithCircles(
        GPUBatchedWorldState worldState,
        GPUBatchedWorldConfig config,
        SimulationConfig simConfig,
        float dt,
        float invDt,
        int iterations,
        MemoryBuffer1D<GPUCircleCollider, Stride1D.Dense> circleColliders,
        int circleCount)
    {
        // 1-4. Apply gravity, save positions, integrate, update geoms (same as base Step)
        if (config.TotalRigidBodies > 0)
        {
            _applyGravityKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                config.RigidBodiesPerWorld,
                simConfig.GravityX,
                simConfig.GravityY,
                dt);

            _savePrevPositionsKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View);

            _integrateKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                dt);
        }

        if (config.TotalGeoms > 0)
        {
            _updateGeomPositionsKernel(
                config.TotalGeoms,
                worldState.Geoms.View,
                worldState.RigidBodies.View,
                config.GeomsPerWorld,
                config.RigidBodiesPerWorld);
        }

        // 5. Clear contact counts
        _clearContactCountsKernel(
            config.WorldCount,
            worldState.ContactCounts.View);

        // 6a. Detect OBB contacts
        int obbPairsPerWorld = config.GeomsPerWorld * config.SharedColliderCount;
        int totalOBBPairs = config.WorldCount * obbPairsPerWorld;

        if (totalOBBPairs > 0 && config.SharedColliderCount > 0)
        {
            _detectOBBContactsKernel(
                totalOBBPairs,
                worldState.RigidBodies.View,
                worldState.Geoms.View,
                worldState.SharedOBBColliders.View,
                worldState.ContactConstraints.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.RigidBodiesPerWorld,
                config.GeomsPerWorld,
                config.SharedColliderCount,
                config.MaxContactsPerWorld,
                simConfig.FrictionMu,
                simConfig.Restitution,
                dt);
        }

        // 6b. Detect circle collider contacts
        int circlePairsPerWorld = config.GeomsPerWorld * circleCount;
        int totalCirclePairs = config.WorldCount * circlePairsPerWorld;

        if (totalCirclePairs > 0 && circleCount > 0)
        {
            _detectCircleContactsKernel(
                totalCirclePairs,
                worldState.RigidBodies.View,
                worldState.Geoms.View,
                circleColliders.View,
                worldState.ContactConstraints.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.RigidBodiesPerWorld,
                config.GeomsPerWorld,
                circleCount,
                config.MaxContactsPerWorld,
                simConfig.FrictionMu,
                simConfig.Restitution,
                dt);
        }

        // 7. Apply warm-starting from previous frame's cached impulses
        if (config.TotalContacts > 0)
        {
            _applyWarmStartKernel(
                config.TotalContacts,
                worldState.RigidBodies.View,
                worldState.ContactConstraints.View,
                worldState.ContactCache.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.RigidBodiesPerWorld,
                config.MaxContactsPerWorld,
                config.MaxContactsPerWorld);
        }

        // 8. Initialize joint constraints
        if (config.TotalJoints > 0)
        {
            _initializeJointConstraintsKernel(
                config.TotalJoints,
                worldState.RigidBodies.View,
                worldState.Joints.View,
                worldState.JointConstraints.View,
                dt);
        }

        // 9. Solve constraints iteratively
        for (int iter = 0; iter < iterations; iter++)
        {
            // Solve contact velocity constraints (one thread per world, Gauss-Seidel)
            if (config.TotalContacts > 0)
            {
                _solveContactVelocitiesKernel(
                    config.WorldCount,
                    worldState.ContactConstraints.View,
                    worldState.RigidBodies.View,
                    worldState.ContactCounts.View,
                    config.WorldCount,
                    config.RigidBodiesPerWorld,
                    config.MaxContactsPerWorld);
            }

            // Solve joint velocity constraints (one thread per world, Gauss-Seidel)
            if (config.TotalJoints > 0)
            {
                _solveJointVelocitiesKernel(
                    config.WorldCount,
                    worldState.JointConstraints.View,
                    worldState.RigidBodies.View,
                    config.JointsPerWorld,
                    dt);
            }
        }

        // 10. Solve joint position constraints (one thread per world, Gauss-Seidel)
        if (config.TotalJoints > 0)
        {
            _solveJointPositionsKernel(
                config.WorldCount,
                worldState.JointConstraints.View,
                worldState.RigidBodies.View,
                config.JointsPerWorld);
        }

        // 11. Store contact impulses to cache for next frame's warm-starting
        if (config.TotalContacts > 0)
        {
            _storeToCacheKernel(
                config.TotalContacts,
                worldState.ContactConstraints.View,
                worldState.ContactCache.View,
                worldState.ContactCounts.View,
                config.WorldCount,
                config.MaxContactsPerWorld);
        }

        // NOTE: No velocity stabilization for rigid bodies (see SubStep comment)

        if (config.TotalRigidBodies > 0 && (simConfig.GlobalDamping > 0f || simConfig.AngularDamping > 0f))
        {
            _dampVelocitiesKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                simConfig.GlobalDamping,
                simConfig.AngularDamping,
                dt);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Kernels are managed by the accelerator, no explicit disposal needed
    }
}
