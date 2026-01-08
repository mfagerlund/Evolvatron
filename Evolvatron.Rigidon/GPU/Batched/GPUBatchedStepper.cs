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
    }

    /// <summary>
    /// Step all N worlds by one physics timestep.
    /// </summary>
    /// <param name="worldState">The batched GPU world state containing all bodies, geoms, and contacts.</param>
    /// <param name="simConfig">The simulation configuration (timestep, iterations, physics parameters).</param>
    public void Step(GPUBatchedWorldState worldState, SimulationConfig simConfig)
    {
        var config = worldState.Config;
        float dt = simConfig.Dt;
        float invDt = 1f / dt;
        int iterations = simConfig.XpbdIterations;

        // Run substeps if configured
        for (int substep = 0; substep < simConfig.Substeps; substep++)
        {
            SubStep(worldState, config, simConfig, dt, invDt, iterations);
        }

        // Synchronize to ensure all GPU work is complete
        _accelerator.Synchronize();
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

        // 7. Solve constraints iteratively (sequential impulse method)
        for (int iter = 0; iter < iterations; iter++)
        {
            // Solve contact velocity constraints
            if (config.TotalContacts > 0)
            {
                _solveContactVelocitiesKernel(
                    config.TotalContacts,
                    worldState.ContactConstraints.View,
                    worldState.RigidBodies.View,
                    worldState.ContactCounts.View,
                    config.WorldCount,
                    config.RigidBodiesPerWorld,
                    config.MaxContactsPerWorld);
            }

            // TODO: Add joint solving kernels when GPUBatchedJointKernels is implemented
            // _solveJointVelocitiesKernel(...)
        }

        // TODO: Add joint position solving when GPUBatchedJointKernels is implemented
        // _solveJointPositionsKernel(...)

        // 8. Velocity stabilization: derive velocities from position changes
        if (config.TotalRigidBodies > 0 && simConfig.VelocityStabilizationBeta > 0f)
        {
            _velocityStabilizationKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                invDt,
                simConfig.VelocityStabilizationBeta,
                simConfig.MaxVelocity);
        }

        // 9. Apply velocity damping
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

        // 7-9. Solve, stabilize, damp (same as base Step)
        for (int iter = 0; iter < iterations; iter++)
        {
            if (config.TotalContacts > 0)
            {
                _solveContactVelocitiesKernel(
                    config.TotalContacts,
                    worldState.ContactConstraints.View,
                    worldState.RigidBodies.View,
                    worldState.ContactCounts.View,
                    config.WorldCount,
                    config.RigidBodiesPerWorld,
                    config.MaxContactsPerWorld);
            }
        }

        if (config.TotalRigidBodies > 0 && simConfig.VelocityStabilizationBeta > 0f)
        {
            _velocityStabilizationKernel(
                config.TotalRigidBodies,
                worldState.RigidBodies.View,
                invDt,
                simConfig.VelocityStabilizationBeta,
                simConfig.MaxVelocity);
        }

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
