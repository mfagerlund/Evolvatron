using ILGPU;
using ILGPU.Runtime;
using System;

namespace Evolvatron.Core.GPU;

/// <summary>
/// GPU-accelerated stepper using ILGPU.
/// Drop-in replacement for CPUStepper with identical API.
/// </summary>
public sealed class GPUStepper : IStepper, IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private GPUWorldState? _gpuWorld;

    // Compiled kernels
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float> _applyGravityKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float> _integrateKernel;
    private readonly Action<Index1D, ArrayView<GPURod>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float> _solveRodsKernel;
    private readonly Action<Index1D, ArrayView<GPUAngle>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float> _solveAnglesKernel;
    private readonly Action<Index1D, ArrayView<GPUMotorAngle>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float> _solveMotorsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<GPUCircleCollider>, ArrayView<GPUCapsuleCollider>, ArrayView<GPUOBBCollider>, int, int, int, float, float> _solveContactsKernel;

    // Post-processing kernels
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float> _velocityStabilizationKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<GPUCircleCollider>, ArrayView<GPUCapsuleCollider>, ArrayView<GPUOBBCollider>, int, int, int, float> _frictionKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float> _dampingKernel;

    // Rigid body kernels
    private readonly Action<Index1D, ArrayView<GPURigidBody>, float, float, float> _applyRigidBodyGravityKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, float> _integrateRigidBodiesKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, float> _dampRigidBodiesKernel;

    // Rigid body joint kernels
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURevoluteJoint>, ArrayView<GPUJointConstraint>, float> _initializeJointConstraintsKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUJointConstraint>, float> _solveJointVelocitiesKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUJointConstraint>> _solveJointPositionsKernel;

    // Rigid body contact kernels
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>, ArrayView<GPUCircleCollider>, ArrayView<GPUContactConstraint>, int, int, int, int, float, float, float> _detectCircleContactsKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>, ArrayView<GPUCapsuleCollider>, ArrayView<GPUContactConstraint>, int, int, int, int, float, float, float> _detectCapsuleContactsKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>, ArrayView<GPUOBBCollider>, ArrayView<GPUContactConstraint>, int, int, int, int, float, float, float> _detectOBBContactsKernel;
    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUContactConstraint>> _solveContactVelocitiesKernel;

    private bool _initialized = false;

    public GPUStepper()
    {
        // Initialize ILGPU context
        _context = Context.CreateDefault();

        // Try to get CUDA accelerator, fall back to CPU
        // Note: OpenCL backend on some Intel GPUs has issues, use CPU for now
        _accelerator = _context.GetPreferredDevice(preferCPU: true)
            .CreateAccelerator(_context);

        Console.WriteLine($"GPU Stepper initialized on: {_accelerator.Name}");
        Console.WriteLine($"  Device type: {_accelerator.AcceleratorType}");
        Console.WriteLine($"  Memory: {_accelerator.MemorySize / (1024 * 1024)} MB");

        // Load and compile kernels
        _applyGravityKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>(
            GPUKernels.ApplyGravityKernel);

        _integrateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float>(
            GPUKernels.IntegrateKernel);

        _solveRodsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURod>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>(
            GPUKernels.SolveRodsKernel);

        _solveAnglesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPUAngle>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>(
            GPUKernels.SolveAnglesKernel);

        _solveMotorsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPUMotorAngle>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>(
            GPUKernels.SolveMotorsKernel);

        _solveContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<GPUCircleCollider>, ArrayView<GPUCapsuleCollider>, ArrayView<GPUOBBCollider>, int, int, int, float, float>(
            GPUKernels.SolveContactsKernel);

        // Load post-processing kernels
        _velocityStabilizationKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>(
            GPUKernels.VelocityStabilizationKernel);

        _frictionKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<GPUCircleCollider>, ArrayView<GPUCapsuleCollider>, ArrayView<GPUOBBCollider>, int, int, int, float>(
            GPUKernels.FrictionKernel);

        _dampingKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float>(
            GPUKernels.DampingKernel);

        // Load rigid body kernels
        _applyRigidBodyGravityKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, float, float, float>(
            GPURigidBodyContactKernels.ApplyRigidBodyGravityKernel);

        _integrateRigidBodiesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, float>(
            GPURigidBodyContactKernels.IntegrateRigidBodiesKernel);

        _dampRigidBodiesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, float>(
            GPURigidBodyContactKernels.DampRigidBodiesKernel);

        // Load rigid body joint kernels
        _initializeJointConstraintsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURevoluteJoint>, ArrayView<GPUJointConstraint>, float>(
            GPURigidBodyJointKernels.InitializeJointConstraintsKernel);

        _solveJointVelocitiesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUJointConstraint>, float>(
            GPURigidBodyJointKernels.SolveJointVelocitiesKernel);

        _solveJointPositionsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUJointConstraint>>(
            GPURigidBodyJointKernels.SolveJointPositionsKernel);

        // Load rigid body contact kernels
        _detectCircleContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>, ArrayView<GPUCircleCollider>, ArrayView<GPUContactConstraint>, int, int, int, int, float, float, float>(
            GPURigidBodyContactKernels.DetectCircleContactsKernel);

        _detectCapsuleContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>, ArrayView<GPUCapsuleCollider>, ArrayView<GPUContactConstraint>, int, int, int, int, float, float, float>(
            GPURigidBodyContactKernels.DetectCapsuleContactsKernel);

        _detectOBBContactsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPURigidBodyGeom>, ArrayView<GPUOBBCollider>, ArrayView<GPUContactConstraint>, int, int, int, int, float, float, float>(
            GPURigidBodyContactKernels.DetectOBBContactsKernel);

        _solveContactVelocitiesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GPURigidBody>, ArrayView<GPUContactConstraint>>(
            GPURigidBodyContactKernels.SolveContactVelocitiesKernel);
    }

    public void Step(WorldState world, SimulationConfig cfg)
    {
        // Initialize GPU world on first use
        if (!_initialized)
        {
            _gpuWorld = new GPUWorldState(_accelerator, world.Capacity);
            _initialized = true;
        }

        // Upload world state to GPU
        _gpuWorld!.UploadFrom(world);

        // Run substeps
        for (int substep = 0; substep < cfg.Substeps; substep++)
        {
            SubStep(_gpuWorld, cfg);
        }

        // Download results back to CPU
        _gpuWorld.DownloadTo(world);

        // Synchronize accelerator
        _accelerator.Synchronize();
    }

    private void SubStep(GPUWorldState gpu, SimulationConfig cfg)
    {
        float dt = cfg.Dt;
        int particleCount = gpu.ParticleCount;

        // 1. Apply gravity
        _applyGravityKernel(particleCount, gpu.ForceX.View, gpu.ForceY.View, gpu.InvMass.View, cfg.GravityX, cfg.GravityY);

        // 2. Save previous positions for velocity stabilization
        gpu.PosX.CopyTo(gpu.PrevPosX);
        gpu.PosY.CopyTo(gpu.PrevPosY);

        // 3. Integrate
        _integrateKernel(particleCount, gpu.PosX.View, gpu.PosY.View, gpu.VelX.View, gpu.VelY.View,
            gpu.ForceX.View, gpu.ForceY.View, gpu.InvMass.View, dt);

        // 4. Reset lambdas (done on GPU by just running iterations)
        // In XPBD we reset lambdas each step - for GPU we just overwrite them

        // 5. XPBD iterations
        for (int iter = 0; iter < cfg.XpbdIterations; iter++)
        {
            // Solve rods
            if (gpu.RodCount > 0)
            {
                _solveRodsKernel(gpu.RodCount, gpu.Rods.View, gpu.PosX.View, gpu.PosY.View,
                    gpu.InvMass.View, dt, cfg.RodCompliance);
            }

            // Solve angles
            if (gpu.AngleCount > 0)
            {
                _solveAnglesKernel(gpu.AngleCount, gpu.Angles.View, gpu.PosX.View, gpu.PosY.View,
                    gpu.InvMass.View, dt, cfg.AngleCompliance);
            }

            // Solve motors
            if (gpu.MotorCount > 0)
            {
                _solveMotorsKernel(gpu.MotorCount, gpu.Motors.View, gpu.PosX.View, gpu.PosY.View,
                    gpu.InvMass.View, dt, cfg.MotorCompliance);
            }

            // Solve contacts
            _solveContactsKernel(particleCount, gpu.PosX.View, gpu.PosY.View, gpu.InvMass.View, gpu.Radius.View,
                gpu.Circles.View, gpu.Capsules.View, gpu.OBBs.View,
                gpu.CircleCount, gpu.CapsuleCount, gpu.OBBCount,
                dt, cfg.ContactCompliance);
        }

        // 6. Velocity stabilization
        if (cfg.VelocityStabilizationBeta > 0f)
        {
            float invDt = 1f / dt;
            _velocityStabilizationKernel(particleCount,
                gpu.PosX.View, gpu.PosY.View,
                gpu.PrevPosX.View, gpu.PrevPosY.View,
                gpu.VelX.View, gpu.VelY.View,
                gpu.InvMass.View,
                invDt, cfg.VelocityStabilizationBeta);
        }

        // 7. Friction
        if (cfg.FrictionMu > 0f)
        {
            _frictionKernel(particleCount,
                gpu.PosX.View, gpu.PosY.View,
                gpu.VelX.View, gpu.VelY.View,
                gpu.InvMass.View, gpu.Radius.View,
                gpu.Circles.View, gpu.Capsules.View, gpu.OBBs.View,
                gpu.CircleCount, gpu.CapsuleCount, gpu.OBBCount,
                cfg.FrictionMu);
        }

        // 8. Damping
        if (cfg.GlobalDamping > 0f)
        {
            float dampingFactor = MathF.Max(0f, 1f - cfg.GlobalDamping * dt);
            _dampingKernel(particleCount,
                gpu.VelX.View, gpu.VelY.View,
                gpu.InvMass.View,
                dampingFactor);
        }

        // ========== RIGID BODY PHYSICS ==========
        int rigidBodyCount = gpu.RigidBodyCount;
        if (rigidBodyCount > 0)
        {
            // 9. Apply gravity to rigid bodies
            _applyRigidBodyGravityKernel(rigidBodyCount, gpu.RigidBodies.View, cfg.GravityX, cfg.GravityY, dt);

            // 10. Integrate rigid body velocities to positions
            _integrateRigidBodiesKernel(rigidBodyCount, gpu.RigidBodies.View, dt);

            // 11. Detect contacts (rigid body geoms vs all static collider types)
            int contactOffset = 0;

            // Circle colliders
            int circleContactCount = rigidBodyCount * gpu.MaxGeomsPerBody * gpu.CircleCount;
            if (circleContactCount > 0 && gpu.CircleCount > 0)
            {
                _detectCircleContactsKernel(circleContactCount,
                    gpu.RigidBodies.View, gpu.RigidBodyGeoms.View, gpu.Circles.View, gpu.ContactConstraints.View,
                    contactOffset,
                    rigidBodyCount, gpu.MaxGeomsPerBody, gpu.CircleCount,
                    cfg.FrictionMu, cfg.Restitution, dt);
            }
            contactOffset += circleContactCount;

            // Capsule colliders
            int capsuleContactCount = rigidBodyCount * gpu.MaxGeomsPerBody * gpu.CapsuleCount;
            if (capsuleContactCount > 0 && gpu.CapsuleCount > 0)
            {
                _detectCapsuleContactsKernel(capsuleContactCount,
                    gpu.RigidBodies.View, gpu.RigidBodyGeoms.View, gpu.Capsules.View, gpu.ContactConstraints.View,
                    contactOffset,
                    rigidBodyCount, gpu.MaxGeomsPerBody, gpu.CapsuleCount,
                    cfg.FrictionMu, cfg.Restitution, dt);
            }
            contactOffset += capsuleContactCount;

            // OBB colliders
            int obbContactCount = rigidBodyCount * gpu.MaxGeomsPerBody * gpu.OBBCount;
            if (obbContactCount > 0 && gpu.OBBCount > 0)
            {
                _detectOBBContactsKernel(obbContactCount,
                    gpu.RigidBodies.View, gpu.RigidBodyGeoms.View, gpu.OBBs.View, gpu.ContactConstraints.View,
                    contactOffset,
                    rigidBodyCount, gpu.MaxGeomsPerBody, gpu.OBBCount,
                    cfg.FrictionMu, cfg.Restitution, dt);
            }

            int totalContacts = circleContactCount + capsuleContactCount + obbContactCount;

            // 12. Initialize joint constraints
            if (gpu.RevoluteJointCount > 0)
            {
                _initializeJointConstraintsKernel(gpu.RevoluteJointCount,
                    gpu.RigidBodies.View, gpu.RevoluteJoints.View, gpu.JointConstraints.View, dt);
            }

            // 13. Sequential impulse iterations
            for (int iter = 0; iter < cfg.XpbdIterations; iter++)
            {
                // Solve contact velocities
                if (totalContacts > 0)
                {
                    _solveContactVelocitiesKernel(totalContacts, gpu.RigidBodies.View, gpu.ContactConstraints.View);
                }

                // Solve joint velocities
                if (gpu.RevoluteJointCount > 0)
                {
                    _solveJointVelocitiesKernel(gpu.RevoluteJointCount, gpu.RigidBodies.View, gpu.JointConstraints.View, dt);
                }
            }

            // 14. Solve joint position constraints (for stability)
            if (gpu.RevoluteJointCount > 0)
            {
                _solveJointPositionsKernel(gpu.RevoluteJointCount, gpu.RigidBodies.View, gpu.JointConstraints.View);
            }

            // 15. Damping for rigid bodies
            if (cfg.GlobalDamping > 0f)
            {
                float dampingFactor = MathF.Max(0f, 1f - cfg.GlobalDamping * dt);
                _dampRigidBodiesKernel(rigidBodyCount, gpu.RigidBodies.View, dampingFactor);
            }
        }
    }

    private void StabilizeVelocities(GPUWorldState gpu, float dt, float beta)
    {
        if (beta <= 0f) return;

        // Simple kernel for velocity stabilization
        // For now, download and do on CPU (could be optimized with custom kernel)
        var posX = gpu.PosX.GetAsArray1D();
        var posY = gpu.PosY.GetAsArray1D();
        var prevPosX = gpu.PrevPosX.GetAsArray1D();
        var prevPosY = gpu.PrevPosY.GetAsArray1D();
        var velX = gpu.VelX.GetAsArray1D();
        var velY = gpu.VelY.GetAsArray1D();

        float invDt = 1f / dt;
        float oneMinusBeta = 1f - beta;

        for (int i = 0; i < gpu.ParticleCount; i++)
        {
            float correctedVx = (posX[i] - prevPosX[i]) * invDt;
            float correctedVy = (posY[i] - prevPosY[i]) * invDt;
            velX[i] = correctedVx * beta + velX[i] * oneMinusBeta;
            velY[i] = correctedVy * beta + velY[i] * oneMinusBeta;
        }

        gpu.VelX.CopyFromCPU(velX);
        gpu.VelY.CopyFromCPU(velY);
    }

    private void ApplyDamping(GPUWorldState gpu, float damping, float dt)
    {
        if (damping <= 0f) return;

        float factor = MathF.Max(0f, 1f - damping * dt);
        var velX = gpu.VelX.GetAsArray1D();
        var velY = gpu.VelY.GetAsArray1D();

        for (int i = 0; i < gpu.ParticleCount; i++)
        {
            velX[i] *= factor;
            velY[i] *= factor;
        }

        gpu.VelX.CopyFromCPU(velX);
        gpu.VelY.CopyFromCPU(velY);
    }

    public void Dispose()
    {
        _gpuWorld?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
