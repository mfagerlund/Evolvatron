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

        // 6. Velocity stabilization (skipped for GPU - requires custom kernel)
        // TODO: Implement as GPU kernel for better performance

        // 7. Friction (skipped for GPU - requires custom kernel)
        // TODO: Implement as GPU kernel

        // 8. Damping (skipped for GPU - requires custom kernel)
        // TODO: Implement as GPU kernel
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
