using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU.MegaKernel;
using Evolvatron.Evolvion.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated fitness evaluator using the fused mega-kernel.
/// One kernel launch per step instead of ~35 separate dispatches.
/// Host loop: 900 steps × 1 kernel = 900 dispatches (vs 31,500 in original).
/// </summary>
public class GPURocketLandingMegaEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly int _maxIndividuals;
    private bool _disposed;

    // Neural network state
    private GPUEvolvionState? _neuralState;
    private SpeciesSpec? _currentSpec;

    // Physics state
    private GPUBatchedWorldState? _worldState;
    private int _lastWorldCount;

    // Landing-specific GPU buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _observations;
    private MemoryBuffer1D<float, Stride1D.Dense>? _actions;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentThrottle;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentGimbal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hasLanded;
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;

    // The single fused kernel
    private readonly Action<Index1D, PhysicsViews, NNViews, EpisodeViews,
        MegaKernelConfig, ArrayView<float>, ArrayView<float>> _fusedStepKernel;

    // Reset kernel (run once per episode)
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _resetLandingStateKernel;

    // Host caches
    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private byte[]? _landedCache;

    // Landing parameters (matching RocketEnvironment defaults)
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public float GroundY { get; set; } = -5f;
    public float SpawnHeight { get; set; } = 15f;
    public float PadX { get; set; } = 0f;
    public float PadY => GroundY + 0.5f;
    public float PadHalfWidth { get; set; } = 2f;
    public float MaxLandingVel { get; set; } = 2f;
    public float MaxLandingAngle { get; set; } = 15f * MathF.PI / 180f;
    public float LandingBonus { get; set; } = 200f;

    // Difficulty knobs
    public float SpawnXRange { get; set; } = 2f;
    public float SpawnXMin { get; set; } = 0f;
    public float SpawnAngleRange { get; set; } = 0f;
    public float InitialVelXRange { get; set; } = 1f;
    public float InitialVelYMax { get; set; } = 2f;

    public int SolverIterations { get; set; } = 8;

    public Accelerator Accelerator => _accelerator;

    public GPURocketLandingMegaEvaluator(int maxIndividuals = 1500)
    {
        _maxIndividuals = maxIndividuals;

        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPURocketLandingMegaEvaluator on: {_accelerator.Name}");

        // Load the single fused step kernel
        _fusedStepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, PhysicsViews, NNViews, EpisodeViews,
            MegaKernelConfig, ArrayView<float>, ArrayView<float>>(
            RocketLandingStepKernel.StepKernel);

        // Load helper kernel (run once per episode, not per step)
        _resetLandingStateKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>>(
            GPUBatchedRocketLandingKernels.ResetLandingStateKernel);
    }

    /// <summary>
    /// Evaluates all individuals using the fused mega-kernel.
    /// </summary>
    public (float[] fitness, int landings) EvaluatePopulation(
        SpeciesSpec spec,
        List<Individual> individuals,
        int seed,
        int maxSteps = 600)
    {
        int worldCount = individuals.Count;
        if (worldCount == 0)
            return (Array.Empty<float>(), 0);

        if (worldCount > _maxIndividuals)
            throw new ArgumentException($"Population size ({worldCount}) exceeds maximum ({_maxIndividuals})");

        InitializeNeuralState(spec, individuals);
        EnsureBuffers(worldCount);
        CreateAndUploadRocketTemplate(worldCount, seed);

        _worldState!.UploadSharedColliders(new GPUOBBCollider[]
        {
            new GPUOBBCollider { CX = 0f, CY = GroundY, UX = 1f, UY = 0f, HalfExtentX = 30f, HalfExtentY = 0.5f }
        });

        // Reset landing state
        _resetLandingStateKernel(
            worldCount,
            _stepCounters!.View,
            _isTerminal!.View,
            _hasLanded!.View,
            _currentThrottle!.View,
            _currentGimbal!.View,
            _fitnessValues!.View);
        _worldState.ClearContactCounts();
        _worldState.ClearContactCache();
        _accelerator.Synchronize();

        // Build config struct
        var megaConfig = new MegaKernelConfig
        {
            BodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = 1,
            MaxContactsPerWorld = 48,
            Dt = 1f / 120f,
            GravityX = 0f,
            GravityY = -9.81f,
            FrictionMu = 0.8f,
            Restitution = 0.0f,
            GlobalDamping = 0.02f,
            AngularDamping = 0.1f,
            SolverIterations = SolverIterations,
            TotalNodes = spec.TotalNodes,
            TotalEdges = spec.TotalEdges,
            NumRows = spec.RowPlans.Length,
            InputSize = 8,
            OutputSize = 2,
            MaxSteps = maxSteps,
            MaxThrust = MaxThrust,
            MaxGimbalTorque = MaxGimbalTorque,
            PadX = PadX,
            PadY = PadY,
            PadHalfWidth = PadHalfWidth,
            MaxLandingVel = MaxLandingVel,
            MaxLandingAngle = MaxLandingAngle,
            GroundY = GroundY,
            SpawnHeight = SpawnHeight,
            LandingBonus = LandingBonus
        };

        // Build view structs
        var physicsViews = new PhysicsViews
        {
            Bodies = _worldState.RigidBodies.View,
            Geoms = _worldState.Geoms.View,
            Joints = _worldState.Joints.View,
            JointConstraints = _worldState.JointConstraints.View,
            Contacts = _worldState.ContactConstraints.View,
            ContactCache = _worldState.ContactCache.View,
            ContactCounts = _worldState.ContactCounts.View,
            SharedOBBColliders = _worldState.SharedOBBColliders.View
        };

        var nnViews = new NNViews
        {
            NodeValues = _neuralState!.NodeValues.View,
            Edges = _neuralState.Edges.View,
            Weights = _neuralState.Individuals.Weights.View,
            Biases = _neuralState.Individuals.Biases.View,
            Activations = _neuralState.Individuals.Activations.View,
            NodeParams = _neuralState.Individuals.NodeParams.View,
            RowPlans = _neuralState.RowPlans.View
        };

        var episodeViews = new EpisodeViews
        {
            CurrentThrottle = _currentThrottle!.View,
            CurrentGimbal = _currentGimbal!.View,
            IsTerminal = _isTerminal!.View,
            HasLanded = _hasLanded!.View,
            StepCounters = _stepCounters!.View,
            FitnessValues = _fitnessValues!.View
        };

        // Main simulation loop — 1 kernel per step
        for (int step = 0; step < maxSteps; step++)
        {
            // Early exit check every 60 steps
            if (step > 0 && step % 60 == 0)
            {
                _accelerator.Synchronize();
                _isTerminal!.CopyToCPU(_terminalCache!);
                bool allDone = true;
                for (int i = 0; i < worldCount; i++)
                {
                    if (_terminalCache![i] == 0) { allDone = false; break; }
                }
                if (allDone) break;
            }

            _fusedStepKernel(
                worldCount,
                physicsViews,
                nnViews,
                episodeViews,
                megaConfig,
                _observations!.View,
                _actions!.View);
        }

        // Fitness is computed inline at termination — no post-loop kernel needed
        _accelerator.Synchronize();
        _fitnessValues!.CopyToCPU(_fitnessCache!);
        _hasLanded!.CopyToCPU(_landedCache!);

        int landings = 0;
        for (int i = 0; i < worldCount; i++)
            if (_landedCache![i] != 0) landings++;

        var result = new float[worldCount];
        Array.Copy(_fitnessCache!, result, worldCount);
        return (result, landings);
    }

    private void EnsureBuffers(int worldCount)
    {
        if (_lastWorldCount == worldCount && _worldState != null)
            return;

        _worldState?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _currentThrottle?.Dispose();
        _currentGimbal?.Dispose();
        _isTerminal?.Dispose();
        _hasLanded?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();

        var worldConfig = new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = 1,
            TargetsPerWorld = 0,
            MaxContactsPerWorld = 48
        };

        _worldState = new GPUBatchedWorldState(_accelerator, worldConfig);

        _observations = _accelerator.Allocate1D<float>(worldCount * 8);
        _actions = _accelerator.Allocate1D<float>(worldCount * 2);
        _currentThrottle = _accelerator.Allocate1D<float>(worldCount);
        _currentGimbal = _accelerator.Allocate1D<float>(worldCount);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _hasLanded = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _landedCache = new byte[worldCount];
        _lastWorldCount = worldCount;
    }

    private void CreateAndUploadRocketTemplate(int worldCount, int seed)
    {
        const float bodyHeight = 1.5f;
        const float bodyRadius = 0.2f;
        const float bodyMass = 8f;
        const float bodyHalfLength = bodyHeight * 0.5f;
        float bodyInertia = bodyMass * (bodyRadius * bodyRadius * 0.25f + bodyHeight * bodyHeight / 12f);

        const float legLength = 1.0f;
        const float legRadius = 0.1f;
        const float legMass = 1.5f;
        const float legHalfLength = legLength * 0.5f;
        float legInertia = legMass * (legRadius * legRadius * 0.25f + legLength * legLength / 12f);

        const float bodyAngle = MathF.PI / 2f;
        float leftLegAngle = 225f * MathF.PI / 180f;
        float rightLegAngle = 315f * MathF.PI / 180f;

        float leftOffX = MathF.Cos(leftLegAngle) * legHalfLength;
        float leftOffY = MathF.Sin(leftLegAngle) * legHalfLength;
        float rightOffX = MathF.Cos(rightLegAngle) * legHalfLength;
        float rightOffY = MathF.Sin(rightLegAngle) * legHalfLength;

        float leftRefAngle = leftLegAngle - bodyAngle;
        float rightRefAngle = rightLegAngle - bodyAngle;

        int bodyGeomCount = Math.Clamp((int)(bodyHalfLength / bodyRadius) + 2, 3, 7);
        int legGeomCount = Math.Clamp((int)(legHalfLength / legRadius) + 2, 3, 7);

        var bodyGeoms = new GPURigidBodyGeom[bodyGeomCount];
        for (int i = 0; i < bodyGeomCount; i++)
        {
            float t = (float)i / (bodyGeomCount - 1);
            bodyGeoms[i] = new GPURigidBodyGeom
            {
                LocalX = -bodyHalfLength + t * bodyHeight,
                LocalY = 0f,
                Radius = bodyRadius,
                BodyIndex = 0
            };
        }

        var legGeoms = new GPURigidBodyGeom[legGeomCount];
        for (int i = 0; i < legGeomCount; i++)
        {
            float t = (float)i / (legGeomCount - 1);
            legGeoms[i] = new GPURigidBodyGeom
            {
                LocalX = -legHalfLength + t * legLength,
                LocalY = 0f,
                Radius = legRadius,
                BodyIndex = 0
            };
        }

        var rng = new Random(seed);
        var config = _worldState!.Config;

        var allBodies = new GPURigidBody[config.TotalRigidBodies];
        var allGeoms = new GPURigidBodyGeom[config.TotalGeoms];
        var allJoints = new GPURevoluteJoint[config.TotalJoints];

        for (int w = 0; w < worldCount; w++)
        {
            float spawnX;
            if (SpawnXMin > 0f)
            {
                float side = rng.NextDouble() < 0.5 ? -1f : 1f;
                spawnX = side * (SpawnXMin + (float)(rng.NextDouble() * (SpawnXRange - SpawnXMin)));
            }
            else
            {
                spawnX = (float)(rng.NextDouble() * SpawnXRange * 2 - SpawnXRange);
            }
            float spawnY = SpawnHeight + (float)(rng.NextDouble() * 3.0);
            float spawnTilt = (float)(rng.NextDouble() * SpawnAngleRange * 2 - SpawnAngleRange);
            float velX = (float)(rng.NextDouble() * InitialVelXRange * 2 - InitialVelXRange);
            float velY = (float)(rng.NextDouble() * -InitialVelYMax);

            float centerY = spawnY;
            float cosT = MathF.Cos(spawnTilt);
            float sinT = MathF.Sin(spawnTilt);

            float bodyX = spawnX + 0f * cosT - bodyHalfLength * sinT;
            float bodyY = centerY + 0f * sinT + bodyHalfLength * cosT;
            float bodyAngleFinal = bodyAngle + spawnTilt;

            allBodies[config.GetRigidBodyIndex(w, 0)] = new GPURigidBody
            {
                X = bodyX, Y = bodyY,
                Angle = bodyAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = bodyX, PrevY = bodyY, PrevAngle = bodyAngleFinal,
                InvMass = 1f / bodyMass,
                InvInertia = 1f / bodyInertia,
                GeomStartIndex = 0,
                GeomCount = bodyGeomCount
            };

            float leftLegX = spawnX + leftOffX * cosT - leftOffY * sinT;
            float leftLegY = centerY + leftOffX * sinT + leftOffY * cosT;
            float leftAngleFinal = leftLegAngle + spawnTilt;
            allBodies[config.GetRigidBodyIndex(w, 1)] = new GPURigidBody
            {
                X = leftLegX, Y = leftLegY,
                Angle = leftAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = leftLegX, PrevY = leftLegY, PrevAngle = leftAngleFinal,
                InvMass = 1f / legMass,
                InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount,
                GeomCount = legGeomCount
            };

            float rightLegX = spawnX + rightOffX * cosT - rightOffY * sinT;
            float rightLegY = centerY + rightOffX * sinT + rightOffY * cosT;
            float rightAngleFinal = rightLegAngle + spawnTilt;
            allBodies[config.GetRigidBodyIndex(w, 2)] = new GPURigidBody
            {
                X = rightLegX, Y = rightLegY,
                Angle = rightAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = rightLegX, PrevY = rightLegY, PrevAngle = rightAngleFinal,
                InvMass = 1f / legMass,
                InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount + legGeomCount,
                GeomCount = legGeomCount
            };

            for (int i = 0; i < bodyGeomCount; i++)
            {
                var g = bodyGeoms[i];
                g.BodyIndex = 0;
                allGeoms[config.GetGeomIndex(w, i)] = g;
            }

            for (int i = 0; i < legGeomCount; i++)
            {
                var g = legGeoms[i];
                g.BodyIndex = 1;
                allGeoms[config.GetGeomIndex(w, bodyGeomCount + i)] = g;
            }

            for (int i = 0; i < legGeomCount; i++)
            {
                var g = legGeoms[i];
                g.BodyIndex = 2;
                allGeoms[config.GetGeomIndex(w, bodyGeomCount + legGeomCount + i)] = g;
            }

            int globalBody = config.GetRigidBodyIndex(w, 0);
            int globalLeftLeg = config.GetRigidBodyIndex(w, 1);
            int globalRightLeg = config.GetRigidBodyIndex(w, 2);

            allJoints[config.GetJointIndex(w, 0)] = new GPURevoluteJoint
            {
                BodyA = globalBody,
                BodyB = globalLeftLeg,
                LocalAnchorAX = -bodyHalfLength, LocalAnchorAY = 0f,
                LocalAnchorBX = -legHalfLength, LocalAnchorBY = 0f,
                ReferenceAngle = leftRefAngle,
                EnableLimits = 0,
                LowerAngle = 0f, UpperAngle = 0f,
                EnableMotor = 1,
                MotorSpeed = 0f,
                MaxMotorTorque = 1000f
            };

            allJoints[config.GetJointIndex(w, 1)] = new GPURevoluteJoint
            {
                BodyA = globalBody,
                BodyB = globalRightLeg,
                LocalAnchorAX = -bodyHalfLength, LocalAnchorAY = 0f,
                LocalAnchorBX = -legHalfLength, LocalAnchorBY = 0f,
                ReferenceAngle = rightRefAngle,
                EnableLimits = 0,
                LowerAngle = 0f, UpperAngle = 0f,
                EnableMotor = 1,
                MotorSpeed = 0f,
                MaxMotorTorque = 1000f
            };
        }

        _worldState.RigidBodies.CopyFromCPU(allBodies);
        _worldState.Geoms.CopyFromCPU(allGeoms);
        _worldState.Joints.CopyFromCPU(allJoints);
    }

    private void InitializeNeuralState(SpeciesSpec spec, List<Individual> individuals)
    {
        _currentSpec = spec;

        if (_neuralState == null ||
            individuals.Count > _neuralState.MaxIndividuals ||
            spec.TotalNodes > _neuralState.MaxNodes ||
            spec.TotalEdges > _neuralState.MaxEdges)
        {
            _neuralState?.Dispose();
            _neuralState = new GPUEvolvionState(
                _accelerator,
                maxIndividuals: Math.Max(_maxIndividuals, individuals.Count),
                maxNodes: Math.Max(100, spec.TotalNodes),
                maxEdges: Math.Max(500, spec.TotalEdges));
        }

        _neuralState.UploadTopology(spec);
        _neuralState.UploadIndividuals(individuals, spec);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _worldState?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _currentThrottle?.Dispose();
        _currentGimbal?.Dispose();
        _isTerminal?.Dispose();
        _hasLanded?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _neuralState?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
