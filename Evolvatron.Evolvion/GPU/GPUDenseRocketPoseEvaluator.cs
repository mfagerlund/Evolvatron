using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU.MegaKernel;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// Goal-relative POSE-REACHING controller evaluator (used by the rocket editor's "Train Controller").
///
/// Trains ONE reactive controller to fly to and briefly HOLD a target pose (position + orientation)
/// in free space — no pad, no obstacles. Same rocket template, actuators, and physics as the other
/// dense evaluators; only the observation (10-D goal-relative), reward (pose-error shaping), and
/// success criterion (held the pose within tolerance) differ. See <see cref="DenseRocketPoseStepKernel"/>.
///
/// Each "spawn" = one shared initial state + one shared random target pose (seeded by baseSeed+spawn),
/// applied to the whole population — the multi-position training pattern. Physical note: a single
/// bottom thruster can only hover near-upright, so target angles are sampled within a modest range
/// about upright (<see cref="TargetAngleRange"/>); large tilts at zero velocity are not physical.
/// </summary>
public class GPUDenseRocketPoseEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly DenseTopology _topology;
    private bool _disposed;

    private MemoryBuffer1D<float, Stride1D.Dense>? _weights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _biases;
    private MemoryBuffer1D<int, Stride1D.Dense> _layerSizes;

    private GPUBatchedWorldState? _worldState;

    private MemoryBuffer1D<float, Stride1D.Dense>? _observations;
    private MemoryBuffer1D<float, Stride1D.Dense>? _actions;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentThrottle;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentGimbal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hasHit;       // reused as the "held the pose" flag
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;
    private MemoryBuffer1D<float, Stride1D.Dense>? _shapeAccum;  // EpisodeViews.WaggleAccum
    private MemoryBuffer1D<int, Stride1D.Dense>? _holdCounter;   // EpisodeViews.SettledSteps

    private readonly Action<Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
        MegaKernelConfig, DenseRocketNNConfig,
        ArrayView<float>, ArrayView<float>> _fusedStepKernel;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _resetKernel;

    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private byte[]? _hitCache;
    private float[]? _weightsCPU;
    private float[]? _biasesCPU;
    private int _lastWorldCount;

    // === Physics params (parity with the other dense evaluators) ===
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public int SolverIterations { get; set; } = 6;
    public int MaxSteps { get; set; } = 400;

    // === Initial state randomization (per spawn, shared across the population) ===
    public float SpawnAngleRange { get; set; } = 0.3f;   // rad, ± about upright
    public float InitialSpeedMax { get; set; } = 2f;     // m/s
    public float InitialAngVelMax { get; set; } = 0.4f;  // rad/s

    // === Target pose randomization (per spawn) ===
    public float TargetXRange { get; set; } = 6f;        // ± about origin
    public float TargetYRange { get; set; } = 5f;        // ± about origin
    public float TargetAngleCenter { get; set; } = MathF.PI / 2f;  // upright
    public float TargetAngleRange { get; set; } = 60f * MathF.PI / 180f;

    // === Pose reward / hit tolerance ===
    public float PoseHitRadius { get; set; } = 1.2f;
    public float PoseHitAngle { get; set; } = 15f * MathF.PI / 180f;
    public float PoseHitSpeed { get; set; } = 1.5f;
    public int PoseHoldSteps { get; set; } = 30;         // ~0.25s at 120Hz
    public float PosePosWeight { get; set; } = 1.0f;
    public float PoseAngleWeight { get; set; } = 0.7f;
    public float PoseVelWeight { get; set; } = 0.7f;
    public float PoseHitBonus { get; set; } = 200f;
    public float HasteBonus { get; set; } = 0.5f;
    public float WagglePenalty { get; set; } = 0f;

    public int SpawnCount { get; set; } = 10;
    public int SpawnSeed { get; set; } = 0;

    public DenseTopology Topology => _topology;
    public Accelerator Accelerator => _accelerator;

    // Current per-spawn target (set before each spawn's run; read by BuildMegaConfig).
    private float _targetX, _targetY, _targetAngle = MathF.PI / 2f;

    public int OptimalPopulationSize
    {
        get
        {
            int sms = _accelerator.NumMultiprocessors;
            int warpSize = _accelerator.WarpSize;
            return sms * 4 * warpSize;
        }
    }

    public GPUDenseRocketPoseEvaluator(DenseTopology topology)
    {
        const int gpuMaxLayerWidth = 64;
        if (topology.MaxLayerWidth > gpuMaxLayerWidth)
            throw new ArgumentException(
                $"Pose NN topology has max layer width {topology.MaxLayerWidth} but GPU kernel limit is {gpuMaxLayerWidth}. " +
                $"Reduce hidden/output layer sizes. Topology: {topology}");
        if (topology.InputSize != 10)
            throw new ArgumentException(
                $"Pose NN input size ({topology.InputSize}) != 10. Use DenseTopology.ForRocketPose(hidden). Topology: {topology}");
        if (topology.OutputSize != 2)
            throw new ArgumentException(
                $"Pose NN output size ({topology.OutputSize}) != 2 (throttle+gimbal). Topology: {topology}");

        _topology = topology;
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPUDenseRocketPoseEvaluator on: {_accelerator.Name} " +
                          $"({_accelerator.NumMultiprocessors} SMs, optimal pop={OptimalPopulationSize}), topology={topology}");

        _fusedStepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
            MegaKernelConfig, DenseRocketNNConfig,
            ArrayView<float>, ArrayView<float>>(
            DenseRocketPoseStepKernel.StepKernel);

        _resetKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>>(
            GPUBatchedRocketLandingKernels.ResetLandingStateKernel);

        _layerSizes = _accelerator.Allocate1D<int>(topology.NumLayers);
        _layerSizes.CopyFromCPU(topology.LayerSizes);
    }

    /// <summary>
    /// Evaluate a population across `numSpawns` (initial-state, target-pose) conditions.
    /// Returns aggregated fitness (hitCount * MaxSteps + meanFitness) per individual.
    /// </summary>
    public (float[] fitness, int totalHits, int maxHitCount, int maxHitIdx) EvaluateMultiSpawn(
        float[] paramVectors, int totalPop, int numSpawns, int baseSeed)
    {
        if (totalPop == 0)
            return (Array.Empty<float>(), 0, 0, 0);

        EnsureBuffers(totalPop);
        SplitAndUpload(paramVectors, totalPop);

        var nnConfig = BuildNNConfig();
        var nnViews = BuildNNViews();

        var totalFitness = new float[totalPop];
        var hitCounts = new int[totalPop];

        for (int spawn = 0; spawn < numSpawns; spawn++)
        {
            int seed = baseSeed + spawn;
            SampleTargetPose(seed, out _targetX, out _targetY, out _targetAngle);
            CreateAndUploadRocketTemplate(totalPop, seed);
            ResetEpisodeState(totalPop);

            var megaConfig = BuildMegaConfig();
            var physicsViews = BuildPhysicsViews();
            var episodeViews = BuildEpisodeViews();

            RunSimulationLoop(totalPop, megaConfig, nnConfig, physicsViews, nnViews, episodeViews);

            _accelerator.Synchronize();
            _fitnessValues!.CopyToCPU(_fitnessCache!);
            _hasHit!.CopyToCPU(_hitCache!);

            for (int i = 0; i < totalPop; i++)
            {
                totalFitness[i] += _fitnessCache![i];
                if (_hitCache![i] != 0) hitCounts[i]++;
            }
        }

        var result = new float[totalPop];
        int totalHits = 0;
        for (int i = 0; i < totalPop; i++)
        {
            float meanFitness = totalFitness[i] / numSpawns;
            result[i] = hitCounts[i] * MaxSteps + meanFitness;
            totalHits += hitCounts[i];
        }

        int maxHitCount = 0, maxHitIdx = 0;
        for (int i = 0; i < totalPop; i++)
        {
            if (hitCounts[i] > maxHitCount ||
                (hitCounts[i] == maxHitCount && result[i] > result[maxHitIdx]))
            {
                maxHitCount = hitCounts[i];
                maxHitIdx = i;
            }
        }

        return (result, totalHits, maxHitCount, maxHitIdx);
    }

    public (int hits, int total) EvaluateChampion(float[] mu, int numSpawns, int baseSeed)
    {
        var (_, hits, _, _) = EvaluateMultiSpawn(mu, 1, numSpawns, baseSeed);
        return (hits, numSpawns);
    }

    /// <summary>
    /// Deterministic per-spawn target-pose sampler. Exposed static so editor-side CPU visualization
    /// can mirror the exact same target distribution.
    /// </summary>
    public void SampleTargetPose(int seed, out float tx, out float ty, out float tAngle)
        => SampleTargetPose(seed, TargetXRange, TargetYRange, TargetAngleCenter, TargetAngleRange,
                            out tx, out ty, out tAngle);

    public static void SampleTargetPose(int seed, float xRange, float yRange,
        float angleCenter, float angleRange, out float tx, out float ty, out float tAngle)
    {
        var r = new Random(seed * 9176 + 1234);
        tx = (float)(r.NextDouble() * 2 - 1) * xRange;
        ty = (float)(r.NextDouble() * 2 - 1) * yRange;
        tAngle = angleCenter + (float)(r.NextDouble() * 2 - 1) * angleRange;
    }

    private DenseRocketNNConfig BuildNNConfig() => new()
    {
        NumLayers = _topology.NumLayers,
        TotalWeightsPerNet = _topology.TotalWeights,
        TotalBiasesPerNet = _topology.TotalBiases
    };

    private MegaKernelConfig BuildMegaConfig() => new()
    {
        BodiesPerWorld = 3,
        GeomsPerWorld = 19,
        JointsPerWorld = 2,
        SharedColliderCount = 0,    // free space
        MaxContactsPerWorld = 24,
        Dt = 1f / 120f,
        GravityX = 0f,
        GravityY = -9.81f,
        FrictionMu = 0.8f,
        Restitution = 0.0f,
        GlobalDamping = 0.02f,
        AngularDamping = 0.1f,
        SolverIterations = SolverIterations,
        InputSize = _topology.InputSize,    // 10
        OutputSize = _topology.OutputSize,  // 2
        MaxSteps = MaxSteps,
        MaxThrust = MaxThrust,
        MaxGimbalTorque = MaxGimbalTorque,
        GroundY = -1000f,                   // free space — bounds handled by dist-from-target
        SpawnHeight = 1000f,
        HasteBonus = HasteBonus,
        WagglePenalty = WagglePenalty,
        // pose target + tolerances
        PadX = _targetX,
        PadY = _targetY,
        TargetAngle = _targetAngle,
        PoseHitRadius = PoseHitRadius,
        PoseHitAngle = PoseHitAngle,
        PoseHitSpeed = PoseHitSpeed,
        PoseHoldSteps = PoseHoldSteps,
        PosePosWeight = PosePosWeight,
        PoseAngleWeight = PoseAngleWeight,
        PoseVelWeight = PoseVelWeight,
        PoseHitBonus = PoseHitBonus
    };

    private DenseNNViews BuildNNViews() => new()
    {
        Weights = _weights!.View,
        Biases = _biases!.View,
        LayerSizes = _layerSizes.View
    };

    private PhysicsViews BuildPhysicsViews() => new()
    {
        Bodies = _worldState!.RigidBodies.View,
        Geoms = _worldState.Geoms.View,
        Joints = _worldState.Joints.View,
        JointConstraints = _worldState.JointConstraints.View,
        Contacts = _worldState.ContactConstraints.View,
        ContactCache = _worldState.ContactCache.View,
        ContactCounts = _worldState.ContactCounts.View,
        SharedOBBColliders = _worldState.SharedOBBColliders.View
    };

    private EpisodeViews BuildEpisodeViews() => new()
    {
        CurrentThrottle = _currentThrottle!.View,
        CurrentGimbal = _currentGimbal!.View,
        IsTerminal = _isTerminal!.View,
        HasLanded = _hasHit!.View,
        StepCounters = _stepCounters!.View,
        FitnessValues = _fitnessValues!.View,
        WaggleAccum = _shapeAccum!.View,
        SettledSteps = _holdCounter!.View
    };

    private void RunSimulationLoop(int worldCount,
        MegaKernelConfig megaConfig, DenseRocketNNConfig nnConfig,
        PhysicsViews physicsViews, DenseNNViews nnViews, EpisodeViews episodeViews)
    {
        const int batchSize = 10;

        for (int step = 0; step < MaxSteps; step++)
        {
            _fusedStepKernel(
                worldCount, physicsViews, nnViews, episodeViews,
                megaConfig, nnConfig,
                _observations!.View, _actions!.View);

            if ((step + 1) % batchSize == 0)
            {
                try { _accelerator.Synchronize(); }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"GPU FATAL: Synchronize failed at step {step}/{MaxSteps} with {worldCount} worlds. " +
                        $"CUDA context poisoned (likely OOB access). NN={_topology}", ex);
                }

                if (step % 30 == 29)
                {
                    _isTerminal!.CopyToCPU(_terminalCache!);
                    bool allDone = true;
                    for (int i = 0; i < worldCount; i++)
                        if (_terminalCache![i] == 0) { allDone = false; break; }
                    if (allDone) break;
                }
            }
        }
    }

    private void SplitAndUpload(float[] paramVectors, int totalPop)
    {
        int tw = _topology.TotalWeights;
        int tb = _topology.TotalBiases;
        int tp = _topology.TotalParams;

        if (_weightsCPU == null || _weightsCPU.Length != totalPop * tw)
        {
            _weightsCPU = new float[totalPop * tw];
            _biasesCPU = new float[totalPop * tb];
        }

        for (int i = 0; i < totalPop; i++)
        {
            Array.Copy(paramVectors, i * tp, _weightsCPU, i * tw, tw);
            Array.Copy(paramVectors, i * tp + tw, _biasesCPU!, i * tb, tb);
        }

        _weights!.CopyFromCPU(_weightsCPU);
        _biases!.CopyFromCPU(_biasesCPU!);
    }

    private void ResetEpisodeState(int worldCount)
    {
        _resetKernel(
            worldCount,
            _stepCounters!.View, _isTerminal!.View, _hasHit!.View,
            _currentThrottle!.View, _currentGimbal!.View, _fitnessValues!.View);
        _shapeAccum!.MemSetToZero();
        _holdCounter!.MemSetToZero();
        _worldState!.ClearContactCounts();
        _worldState.ClearContactCache();
        _accelerator.Synchronize();
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
        _hasHit?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _shapeAccum?.Dispose();
        _holdCounter?.Dispose();
        _weights?.Dispose();
        _biases?.Dispose();

        var worldConfig = new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = 1,   // ILGPU needs a non-empty view; kernel uses count=0
            TargetsPerWorld = 0,
            MaxContactsPerWorld = 24
        };

        _worldState = new GPUBatchedWorldState(_accelerator, worldConfig);

        int inputSize = _topology.InputSize;
        _observations = _accelerator.Allocate1D<float>(worldCount * inputSize);
        _actions = _accelerator.Allocate1D<float>(worldCount * _topology.OutputSize);
        _currentThrottle = _accelerator.Allocate1D<float>(worldCount);
        _currentGimbal = _accelerator.Allocate1D<float>(worldCount);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _hasHit = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        _shapeAccum = _accelerator.Allocate1D<float>(worldCount);
        _holdCounter = _accelerator.Allocate1D<int>(worldCount);
        _weights = _accelerator.Allocate1D<float>(worldCount * _topology.TotalWeights);
        _biases = _accelerator.Allocate1D<float>(worldCount * _topology.TotalBiases);

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _hitCache = new byte[worldCount];
        _lastWorldCount = worldCount;
    }

    /// <summary>
    /// Build the 3-body rocket (fuselage + 2 legs) at the origin in free space with a randomized
    /// initial state (tilt, velocity, angular velocity). One shared state for all worlds (fair CEM).
    /// </summary>
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
                LocalX = -bodyHalfLength + t * bodyHeight, LocalY = 0f,
                Radius = bodyRadius, BodyIndex = 0
            };
        }

        var legGeoms = new GPURigidBodyGeom[legGeomCount];
        for (int i = 0; i < legGeomCount; i++)
        {
            float t = (float)i / (legGeomCount - 1);
            legGeoms[i] = new GPURigidBodyGeom
            {
                LocalX = -legHalfLength + t * legLength, LocalY = 0f,
                Radius = legRadius, BodyIndex = 0
            };
        }

        var config = _worldState!.Config;
        var allBodies = new GPURigidBody[config.TotalRigidBodies];
        var allGeoms = new GPURigidBodyGeom[config.TotalGeoms];
        var allJoints = new GPURevoluteJoint[config.TotalJoints];

        // Dummy collider (SharedColliderCount = 0 means it is never read).
        _worldState.UploadSharedColliders(new[]
        {
            new GPUOBBCollider { CX = 0f, CY = -1e6f, UX = 1f, UY = 0f, HalfExtentX = 1f, HalfExtentY = 1f }
        });

        var r = new Random(seed);
        float tilt = (float)(r.NextDouble() * SpawnAngleRange * 2 - SpawnAngleRange);
        double vang = r.NextDouble() * 2.0 * Math.PI;
        double vspd = r.NextDouble() * InitialSpeedMax;
        float velX = (float)(Math.Cos(vang) * vspd);
        float velY = (float)(Math.Sin(vang) * vspd);
        float angVel0 = (float)(r.NextDouble() * InitialAngVelMax * 2 - InitialAngVelMax);

        const float spawnX = 0f, spawnY = 0f;
        float cosT = MathF.Cos(tilt);
        float sinT = MathF.Sin(tilt);

        for (int w = 0; w < worldCount; w++)
        {
            float bodyX = spawnX;
            float bodyY = spawnY + bodyHalfLength * cosT;
            float bodyAngleFinal = bodyAngle + tilt;

            allBodies[config.GetRigidBodyIndex(w, 0)] = new GPURigidBody
            {
                X = bodyX, Y = bodyY, Angle = bodyAngleFinal,
                VelX = velX, VelY = velY, AngularVel = angVel0,
                PrevX = bodyX, PrevY = bodyY, PrevAngle = bodyAngleFinal,
                InvMass = 1f / bodyMass, InvInertia = 1f / bodyInertia,
                GeomStartIndex = 0, GeomCount = bodyGeomCount
            };

            float leftLegX = spawnX + leftOffX * cosT - leftOffY * sinT;
            float leftLegY = spawnY + leftOffX * sinT + leftOffY * cosT;
            float leftAngleFinal = leftLegAngle + tilt;
            allBodies[config.GetRigidBodyIndex(w, 1)] = new GPURigidBody
            {
                X = leftLegX, Y = leftLegY, Angle = leftAngleFinal,
                VelX = velX, VelY = velY, AngularVel = angVel0,
                PrevX = leftLegX, PrevY = leftLegY, PrevAngle = leftAngleFinal,
                InvMass = 1f / legMass, InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount, GeomCount = legGeomCount
            };

            float rightLegX = spawnX + rightOffX * cosT - rightOffY * sinT;
            float rightLegY = spawnY + rightOffX * sinT + rightOffY * cosT;
            float rightAngleFinal = rightLegAngle + tilt;
            allBodies[config.GetRigidBodyIndex(w, 2)] = new GPURigidBody
            {
                X = rightLegX, Y = rightLegY, Angle = rightAngleFinal,
                VelX = velX, VelY = velY, AngularVel = angVel0,
                PrevX = rightLegX, PrevY = rightLegY, PrevAngle = rightAngleFinal,
                InvMass = 1f / legMass, InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount + legGeomCount, GeomCount = legGeomCount
            };

            for (int i = 0; i < bodyGeomCount; i++)
            {
                var g = bodyGeoms[i]; g.BodyIndex = 0;
                allGeoms[config.GetGeomIndex(w, i)] = g;
            }
            for (int i = 0; i < legGeomCount; i++)
            {
                var g = legGeoms[i]; g.BodyIndex = 1;
                allGeoms[config.GetGeomIndex(w, bodyGeomCount + i)] = g;
            }
            for (int i = 0; i < legGeomCount; i++)
            {
                var g = legGeoms[i]; g.BodyIndex = 2;
                allGeoms[config.GetGeomIndex(w, bodyGeomCount + legGeomCount + i)] = g;
            }

            int globalBody = config.GetRigidBodyIndex(w, 0);
            int globalLeftLeg = config.GetRigidBodyIndex(w, 1);
            int globalRightLeg = config.GetRigidBodyIndex(w, 2);

            allJoints[config.GetJointIndex(w, 0)] = new GPURevoluteJoint
            {
                BodyA = globalBody, BodyB = globalLeftLeg,
                LocalAnchorAX = -bodyHalfLength, LocalAnchorAY = 0f,
                LocalAnchorBX = -legHalfLength, LocalAnchorBY = 0f,
                ReferenceAngle = leftRefAngle,
                EnableLimits = 0, LowerAngle = 0f, UpperAngle = 0f,
                EnableMotor = 1, MotorSpeed = 0f, MaxMotorTorque = 1000f
            };
            allJoints[config.GetJointIndex(w, 1)] = new GPURevoluteJoint
            {
                BodyA = globalBody, BodyB = globalRightLeg,
                LocalAnchorAX = -bodyHalfLength, LocalAnchorAY = 0f,
                LocalAnchorBX = -legHalfLength, LocalAnchorBY = 0f,
                ReferenceAngle = rightRefAngle,
                EnableLimits = 0, LowerAngle = 0f, UpperAngle = 0f,
                EnableMotor = 1, MotorSpeed = 0f, MaxMotorTorque = 1000f
            };
        }

        _worldState.RigidBodies.CopyFromCPU(allBodies);
        _worldState.Geoms.CopyFromCPU(allGeoms);
        _worldState.Joints.CopyFromCPU(allJoints);
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
        _hasHit?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _shapeAccum?.Dispose();
        _holdCounter?.Dispose();
        _weights?.Dispose();
        _biases?.Dispose();
        _layerSizes.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }
}
