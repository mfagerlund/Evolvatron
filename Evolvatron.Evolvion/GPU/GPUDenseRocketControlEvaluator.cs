using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU.MegaKernel;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// Phase-1 maneuvering-controller evaluator (see docs/phase1_controller_spec.md).
///
/// Trains ONE controller to track world-frame velocity commands in free space —
/// no pad, no obstacles, no maze. Same rocket template, actuators, and physics as
/// GPUDenseRocketLandingEvaluator; only the observation (9-D dynamics+command),
/// reward (velocity tracking), and terminals (tumble / MaxSteps) differ.
///
/// Each "condition" = one shared initial state + one shared piecewise-constant command
/// schedule (seeded by baseSeed+conditionIdx), applied to the whole population for a fair
/// CEM comparison — the controller analog of multi-position training.
/// </summary>
public class GPUDenseRocketControlEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly DenseTopology _topology;
    private bool _disposed;

    // Dense NN buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _weights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _biases;
    private MemoryBuffer1D<int, Stride1D.Dense> _layerSizes;

    // Physics state
    private GPUBatchedWorldState? _worldState;

    // Episode buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _observations;
    private MemoryBuffer1D<float, Stride1D.Dense>? _actions;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentThrottle;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentGimbal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hasLanded; // unused; kept for reset-kernel signature
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;
    private MemoryBuffer1D<float, Stride1D.Dense>? _waggleAccum;
    private MemoryBuffer1D<int, Stride1D.Dense>? _settledSteps; // unused; kept for EpisodeViews

    // Control buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _cmdVx;
    private MemoryBuffer1D<float, Stride1D.Dense>? _cmdVy;
    private MemoryBuffer1D<float, Stride1D.Dense>? _trackRewardAccum;
    private MemoryBuffer1D<float, Stride1D.Dense>? _elmanContext;  // Elman feedback state [worldCount * max(1,ContextSize)]

    // CPU copy of the last-uploaded schedule (for replay command-arrow drawing)
    private float[]? _cmdVxCPU;
    private float[]? _cmdVyCPU;

    // Kernels
    private readonly Action<Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
        MegaKernelConfig, DenseRocketControlConfig, ControlViews,
        ArrayView<float>, ArrayView<float>> _fusedStepKernel;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _resetKernel;

    // Host caches
    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private int[]? _stepCache;
    private float[]? _weightsCPU;
    private float[]? _biasesCPU;
    private int _lastWorldCount;

    // === Physics params (parity with GPUDenseRocketLandingEvaluator) ===
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public int SolverIterations { get; set; } = 6;
    public int MaxSteps { get; set; } = 300;

    // === Command schedule ===
    public int SegmentLength { get; set; } = 72;     // ~0.6s at 120Hz
    public float CmdSpeedMax { get; set; } = 6f;     // m/s, sampled within a disk
    public float HoverFraction { get; set; } = 0.3f; // fraction of segments that command (0,0)
    public int SegmentsPerEpisode => (MaxSteps + SegmentLength - 1) / SegmentLength;

    // === Initial state randomization (per condition) ===
    public float SpawnAngleRange { get; set; } = 0.4f;   // rad, ± about upright
    public float InitialSpeedMax { get; set; } = 3f;     // m/s
    public float InitialAngVelMax { get; set; } = 0.5f;  // rad/s

    // === Tracking reward ===
    public float VErrScale { get; set; } = 8f;
    public float RewardTrackWeight { get; set; } = 1f;
    public float RewardEffortWeight { get; set; } = 0.05f;
    public float AngVelPenalty { get; set; } = 0f;
    public float TumblePenalty { get; set; } = 50f;
    public float TrackSolveThreshold { get; set; } = 0.7f;

    /// <summary>Elman feedback width, derived from the topology (InputSize − 9 base). 0 = plain reactive.</summary>
    public int ContextSize => _topology.InputSize - DenseTopology.ControllerInputSize;

    public DenseTopology Topology => _topology;
    public Accelerator Accelerator => _accelerator;

    public int OptimalPopulationSize
    {
        get
        {
            int sms = _accelerator.NumMultiprocessors;
            int warpSize = _accelerator.WarpSize;
            return sms * 4 * warpSize;
        }
    }

    public GPUDenseRocketControlEvaluator(DenseTopology topology)
    {
        const int gpuMaxLayerWidth = 64;
        if (topology.MaxLayerWidth > gpuMaxLayerWidth)
            throw new ArgumentException(
                $"Control NN topology has max layer width {topology.MaxLayerWidth} but GPU kernel limit is {gpuMaxLayerWidth}. " +
                $"Launching this kernel would corrupt GPU memory. Reduce hidden/output layer sizes. Topology: {topology}");
        int ctx = topology.InputSize - DenseTopology.ControllerInputSize;
        if (ctx < 0)
            throw new ArgumentException(
                $"Control NN input size ({topology.InputSize}) < base {DenseTopology.ControllerInputSize}. " +
                $"Use DenseTopology.ForRocketController(hidden, contextSize). Topology: {topology}");
        if (topology.OutputSize != 2 + ctx)
            throw new ArgumentException(
                $"Control NN output size ({topology.OutputSize}) != 2 (throttle+gimbal) + ContextSize ({ctx}) = {2 + ctx}. " +
                $"Input implies ContextSize={ctx}; output must match. Topology: {topology}");

        _topology = topology;
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPUDenseRocketControlEvaluator on: {_accelerator.Name} " +
                          $"({_accelerator.NumMultiprocessors} SMs, optimal pop={OptimalPopulationSize}), topology={topology}");

        _fusedStepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
            MegaKernelConfig, DenseRocketControlConfig, ControlViews,
            ArrayView<float>, ArrayView<float>>(
            DenseRocketControlStepKernel.StepKernel);

        _resetKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>>(
            GPUBatchedRocketLandingKernels.ResetLandingStateKernel);

        _layerSizes = _accelerator.Allocate1D<int>(topology.NumLayers);
        _layerSizes.CopyFromCPU(topology.LayerSizes);
    }

    /// <summary>
    /// Evaluate a population across multiple (initial-state, command-schedule) conditions.
    /// Returns aggregated fitness (solvedCount * MaxSteps + meanFitness) per individual.
    /// </summary>
    public (float[] fitness, int totalSolved, int maxSolvedCount, int maxSolvedIdx) EvaluateMultiCondition(
        float[] paramVectors, int totalPop, int numConditions, int baseSeed)
    {
        if (totalPop == 0)
            return (Array.Empty<float>(), 0, 0, 0);

        EnsureBuffers(totalPop);
        SplitAndUpload(paramVectors, totalPop);

        var ctrlConfig = BuildControlConfig();
        var nnViews = BuildNNViews();

        var totalFitness = new float[totalPop];
        var solvedCounts = new int[totalPop];

        for (int c = 0; c < numConditions; c++)
        {
            int seed = baseSeed + c;
            CreateAndUploadRocketTemplate(totalPop, seed);
            UploadCommandSchedule(totalPop, seed);
            ResetEpisodeState(totalPop);

            var megaConfig = BuildMegaConfig();
            var physicsViews = BuildPhysicsViews();
            var episodeViews = BuildEpisodeViews();
            var controlViews = BuildControlViews();

            RunSimulationLoop(totalPop, megaConfig, ctrlConfig, physicsViews, nnViews, episodeViews, controlViews);

            _accelerator.Synchronize();
            _fitnessValues!.CopyToCPU(_fitnessCache!);
            _stepCounters!.CopyToCPU(_stepCache!);

            float solveScore = MaxSteps * RewardTrackWeight;
            for (int i = 0; i < totalPop; i++)
            {
                totalFitness[i] += _fitnessCache![i];
                float q = solveScore > 0f ? _fitnessCache[i] / solveScore : 0f;
                if (_stepCache![i] >= MaxSteps && q >= TrackSolveThreshold)
                    solvedCounts[i]++;
            }
        }

        var result = new float[totalPop];
        int totalSolved = 0;
        for (int i = 0; i < totalPop; i++)
        {
            float meanFitness = totalFitness[i] / numConditions;
            result[i] = solvedCounts[i] * MaxSteps + meanFitness;
            totalSolved += solvedCounts[i];
        }

        int maxSolvedCount = 0, maxSolvedIdx = 0;
        for (int i = 0; i < totalPop; i++)
        {
            if (solvedCounts[i] > maxSolvedCount ||
                (solvedCounts[i] == maxSolvedCount && result[i] > result[maxSolvedIdx]))
            {
                maxSolvedCount = solvedCounts[i];
                maxSolvedIdx = i;
            }
        }

        return (result, totalSolved, maxSolvedCount, maxSolvedIdx);
    }

    /// <summary>
    /// Evaluate a single champion across multiple conditions. Returns (solved, total).
    /// </summary>
    public (int solved, int total) EvaluateChampion(float[] mu, int numConditions, int baseSeed)
    {
        var (_, solved, _, _) = EvaluateMultiCondition(mu, 1, numConditions, baseSeed);
        return (solved, numConditions);
    }

    private DenseRocketControlConfig BuildControlConfig()
    {
        return new DenseRocketControlConfig
        {
            NumLayers = _topology.NumLayers,
            TotalWeightsPerNet = _topology.TotalWeights,
            TotalBiasesPerNet = _topology.TotalBiases,
            SegmentLength = SegmentLength,
            SegmentsPerEpisode = SegmentsPerEpisode,
            VErrScale = VErrScale,
            RewardTrackWeight = RewardTrackWeight,
            RewardEffortWeight = RewardEffortWeight,
            AngVelPenalty = AngVelPenalty,
            TumblePenalty = TumblePenalty,
            ContextSize = ContextSize
        };
    }

    private MegaKernelConfig BuildMegaConfig()
    {
        return new MegaKernelConfig
        {
            BodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = 0,    // free space — no ground/pad/obstacles
            MaxContactsPerWorld = 24,
            Dt = 1f / 120f,
            GravityX = 0f,
            GravityY = -9.81f,
            FrictionMu = 0.8f,
            Restitution = 0.0f,
            GlobalDamping = 0.02f,
            AngularDamping = 0.1f,
            SolverIterations = SolverIterations,
            InputSize = _topology.InputSize,    // 9 base + ContextSize
            OutputSize = _topology.OutputSize,  // 2 base + ContextSize
            MaxSteps = MaxSteps,
            MaxThrust = MaxThrust,
            MaxGimbalTorque = MaxGimbalTorque
        };
    }

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
        HasLanded = _hasLanded!.View,
        StepCounters = _stepCounters!.View,
        FitnessValues = _fitnessValues!.View,
        WaggleAccum = _waggleAccum!.View,
        SettledSteps = _settledSteps!.View
    };

    private ControlViews BuildControlViews() => new()
    {
        CmdVx = _cmdVx!.View,
        CmdVy = _cmdVy!.View,
        TrackRewardAccum = _trackRewardAccum!.View,
        ContextBuffer = _elmanContext!.View
    };

    private void RunSimulationLoop(int worldCount,
        MegaKernelConfig megaConfig, DenseRocketControlConfig ctrlConfig,
        PhysicsViews physicsViews, DenseNNViews nnViews, EpisodeViews episodeViews,
        ControlViews controlViews)
    {
        const int batchSize = 10;

        for (int step = 0; step < MaxSteps; step++)
        {
            _fusedStepKernel(
                worldCount, physicsViews, nnViews, episodeViews,
                megaConfig, ctrlConfig, controlViews,
                _observations!.View, _actions!.View);

            if ((step + 1) % batchSize == 0)
            {
                try
                {
                    _accelerator.Synchronize();
                }
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
            _stepCounters!.View, _isTerminal!.View, _hasLanded!.View,
            _currentThrottle!.View, _currentGimbal!.View, _fitnessValues!.View);
        _waggleAccum!.MemSetToZero();
        _settledSteps!.MemSetToZero();
        _trackRewardAccum!.MemSetToZero();
        _elmanContext!.MemSetToZero();   // fresh Elman state each episode
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
        _hasLanded?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _waggleAccum?.Dispose();
        _settledSteps?.Dispose();
        _weights?.Dispose();
        _biases?.Dispose();
        _cmdVx?.Dispose();
        _cmdVy?.Dispose();
        _trackRewardAccum?.Dispose();
        _elmanContext?.Dispose();

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

        int inputSize = _topology.InputSize;   // 9 base + ContextSize
        _observations = _accelerator.Allocate1D<float>(worldCount * inputSize);
        _actions = _accelerator.Allocate1D<float>(worldCount * _topology.OutputSize);
        _currentThrottle = _accelerator.Allocate1D<float>(worldCount);
        _currentGimbal = _accelerator.Allocate1D<float>(worldCount);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _hasLanded = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        _waggleAccum = _accelerator.Allocate1D<float>(worldCount);
        _settledSteps = _accelerator.Allocate1D<int>(worldCount);
        _weights = _accelerator.Allocate1D<float>(worldCount * _topology.TotalWeights);
        _biases = _accelerator.Allocate1D<float>(worldCount * _topology.TotalBiases);

        _cmdVx = _accelerator.Allocate1D<float>(worldCount * SegmentsPerEpisode);
        _cmdVy = _accelerator.Allocate1D<float>(worldCount * SegmentsPerEpisode);
        _trackRewardAccum = _accelerator.Allocate1D<float>(worldCount);
        _elmanContext = _accelerator.Allocate1D<float>(worldCount * Math.Max(1, ContextSize));  // >=1 for non-empty ILGPU view

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _stepCache = new int[worldCount];
        _lastWorldCount = worldCount;
    }

    /// <summary>
    /// Generate a piecewise-constant velocity-command schedule (seeded), replicated to all worlds.
    /// ~HoverFraction of segments command (0,0); the rest are uniform-direction within a speed disk.
    /// </summary>
    private void UploadCommandSchedule(int worldCount, int seed, int[]? perWorldSeeds = null)
    {
        int segs = SegmentsPerEpisode;
        bool perWorld = perWorldSeeds != null;
        var allVx = new float[worldCount * segs];
        var allVy = new float[worldCount * segs];

        int distinct = perWorld ? worldCount : 1;
        for (int d = 0; d < distinct; d++)
        {
            var rng = new Random((perWorld ? perWorldSeeds![d] : seed) * 2 + 1);
            for (int s = 0; s < segs; s++)
            {
                float vx, vy;
                if (rng.NextDouble() < HoverFraction) { vx = 0f; vy = 0f; }
                else
                {
                    double ang = rng.NextDouble() * 2.0 * Math.PI;
                    double spd = CmdSpeedMax * Math.Sqrt(rng.NextDouble());
                    vx = (float)(Math.Cos(ang) * spd);
                    vy = (float)(Math.Sin(ang) * spd);
                }

                if (perWorld)
                {
                    allVx[d * segs + s] = vx;
                    allVy[d * segs + s] = vy;
                }
                else
                {
                    for (int w = 0; w < worldCount; w++)
                    {
                        allVx[w * segs + s] = vx;
                        allVy[w * segs + s] = vy;
                    }
                }
            }
        }

        _cmdVx!.CopyFromCPU(allVx);
        _cmdVy!.CopyFromCPU(allVy);
        _cmdVxCPU = allVx;
        _cmdVyCPU = allVy;
    }

    /// <summary>
    /// Build the 3-body rocket (fuselage + 2 legs) in free space with a randomized initial
    /// state (tilt, velocity, angular velocity). One shared state for all worlds (fair CEM).
    /// </summary>
    private void CreateAndUploadRocketTemplate(int worldCount, int seed, int[]? perWorldSeeds = null)
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

        // Dummy collider (SharedColliderCount = 0 in MegaKernelConfig means it is never read).
        _worldState.UploadSharedColliders(new[]
        {
            new GPUOBBCollider { CX = 0f, CY = -1000f, UX = 1f, UY = 0f, HalfExtentX = 1f, HalfExtentY = 1f }
        });

        // Randomized initial state: one shared state (fair CEM), or one per world (replay).
        bool perWorld = perWorldSeeds != null;
        int distinct = perWorld ? worldCount : 1;
        var states = new (float tilt, float vx, float vy, float av)[distinct];
        for (int d = 0; d < distinct; d++)
        {
            var r = new Random(perWorld ? perWorldSeeds![d] : seed);
            float tilt = (float)(r.NextDouble() * SpawnAngleRange * 2 - SpawnAngleRange);
            double vang = r.NextDouble() * 2.0 * Math.PI;
            double vspd = r.NextDouble() * InitialSpeedMax;
            float vx = (float)(Math.Cos(vang) * vspd);
            float vy = (float)(Math.Sin(vang) * vspd);
            float av = (float)(r.NextDouble() * InitialAngVelMax * 2 - InitialAngVelMax);
            states[d] = (tilt, vx, vy, av);
        }

        const float spawnX = 0f, spawnY = 0f;

        for (int w = 0; w < worldCount; w++)
        {
            var st = states[perWorld ? w : 0];
            float spawnTilt = st.tilt;
            float velX = st.vx, velY = st.vy, angVel0 = st.av;
            float cosT = MathF.Cos(spawnTilt);
            float sinT = MathF.Sin(spawnTilt);

            float bodyX = spawnX;
            float bodyY = spawnY + bodyHalfLength * cosT;
            float bodyAngleFinal = bodyAngle + spawnTilt;

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
            float leftAngleFinal = leftLegAngle + spawnTilt;
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
            float rightAngleFinal = rightLegAngle + spawnTilt;
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

    // === Replay (visualization) ===
    private GPURigidBody[]? _replayBodies;
    private GPURigidBodyGeom[]? _replayGeoms;
    private byte[]? _replayTerminal;
    private float[]? _replayThrottle;
    private float[]? _replayGimbal;
    private int[]? _replaySteps;
    private int _replayCount;

    public int ReplayCount => _replayCount;

    /// <summary>
    /// Per-world step counters from the most recent ReadReplayState call.
    /// A world is a genuine tumble iff its counter is &lt; MaxSteps (the kernel also sets
    /// IsTerminal=1 on reaching MaxSteps, so the terminal flag alone cannot distinguish
    /// "tumbled early" from "survived the full episode").
    /// </summary>
    public int[] ReplaySteps => _replaySteps ?? Array.Empty<int>();

    /// <summary>
    /// Prepare a replay of ONE controller across `count` worlds, each with its own
    /// initial state and command schedule (seed = baseSeed + worldIdx).
    /// </summary>
    public void PrepareReplay(float[] controllerParams, int count, int baseSeed)
    {
        _replayCount = count;
        EnsureBuffers(count);

        int pc = _topology.TotalParams;
        if (controllerParams.Length != pc)
            throw new ArgumentException(
                $"controllerParams length {controllerParams.Length} != topology params {pc} ({_topology}).");

        var flat = new float[count * pc];
        for (int w = 0; w < count; w++)
            Array.Copy(controllerParams, 0, flat, w * pc, pc);
        SplitAndUpload(flat, count);

        var perWorldSeeds = new int[count];
        for (int w = 0; w < count; w++) perWorldSeeds[w] = baseSeed + w;

        CreateAndUploadRocketTemplate(count, baseSeed, perWorldSeeds);
        UploadCommandSchedule(count, baseSeed, perWorldSeeds);
        ResetEpisodeState(count);

        _replayBodies = new GPURigidBody[_worldState!.Config.TotalRigidBodies];
        _replayGeoms = new GPURigidBodyGeom[_worldState.Config.TotalGeoms];
        _replayTerminal = new byte[count];
        _replayThrottle = new float[count];
        _replayGimbal = new float[count];
        _replaySteps = new int[count];
    }

    public void StepReplay()
    {
        _fusedStepKernel(_replayCount, BuildPhysicsViews(), BuildNNViews(), BuildEpisodeViews(),
            BuildMegaConfig(), BuildControlConfig(), BuildControlViews(),
            _observations!.View, _actions!.View);
        _accelerator.Synchronize();
    }

    /// <summary>
    /// Read replay state. cmdVx/cmdVy are the per-world velocity command being tracked.
    /// </summary>
    public void ReadReplayState(
        out GPURigidBody[] bodies, out GPURigidBodyGeom[] geoms,
        out byte[] terminal, out float[] throttle, out float[] gimbal,
        out float[] cmdVx, out float[] cmdVy)
    {
        _worldState!.RigidBodies.CopyToCPU(_replayBodies!);
        _worldState.Geoms.CopyToCPU(_replayGeoms!);
        _isTerminal!.CopyToCPU(_replayTerminal!);
        _currentThrottle!.CopyToCPU(_replayThrottle!);
        _currentGimbal!.CopyToCPU(_replayGimbal!);
        _stepCounters!.CopyToCPU(_replaySteps!);

        int segs = SegmentsPerEpisode;
        var cvx = new float[_replayCount];
        var cvy = new float[_replayCount];
        for (int w = 0; w < _replayCount; w++)
        {
            int seg = _replaySteps![w] / SegmentLength;
            if (seg >= segs) seg = segs - 1;
            cvx[w] = _cmdVxCPU![w * segs + seg];
            cvy[w] = _cmdVyCPU![w * segs + seg];
        }

        bodies = _replayBodies!;
        geoms = _replayGeoms!;
        terminal = _replayTerminal!;
        throttle = _replayThrottle!;
        gimbal = _replayGimbal!;
        cmdVx = cvx;
        cmdVy = cvy;
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
        _waggleAccum?.Dispose();
        _settledSteps?.Dispose();
        _weights?.Dispose();
        _biases?.Dispose();
        _layerSizes.Dispose();
        _cmdVx?.Dispose();
        _cmdVy?.Dispose();
        _trackRewardAccum?.Dispose();
        _elmanContext?.Dispose();
        _accelerator.Dispose();
        _context.Dispose();   // ILGPU Context — disposed last (owns the accelerator)
    }
}
