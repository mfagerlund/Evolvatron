using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU.MegaKernel;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU.MegaKernel;
using Evolvatron.Evolvion.World;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated rocket landing evaluator using dense NN + fused mega-kernel.
/// Takes flat parameter vectors (weights + biases) from IslandOptimizer.
/// Drop-in replacement for GPURocketLandingMegaEvaluator when using CEM/ES/SNES.
///
/// The rocket template, physics, fitness function, and terminal conditions are
/// identical to GPURocketLandingMegaEvaluator — only the NN architecture differs
/// (dense fixed-topology vs sparse variable-topology).
/// </summary>
public class GPUDenseRocketLandingEvaluator : IDisposable
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
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hasLanded;
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;
    private MemoryBuffer1D<float, Stride1D.Dense>? _waggleAccum;
    private MemoryBuffer1D<int, Stride1D.Dense>? _settledSteps;

    // Zone buffers (Phase 3)
    private MemoryBuffer1D<GPUCheckpoint, Stride1D.Dense>? _checkpointsBuf;
    private MemoryBuffer1D<GPUDangerZone, Stride1D.Dense>? _dangerZonesBuf;
    private MemoryBuffer1D<GPUSpeedZone, Stride1D.Dense>? _speedZonesBuf;
    private MemoryBuffer1D<GPUAttractor, Stride1D.Dense>? _attractorsBuf;
    private MemoryBuffer1D<float, Stride1D.Dense>? _zoneRewardAccum;
    private MemoryBuffer1D<int, Stride1D.Dense>? _checkpointProgress;
    private MemoryBuffer1D<int, Stride1D.Dense>? _attractorContacted;

    // Kernels
    private readonly Action<Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
        MegaKernelConfig, DenseRocketNNConfig, ZoneViews,
        ArrayView<float>, ArrayView<float>> _fusedStepKernel;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _resetKernel;

    // Host caches
    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private byte[]? _landedCache;
    private float[]? _weightsCPU;
    private float[]? _biasesCPU;
    private int _lastWorldCount;
    private int _lastObstacleCount = -1;
    private int _lastSensorCount = -1;
    private int _lastZoneCount = -1;

    // Landing parameters (same defaults as GPURocketLandingMegaEvaluator)
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public float GroundY { get; set; } = -5f;
    public float SpawnHeight { get; set; } = 15f;
    public float SpawnHeightRange { get; set; } = 3f;
    public float PadX { get; set; } = 0f;
    private float? _padY;
    public float PadY
    {
        get => _padY ?? GroundY + 0.5f;
        set => _padY = value;
    }
    public float PadHalfWidth { get; set; } = 2f;
    public float PadHalfHeight { get; set; } = 0.25f;
    public float MaxLandingVel { get; set; } = 2f;
    public float MaxLandingAngle { get; set; } = 15f * MathF.PI / 180f;
    public float LandingBonus { get; set; } = 200f;
    public float HasteBonus { get; set; } = 1.0f;
    public float SettleSpeedThreshold { get; set; } = 0.3f;
    public float SettleAngVelThreshold { get; set; } = 0.5f;
    public int SettleStepsRequired { get; set; } = 30;   // ~0.25s at 120Hz
    public float SettleTipAngle { get; set; } = 45f * MathF.PI / 180f;

    // Difficulty knobs
    public float SpawnCenterX { get; set; } = 0f;
    public float SpawnXRange { get; set; } = 2f;
    public float SpawnXMin { get; set; } = 0f;
    public float SpawnAngleRange { get; set; } = 0f;
    public float InitialVelXRange { get; set; } = 1f;
    public float InitialVelYMax { get; set; } = 2f;

    public int SolverIterations { get; set; } = 6;
    public int MaxSteps { get; set; } = 600;

    // Obstacles and sensors
    public List<GPUOBBCollider> Obstacles { get; set; } = new();
    public int SensorCount { get; set; } = 0;
    public float MaxSensorRange { get; set; } = 30f;

    // Behaviour shaping
    public float WagglePenalty { get; set; } = 0f;
    public bool ObstacleDeathEnabled { get; set; } = false;
    public int SpawnCount { get; set; } = 10;
    public int SpawnSeed { get; set; } = 0;

    // Reward weights (defaults reproduce original hardcoded fitness function)
    public float RewardSurvivalWeight { get; set; } = 20f;
    public float RewardPositionWeight { get; set; } = 20f;
    public float RewardVelocityWeight { get; set; } = 5f;
    public float RewardAngleWeight { get; set; } = 5f;
    public float RewardAngVelWeight { get; set; } = 0f;

    // Reward zones (Phase 3)
    public List<GPUCheckpoint> Checkpoints { get; set; } = new();
    public List<GPUDangerZone> DangerZones { get; set; } = new();
    public List<GPUSpeedZone> SpeedZones { get; set; } = new();
    public List<GPUAttractor> Attractors { get; set; } = new();

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

    public GPUDenseRocketLandingEvaluator(DenseTopology topology)
    {
        _topology = topology;
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPUDenseRocketLandingEvaluator on: {_accelerator.Name} " +
                          $"({_accelerator.NumMultiprocessors} SMs, optimal pop={OptimalPopulationSize}), " +
                          $"topology={topology}");

        _fusedStepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
            MegaKernelConfig, DenseRocketNNConfig, ZoneViews,
            ArrayView<float>, ArrayView<float>>(
            DenseRocketLandingStepKernel.StepKernel);

        _resetKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>>(
            GPUBatchedRocketLandingKernels.ResetLandingStateKernel);

        _layerSizes = _accelerator.Allocate1D<int>(topology.NumLayers);
        _layerSizes.CopyFromCPU(topology.LayerSizes);
    }

    /// <summary>
    /// Evaluate population from flat parameter vectors with multiple spawn conditions.
    /// Returns aggregated fitness (solvedCount * maxSteps + meanFitness) per individual.
    /// </summary>
    public (float[] fitness, int landings, int maxLandingCount, int maxLandingIdx) EvaluateMultiSpawn(
        float[] paramVectors, int totalPop, int numSpawns, int baseSeed)
    {
        if (totalPop == 0)
            return (Array.Empty<float>(), 0, 0, 0);

        EnsureBuffers(totalPop);
        SplitAndUpload(paramVectors, totalPop);

        var nnConfig = BuildNNConfig();
        var nnViews = BuildNNViews();

        var totalFitness = new float[totalPop];
        var landingCounts = new int[totalPop];

        for (int spawn = 0; spawn < numSpawns; spawn++)
        {
            int seed = baseSeed + spawn;
            CreateAndUploadRocketTemplate(totalPop, seed);
            ResetEpisodeState(totalPop);

            var megaConfig = BuildMegaConfig();
            var physicsViews = BuildPhysicsViews();
            var episodeViews = BuildEpisodeViews();
            var zoneViews = BuildZoneViews();

            RunSimulationLoop(totalPop, megaConfig, nnConfig, physicsViews, nnViews, episodeViews, zoneViews);

            _accelerator.Synchronize();
            _fitnessValues!.CopyToCPU(_fitnessCache!);
            _hasLanded!.CopyToCPU(_landedCache!);

            for (int i = 0; i < totalPop; i++)
            {
                totalFitness[i] += _fitnessCache![i];
                if (_landedCache![i] != 0)
                    landingCounts[i]++;
            }
        }

        var result = new float[totalPop];
        int totalLandings = 0;
        for (int i = 0; i < totalPop; i++)
        {
            float meanFitness = totalFitness[i] / numSpawns;
            result[i] = landingCounts[i] * MaxSteps + meanFitness;
            totalLandings += landingCounts[i];
        }

        int maxLandingCount = 0;
        int maxLandingIdx = 0;
        for (int i = 0; i < totalPop; i++)
        {
            if (landingCounts[i] > maxLandingCount ||
                (landingCounts[i] == maxLandingCount && result[i] > result[maxLandingIdx]))
            {
                maxLandingCount = landingCounts[i];
                maxLandingIdx = i;
            }
        }

        return (result, totalLandings, maxLandingCount, maxLandingIdx);
    }

    /// <summary>
    /// Evaluate a single champion across multiple spawn conditions.
    /// Returns (landingCount, totalSpawns).
    /// </summary>
    public (int landings, int total) EvaluateChampion(float[] mu, int numSpawns, int baseSeed)
    {
        var (fitness, landings, _, _) = EvaluateMultiSpawn(mu, 1, numSpawns, baseSeed);
        return (landings, numSpawns);
    }

    /// <summary>
    /// Configure all evaluator properties from a loaded SimWorld.
    /// Single entry point where editor data enters the GPU pipeline.
    /// </summary>
    public void Configure(SimWorld world)
    {
        // Landing pad
        PadX = world.LandingPad.PadX;
        PadY = world.LandingPad.PadY;
        PadHalfWidth = world.LandingPad.PadHalfWidth;
        PadHalfHeight = world.LandingPad.PadHalfHeight;
        MaxLandingVel = world.LandingPad.MaxLandingVelocity;
        MaxLandingAngle = world.LandingPad.MaxLandingAngle;
        LandingBonus = world.LandingPad.LandingBonus;

        // Ground
        GroundY = world.GroundY;

        // Spawn — editor sends full-width XRange centered on (X,Y),
        // but C# uses XRange as half-width and Height as bottom edge
        SpawnCenterX = world.Spawn.X;
        SpawnHeight = world.Spawn.Y - world.Spawn.HeightRange / 2f;
        SpawnHeightRange = world.Spawn.HeightRange;
        SpawnXRange = world.Spawn.XRange / 2f;
        SpawnAngleRange = world.Spawn.AngleRange;
        InitialVelXRange = world.Spawn.VelXRange;
        InitialVelYMax = world.Spawn.VelYMax;
        if (world.Spawn.SpawnCount > 0)
            SpawnCount = world.Spawn.SpawnCount;
        SpawnSeed = world.Spawn.SpawnSeed;

        // Physics
        MaxThrust = world.SimulationConfig.MaxThrust;
        SolverIterations = world.SimulationConfig.SolverIterations;
        MaxSteps = world.SimulationConfig.MaxSteps;
        HasteBonus = world.SimulationConfig.HasteBonus;

        // Sensors
        SensorCount = world.SimulationConfig.SensorCount;

        // Obstacles
        Obstacles = world.Obstacles.Select(o => new GPUOBBCollider
        {
            CX = o.CX, CY = o.CY,
            UX = o.UX, UY = o.UY,
            HalfExtentX = o.HalfExtentX, HalfExtentY = o.HalfExtentY
        }).ToList();

        ObstacleDeathEnabled = world.Obstacles.Any(o => o.IsLethal);

        // Reward weights — editor provides relative scaling on base magnitudes
        RewardSurvivalWeight = 20f;
        RewardPositionWeight = 20f * world.RewardWeights.PositionWeight;
        RewardVelocityWeight = 5f * world.RewardWeights.VelocityWeight;
        RewardAngleWeight = 5f * world.RewardWeights.AngleWeight;
        RewardAngVelWeight = 5f * world.RewardWeights.AngularVelocityWeight;
        WagglePenalty = world.RewardWeights.ControlEffortWeight;

        // Reward zones (Phase 3)
        Checkpoints = world.Checkpoints.Select(c => new GPUCheckpoint
        {
            X = c.X, Y = c.Y, Radius = c.Radius,
            Order = c.Order, RewardBonus = c.RewardBonus,
            InfluenceRadius = c.InfluenceRadius
        }).ToList();

        // Danger zones: explicit + synthetic from obstacles with penalties
        DangerZones = world.DangerZones.Select(d => new GPUDangerZone
        {
            X = d.X, Y = d.Y,
            HalfExtentX = d.HalfExtentX, HalfExtentY = d.HalfExtentY,
            PenaltyPerStep = d.PenaltyPerStep,
            IsLethal = d.IsLethal ? 1 : 0,
            InfluenceRadius = d.InfluenceRadius
        }).ToList();

        // Obstacles with penalty/influence also act as danger zones (axis-aligned approx)
        foreach (var o in world.Obstacles.Where(o => o.PenaltyPerStep > 0 || o.InfluenceRadius > 0))
        {
            DangerZones.Add(new GPUDangerZone
            {
                X = o.CX, Y = o.CY,
                HalfExtentX = o.HalfExtentX, HalfExtentY = o.HalfExtentY,
                PenaltyPerStep = o.PenaltyPerStep,
                IsLethal = o.IsLethal ? 1 : 0,
                InfluenceRadius = o.InfluenceRadius
            });
        }

        SpeedZones = world.SpeedZones.Select(s => new GPUSpeedZone
        {
            X = s.X, Y = s.Y,
            HalfExtentX = s.HalfExtentX, HalfExtentY = s.HalfExtentY,
            MaxSpeed = s.MaxSpeed, RewardPerStep = s.RewardPerStep
        }).ToList();

        Attractors = world.Attractors.Select(a => new GPUAttractor
        {
            X = a.X, Y = a.Y,
            HalfExtentX = a.HalfExtentX, HalfExtentY = a.HalfExtentY,
            Magnitude = a.Magnitude, InfluenceRadius = a.InfluenceRadius,
            ContactBonus = a.ContactBonus
        }).ToList();

        // Landing pad attraction: standard attractor at pad position with influence falloff
        if (world.LandingPad.AttractionMagnitude > 0 && world.LandingPad.AttractionRadius > 0)
        {
            Attractors.Add(new GPUAttractor
            {
                X = world.LandingPad.PadX,
                Y = world.LandingPad.PadY,
                HalfExtentX = world.LandingPad.PadHalfWidth,
                HalfExtentY = world.LandingPad.PadHalfHeight,
                Magnitude = world.LandingPad.AttractionMagnitude,
                InfluenceRadius = world.LandingPad.AttractionRadius,
                ContactBonus = 0f
            });
        }
    }

    private DenseRocketNNConfig BuildNNConfig()
    {
        return new DenseRocketNNConfig
        {
            NumLayers = _topology.NumLayers,
            TotalWeightsPerNet = _topology.TotalWeights,
            TotalBiasesPerNet = _topology.TotalBiases
        };
    }

    private MegaKernelConfig BuildMegaConfig()
    {
        // Collider layout: [0]=ground, [1]=pad, [2..]=obstacles
        int totalColliders = 2 + Obstacles.Count;
        int maxContacts = 24 + Obstacles.Count * 4;
        int inputSize = 8 + SensorCount;

        return new MegaKernelConfig
        {
            BodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = totalColliders,
            MaxContactsPerWorld = maxContacts,
            Dt = 1f / 120f,
            GravityX = 0f,
            GravityY = -9.81f,
            FrictionMu = 0.8f,
            Restitution = 0.0f,
            GlobalDamping = 0.02f,
            AngularDamping = 0.1f,
            SolverIterations = SolverIterations,
            TotalNodes = 0, // not used by dense kernel
            TotalEdges = 0,
            NumRows = 0,
            InputSize = inputSize,
            OutputSize = _topology.OutputSize,
            MaxSteps = MaxSteps,
            MaxThrust = MaxThrust,
            MaxGimbalTorque = MaxGimbalTorque,
            PadX = PadX,
            PadY = PadY,
            PadHalfWidth = PadHalfWidth,
            MaxLandingVel = MaxLandingVel,
            MaxLandingAngle = MaxLandingAngle,
            SettleSpeedThreshold = SettleSpeedThreshold,
            SettleAngVelThreshold = SettleAngVelThreshold,
            SettleStepsRequired = SettleStepsRequired,
            SettleTipAngle = SettleTipAngle,
            GroundY = GroundY,
            SpawnHeight = SpawnHeight,
            LandingBonus = LandingBonus,
            HasteBonus = HasteBonus,
            SensorCount = SensorCount,
            MaxSensorRange = MaxSensorRange,
            WagglePenalty = WagglePenalty,
            ObstacleDeathEnabled = ObstacleDeathEnabled ? 1 : 0,
            FirstObstacleIndex = 2, // [0]=ground, [1]=pad, [2..]=obstacles
            RewardSurvivalWeight = RewardSurvivalWeight,
            RewardPositionWeight = RewardPositionWeight,
            RewardVelocityWeight = RewardVelocityWeight,
            RewardAngleWeight = RewardAngleWeight,
            RewardAngVelWeight = RewardAngVelWeight,
            CheckpointCount = Checkpoints.Count,
            DangerZoneCount = DangerZones.Count,
            SpeedZoneCount = SpeedZones.Count,
            AttractorCount = Attractors.Count
        };
    }

    private DenseNNViews BuildNNViews()
    {
        return new DenseNNViews
        {
            Weights = _weights!.View,
            Biases = _biases!.View,
            LayerSizes = _layerSizes.View
        };
    }

    private PhysicsViews BuildPhysicsViews()
    {
        return new PhysicsViews
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
    }

    private EpisodeViews BuildEpisodeViews()
    {
        return new EpisodeViews
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
    }

    private ZoneViews BuildZoneViews()
    {
        return new ZoneViews
        {
            Checkpoints = _checkpointsBuf!.View,
            DangerZones = _dangerZonesBuf!.View,
            SpeedZones = _speedZonesBuf!.View,
            Attractors = _attractorsBuf!.View,
            ZoneRewardAccum = _zoneRewardAccum!.View,
            CheckpointProgress = _checkpointProgress!.View,
            AttractorContacted = _attractorContacted!.View
        };
    }

    private void RunSimulationLoop(int worldCount,
        MegaKernelConfig megaConfig, DenseRocketNNConfig nnConfig,
        PhysicsViews physicsViews, DenseNNViews nnViews, EpisodeViews episodeViews,
        ZoneViews zoneViews)
    {
        for (int step = 0; step < MaxSteps; step++)
        {
            if (step > 0 && step % 30 == 0)
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
                nnConfig,
                zoneViews,
                _observations!.View,
                _actions!.View);
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
            _stepCounters!.View,
            _isTerminal!.View,
            _hasLanded!.View,
            _currentThrottle!.View,
            _currentGimbal!.View,
            _fitnessValues!.View);
        _waggleAccum!.MemSetToZero();
        _settledSteps!.MemSetToZero();
        _zoneRewardAccum!.MemSetToZero();
        _checkpointProgress!.MemSetToZero();
        _attractorContacted!.MemSetToZero();
        _worldState!.ClearContactCounts();
        _worldState.ClearContactCache();
        _accelerator.Synchronize();
    }

    private int TotalZoneCount => Checkpoints.Count + DangerZones.Count + SpeedZones.Count + Attractors.Count;

    private void EnsureBuffers(int worldCount)
    {
        int currentZoneCount = TotalZoneCount;
        if (_lastWorldCount == worldCount && _lastObstacleCount == Obstacles.Count
            && _lastSensorCount == SensorCount && _lastZoneCount == currentZoneCount
            && _worldState != null)
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
        _checkpointsBuf?.Dispose();
        _dangerZonesBuf?.Dispose();
        _speedZonesBuf?.Dispose();
        _attractorsBuf?.Dispose();
        _zoneRewardAccum?.Dispose();
        _checkpointProgress?.Dispose();
        _attractorContacted?.Dispose();

        int totalColliders = 2 + Obstacles.Count;
        int maxContacts = 24 + Obstacles.Count * 4;
        int inputSize = 8 + SensorCount;

        var worldConfig = new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = totalColliders,
            TargetsPerWorld = 0,
            MaxContactsPerWorld = maxContacts
        };

        _worldState = new GPUBatchedWorldState(_accelerator, worldConfig);

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

        // Zone buffers — allocate at least 1 element (ILGPU requires non-empty views)
        _checkpointsBuf = _accelerator.Allocate1D<GPUCheckpoint>(Math.Max(1, Checkpoints.Count));
        _dangerZonesBuf = _accelerator.Allocate1D<GPUDangerZone>(Math.Max(1, DangerZones.Count));
        _speedZonesBuf = _accelerator.Allocate1D<GPUSpeedZone>(Math.Max(1, SpeedZones.Count));
        _attractorsBuf = _accelerator.Allocate1D<GPUAttractor>(Math.Max(1, Attractors.Count));
        _zoneRewardAccum = _accelerator.Allocate1D<float>(worldCount);
        _checkpointProgress = _accelerator.Allocate1D<int>(worldCount);
        _attractorContacted = _accelerator.Allocate1D<int>(worldCount);

        // Upload zone definitions
        if (Checkpoints.Count > 0)
            _checkpointsBuf.CopyFromCPU(Checkpoints.ToArray());
        if (DangerZones.Count > 0)
            _dangerZonesBuf.CopyFromCPU(DangerZones.ToArray());
        if (SpeedZones.Count > 0)
            _speedZonesBuf.CopyFromCPU(SpeedZones.ToArray());
        if (Attractors.Count > 0)
            _attractorsBuf.CopyFromCPU(Attractors.ToArray());

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _landedCache = new byte[worldCount];
        _lastWorldCount = worldCount;
        _lastObstacleCount = Obstacles.Count;
        _lastSensorCount = SensorCount;
        _lastZoneCount = currentZoneCount;
    }

    // === Replay support (step-by-step visualization, N rockets) ===

    private GPURigidBody[]? _replayBodies;
    private GPURigidBodyGeom[]? _replayGeoms;
    private float[]? _replayThrottle;
    private float[]? _replayGimbal;
    private int _replayCount;

    /// <summary>
    /// Prepare replay for a single champion (backward-compatible).
    /// </summary>
    public void PrepareReplay(float[] championParams, int seed)
    {
        PrepareMultiReplay(championParams, 1, seed);
    }

    /// <summary>
    /// Prepare replay for N rockets simultaneously.
    /// flatParams = [rocket0_params, rocket1_params, ...] (length = count * paramCount).
    /// </summary>
    public void PrepareMultiReplay(float[] flatParams, int count, int seed)
    {
        _replayCount = count;
        EnsureBuffers(count);
        SplitAndUpload(flatParams, count);
        CreateAndUploadRocketTemplate(count, seed);
        ResetEpisodeState(count);

        int totalBodies = _worldState!.Config.TotalRigidBodies;
        int totalGeoms = _worldState.Config.TotalGeoms;
        if (_replayBodies == null || _replayBodies.Length != totalBodies)
        {
            _replayBodies = new GPURigidBody[totalBodies];
            _replayGeoms = new GPURigidBodyGeom[totalGeoms];
        }
        _replayThrottle = new float[count];
        _replayGimbal = new float[count];
    }

    public void StepReplay()
    {
        _fusedStepKernel(_replayCount, BuildPhysicsViews(), BuildNNViews(), BuildEpisodeViews(),
            BuildMegaConfig(), BuildNNConfig(), BuildZoneViews(),
            _observations!.View, _actions!.View);
        _accelerator.Synchronize();
    }

    /// <summary>
    /// Read state for all replay rockets. Arrays are sized for _replayCount worlds.
    /// </summary>
    public void ReadMultiReplayState(out GPURigidBody[] bodies, out GPURigidBodyGeom[] geoms,
        out byte[] terminal, out byte[] landed, out float[] throttle, out float[] gimbal)
    {
        _worldState!.RigidBodies.CopyToCPU(_replayBodies!);
        _worldState.Geoms.CopyToCPU(_replayGeoms!);
        bodies = _replayBodies!;
        geoms = _replayGeoms!;

        _isTerminal!.CopyToCPU(_terminalCache!);
        _hasLanded!.CopyToCPU(_landedCache!);
        _currentThrottle!.CopyToCPU(_replayThrottle!);
        _currentGimbal!.CopyToCPU(_replayGimbal!);

        terminal = _terminalCache!;
        landed = _landedCache!;
        throttle = _replayThrottle!;
        gimbal = _replayGimbal!;
    }

    /// <summary>
    /// Backward-compatible single-rocket read.
    /// </summary>
    public void ReadReplayState(out GPURigidBody[] bodies, out GPURigidBodyGeom[] geoms,
        out bool isTerminal, out bool hasLanded, out float throttle, out float gimbal)
    {
        ReadMultiReplayState(out bodies, out geoms,
            out var terminal, out var landed, out var throttles, out var gimbals);
        isTerminal = terminal[0] != 0;
        hasLanded = landed[0] != 0;
        throttle = throttles[0];
        gimbal = gimbals[0];
    }

    // Rocket template creation — identical to GPURocketLandingMegaEvaluator
    /// <summary>
    /// Prepare replay showing a single individual from all training spawn positions.
    /// Each rocket gets a different spawn seed (baseSeed + rocketIndex).
    /// </summary>
    public void PrepareSpawnSpreadReplay(float[] singleParams, int numSpawns, int baseSeed)
    {
        _replayCount = numSpawns;
        int paramCount = _topology.TotalParams;
        var flatParams = new float[numSpawns * paramCount];
        for (int s = 0; s < numSpawns; s++)
            Array.Copy(singleParams, 0, flatParams, s * paramCount, paramCount);

        EnsureBuffers(numSpawns);
        SplitAndUpload(flatParams, numSpawns);
        CreateAndUploadRocketTemplate(numSpawns, baseSeed, perWorldSeed: true);
        ResetEpisodeState(numSpawns);

        int totalBodies = _worldState!.Config.TotalRigidBodies;
        int totalGeoms = _worldState.Config.TotalGeoms;
        if (_replayBodies == null || _replayBodies.Length != totalBodies)
        {
            _replayBodies = new GPURigidBody[totalBodies];
            _replayGeoms = new GPURigidBodyGeom[totalGeoms];
        }
        _replayThrottle = new float[numSpawns];
        _replayGimbal = new float[numSpawns];
    }

    private void CreateAndUploadRocketTemplate(int worldCount, int seed, bool perWorldSeed = false)
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

        // Collider layout: [0]=ground, [1]=pad, [2..]=obstacles
        var allColliders = new GPUOBBCollider[2 + Obstacles.Count];
        allColliders[0] = new GPUOBBCollider { CX = 0f, CY = GroundY, UX = 1f, UY = 0f, HalfExtentX = 30f, HalfExtentY = 0.5f };
        allColliders[1] = new GPUOBBCollider { CX = PadX, CY = PadY, UX = 1f, UY = 0f, HalfExtentX = PadHalfWidth, HalfExtentY = PadHalfHeight };
        for (int i = 0; i < Obstacles.Count; i++)
            allColliders[2 + i] = Obstacles[i];
        _worldState.UploadSharedColliders(allColliders);

        // Pre-generate spawn positions: one position per world in perWorldSeed mode,
        // or one shared position for all worlds in training mode (fair CEM comparison).
        var spawnPositions = new (float x, float y, float tilt, float vx, float vy)[
            perWorldSeed ? worldCount : 1];

        for (int s = 0; s < spawnPositions.Length; s++)
        {
            var posRng = perWorldSeed ? new Random(seed + s) : (s == 0 ? rng : rng);
            float sx;
            if (SpawnXMin > 0f)
            {
                float side = posRng.NextDouble() < 0.5 ? -1f : 1f;
                sx = SpawnCenterX + side * (SpawnXMin + (float)(posRng.NextDouble() * (SpawnXRange - SpawnXMin)));
            }
            else
            {
                sx = SpawnCenterX + (float)(posRng.NextDouble() * SpawnXRange * 2 - SpawnXRange);
            }
            float sy = SpawnHeight + (float)(posRng.NextDouble() * SpawnHeightRange);
            float st = (float)(posRng.NextDouble() * SpawnAngleRange * 2 - SpawnAngleRange);
            float svx = (float)(posRng.NextDouble() * InitialVelXRange * 2 - InitialVelXRange);
            float svy = (float)(posRng.NextDouble() * -InitialVelYMax);
            spawnPositions[s] = (sx, sy, st, svx, svy);
        }

        for (int w = 0; w < worldCount; w++)
        {
            var sp = spawnPositions[perWorldSeed ? w : 0];
            float spawnX = sp.x;
            float spawnY = sp.y;
            float spawnTilt = sp.tilt;
            float velX = sp.vx;
            float velY = sp.vy;

            float centerY = spawnY;
            float cosT = MathF.Cos(spawnTilt);
            float sinT = MathF.Sin(spawnTilt);

            float bodyX = spawnX;
            float bodyY = centerY + bodyHalfLength * cosT;
            float bodyAngleFinal = bodyAngle + spawnTilt;

            allBodies[config.GetRigidBodyIndex(w, 0)] = new GPURigidBody
            {
                X = bodyX, Y = bodyY, Angle = bodyAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = bodyX, PrevY = bodyY, PrevAngle = bodyAngleFinal,
                InvMass = 1f / bodyMass, InvInertia = 1f / bodyInertia,
                GeomStartIndex = 0, GeomCount = bodyGeomCount
            };

            float leftLegX = spawnX + leftOffX * cosT - leftOffY * sinT;
            float leftLegY = centerY + leftOffX * sinT + leftOffY * cosT;
            float leftAngleFinal = leftLegAngle + spawnTilt;
            allBodies[config.GetRigidBodyIndex(w, 1)] = new GPURigidBody
            {
                X = leftLegX, Y = leftLegY, Angle = leftAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = leftLegX, PrevY = leftLegY, PrevAngle = leftAngleFinal,
                InvMass = 1f / legMass, InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount, GeomCount = legGeomCount
            };

            float rightLegX = spawnX + rightOffX * cosT - rightOffY * sinT;
            float rightLegY = centerY + rightOffX * sinT + rightOffY * cosT;
            float rightAngleFinal = rightLegAngle + spawnTilt;
            allBodies[config.GetRigidBodyIndex(w, 2)] = new GPURigidBody
            {
                X = rightLegX, Y = rightLegY, Angle = rightAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
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
        _hasLanded?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _waggleAccum?.Dispose();
        _settledSteps?.Dispose();
        _weights?.Dispose();
        _biases?.Dispose();
        _layerSizes.Dispose();
        _checkpointsBuf?.Dispose();
        _dangerZonesBuf?.Dispose();
        _speedZonesBuf?.Dispose();
        _attractorsBuf?.Dispose();
        _zoneRewardAccum?.Dispose();
        _checkpointProgress?.Dispose();
        _attractorContacted?.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }
}
