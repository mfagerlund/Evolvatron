using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU.MegaKernel;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU.MegaKernel;

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

    // Kernels
    private readonly Action<Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
        MegaKernelConfig, DenseRocketNNConfig,
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

    // Landing parameters (same defaults as GPURocketLandingMegaEvaluator)
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

    public int SolverIterations { get; set; } = 6;
    public int MaxSteps { get; set; } = 600;

    // Obstacles and sensors
    public List<GPUOBBCollider> Obstacles { get; set; } = new();
    public int SensorCount { get; set; } = 0;
    public float MaxSensorRange { get; set; } = 30f;

    // Behaviour shaping
    public float WagglePenalty { get; set; } = 0f;
    public bool ObstacleDeathEnabled { get; set; } = false;

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
            MegaKernelConfig, DenseRocketNNConfig,
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
    public (float[] fitness, int landings) EvaluateMultiSpawn(
        float[] paramVectors, int totalPop, int numSpawns, int baseSeed)
    {
        if (totalPop == 0)
            return (Array.Empty<float>(), 0);

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

            RunSimulationLoop(totalPop, megaConfig, nnConfig, physicsViews, nnViews, episodeViews);

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

        return (result, totalLandings);
    }

    /// <summary>
    /// Evaluate a single champion across multiple spawn conditions.
    /// Returns (landingCount, totalSpawns).
    /// </summary>
    public (int landings, int total) EvaluateChampion(float[] mu, int numSpawns, int baseSeed)
    {
        var (fitness, landings) = EvaluateMultiSpawn(mu, 1, numSpawns, baseSeed);
        return (landings, numSpawns);
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
        int totalColliders = 1 + Obstacles.Count;
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
            GroundY = GroundY,
            SpawnHeight = SpawnHeight,
            LandingBonus = LandingBonus,
            SensorCount = SensorCount,
            MaxSensorRange = MaxSensorRange,
            WagglePenalty = WagglePenalty,
            ObstacleDeathEnabled = ObstacleDeathEnabled ? 1 : 0
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
            WaggleAccum = _waggleAccum!.View
        };
    }

    private void RunSimulationLoop(int worldCount,
        MegaKernelConfig megaConfig, DenseRocketNNConfig nnConfig,
        PhysicsViews physicsViews, DenseNNViews nnViews, EpisodeViews episodeViews)
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
        _worldState!.ClearContactCounts();
        _worldState.ClearContactCache();
        _accelerator.Synchronize();
    }

    private void EnsureBuffers(int worldCount)
    {
        if (_lastWorldCount == worldCount && _lastObstacleCount == Obstacles.Count
            && _lastSensorCount == SensorCount && _worldState != null)
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
        _weights?.Dispose();
        _biases?.Dispose();

        int totalColliders = 1 + Obstacles.Count;
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
        _weights = _accelerator.Allocate1D<float>(worldCount * _topology.TotalWeights);
        _biases = _accelerator.Allocate1D<float>(worldCount * _topology.TotalBiases);

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _landedCache = new byte[worldCount];
        _lastWorldCount = worldCount;
        _lastObstacleCount = Obstacles.Count;
        _lastSensorCount = SensorCount;
    }

    // Rocket template creation — identical to GPURocketLandingMegaEvaluator
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

        var allColliders = new GPUOBBCollider[1 + Obstacles.Count];
        allColliders[0] = new GPUOBBCollider { CX = 0f, CY = GroundY, UX = 1f, UY = 0f, HalfExtentX = 30f, HalfExtentY = 0.5f };
        for (int i = 0; i < Obstacles.Count; i++)
            allColliders[1 + i] = Obstacles[i];
        _worldState.UploadSharedColliders(allColliders);

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
        _weights?.Dispose();
        _biases?.Dispose();
        _layerSizes.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }
}
