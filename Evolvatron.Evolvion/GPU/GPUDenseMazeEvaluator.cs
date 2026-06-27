using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU.MegaKernel;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// Phase-2 maze navigator evaluator (see docs/phase2_maze_spec.md).
///
/// Evolves a maze policy (NN#1, per-world) that commands a gentle velocity to a FROZEN
/// Phase-1 controller (NN#2, shared weights loaded once from controller_easy.bin). Trains on
/// several procedurally-generated maze layouts per generation for generalization.
/// </summary>
public class GPUDenseMazeEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly DenseTopology _mazeTopology;     // NN#1 (evolved)
    private readonly DenseTopology _ctrlTopology;     // NN#2 (frozen)
    private readonly float[] _ctrlParams;             // frozen controller params (weights+biases flat)
    private bool _disposed;

    // Maze policy weights (per-world, evolved)
    private MemoryBuffer1D<float, Stride1D.Dense>? _mazeWeights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _mazeBiases;
    private MemoryBuffer1D<int, Stride1D.Dense> _mazeLayerSizes;

    // Frozen controller weights (shared — ONE net)
    private MemoryBuffer1D<float, Stride1D.Dense> _ctrlWeights;
    private MemoryBuffer1D<float, Stride1D.Dense> _ctrlBiases;
    private MemoryBuffer1D<int, Stride1D.Dense> _ctrlLayerSizes;

    private GPUBatchedWorldState? _worldState;

    // Episode buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _mazeObs;
    private MemoryBuffer1D<float, Stride1D.Dense>? _mazeAct;
    private MemoryBuffer1D<float, Stride1D.Dense>? _ctrlObs;
    private MemoryBuffer1D<float, Stride1D.Dense>? _ctrlAct;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentThrottle;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentGimbal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hasLanded;   // reused as goal-reached flag
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;
    private MemoryBuffer1D<float, Stride1D.Dense>? _waggleAccum;  // unused; kept for EpisodeViews
    private MemoryBuffer1D<int, Stride1D.Dense>? _settledSteps;   // unused; kept for EpisodeViews
    private MemoryBuffer1D<float, Stride1D.Dense>? _goalX;
    private MemoryBuffer1D<float, Stride1D.Dense>? _goalY;
    private MemoryBuffer1D<float, Stride1D.Dense>? _prevDist;
    private MemoryBuffer1D<float, Stride1D.Dense>? _rewardAccum;

    private readonly Action<Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
        MegaKernelConfig, DenseMazeConfig, MazeViews,
        ArrayView<float>, ArrayView<float>> _fusedStepKernel;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<float>, ArrayView<float>, ArrayView<float>> _resetKernel;

    // Host caches
    private byte[]? _terminalCache;
    private byte[]? _landedCache;
    private float[]? _fitnessCache;
    private float[]? _mazeWeightsCPU;
    private float[]? _mazeBiasesCPU;
    private int _lastWorldCount;

    // === Physics params (parity with the Phase-1 control evaluator) ===
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public int SolverIterations { get; set; } = 6;
    public int MaxSteps { get; set; } = 600;

    // === Task params ===
    public float CmdSpeedMax { get; set; } = 3f;       // gentle command envelope (Phase-1 sweet spot)
    public float GoalRadius { get; set; } = 0.75f;     // COM must reach near the goal CENTER (not just its edge)
    public float PosScale { get; set; } = 20f;
    public float MaxSensorRange { get; set; } = 15f;
    public int SensorCount { get; set; } = 0;          // 0 = no sensors (open space); 4 or 8 with obstacles
    public int NumObstacles { get; set; } = 0;         // procedural maze difficulty

    // === Reward ===
    public float ProgressWeight { get; set; } = 1f;
    public float GoalBonus { get; set; } = 200f;
    public float CollisionPenalty { get; set; } = 100f;
    public float TumblePenalty { get; set; } = 100f;
    public float StepPenalty { get; set; } = 0.01f;

    public DenseTopology MazeTopology => _mazeTopology;
    public Accelerator Accelerator => _accelerator;
    public int MazeInputSize => 6 + (SensorCount >= 4 ? (SensorCount >= 8 ? 8 : 4) : 0);

    public int OptimalPopulationSize
    {
        get
        {
            int sms = _accelerator.NumMultiprocessors;
            int warpSize = _accelerator.WarpSize;
            return sms * 4 * warpSize;
        }
    }

    public GPUDenseMazeEvaluator(DenseTopology mazeTopology, DenseTopology ctrlTopology, float[] ctrlParams)
    {
        const int gpuMaxLayerWidth = 64;
        if (mazeTopology.MaxLayerWidth > gpuMaxLayerWidth || ctrlTopology.MaxLayerWidth > gpuMaxLayerWidth)
            throw new ArgumentException(
                $"NN layer width exceeds GPU kernel limit {gpuMaxLayerWidth}. maze={mazeTopology}, ctrl={ctrlTopology}");
        if (ctrlTopology.InputSize != 9 || ctrlTopology.OutputSize != 2)
            throw new ArgumentException(
                $"Frozen controller must be a plain Phase-1 controller (9→…→2). Got {ctrlTopology}.");
        if (ctrlParams.Length != ctrlTopology.TotalParams)
            throw new ArgumentException(
                $"Frozen controller params length {ctrlParams.Length} != topology {ctrlTopology.TotalParams} ({ctrlTopology}).");
        if (mazeTopology.OutputSize != 2)
            throw new ArgumentException($"Maze policy must output 2 (velocity command). Got {mazeTopology}.");

        _mazeTopology = mazeTopology;
        _ctrlTopology = ctrlTopology;
        _ctrlParams = (float[])ctrlParams.Clone();

        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));
        var cudaDevice = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPUDenseMazeEvaluator on: {_accelerator.Name} " +
                          $"({_accelerator.NumMultiprocessors} SMs, optimal pop={OptimalPopulationSize})");
        Console.WriteLine($"  Maze policy (evolved): {mazeTopology}");
        Console.WriteLine($"  Controller (frozen):   {ctrlTopology}");

        _fusedStepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, PhysicsViews, DenseNNViews, EpisodeViews,
            MegaKernelConfig, DenseMazeConfig, MazeViews,
            ArrayView<float>, ArrayView<float>>(DenseMazeStepKernel.StepKernel);

        _resetKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            GPUBatchedRocketLandingKernels.ResetLandingStateKernel);

        _mazeLayerSizes = _accelerator.Allocate1D<int>(mazeTopology.NumLayers);
        _mazeLayerSizes.CopyFromCPU(mazeTopology.LayerSizes);

        // Frozen controller: split flat params into weights+biases, upload ONCE (shared net).
        _ctrlLayerSizes = _accelerator.Allocate1D<int>(ctrlTopology.NumLayers);
        _ctrlLayerSizes.CopyFromCPU(ctrlTopology.LayerSizes);
        var cw = new float[ctrlTopology.TotalWeights];
        var cb = new float[ctrlTopology.TotalBiases];
        Array.Copy(_ctrlParams, 0, cw, 0, ctrlTopology.TotalWeights);
        Array.Copy(_ctrlParams, ctrlTopology.TotalWeights, cb, 0, ctrlTopology.TotalBiases);
        _ctrlWeights = _accelerator.Allocate1D<float>(ctrlTopology.TotalWeights);
        _ctrlBiases = _accelerator.Allocate1D<float>(ctrlTopology.TotalBiases);
        _ctrlWeights.CopyFromCPU(cw);
        _ctrlBiases.CopyFromCPU(cb);
    }

    /// <summary>
    /// Evaluate a population across multiple procedurally-generated mazes.
    /// Returns aggregated fitness (solvedMazeCount * GoalBonus + meanReward) per individual.
    /// </summary>
    public (float[] fitness, int totalSolved, int maxSolvedCount, int maxSolvedIdx) EvaluateMultiMaze(
        float[] paramVectors, int totalPop, int numMazes, int baseSeed)
    {
        if (totalPop == 0) return (Array.Empty<float>(), 0, 0, 0);

        EnsureBuffers(totalPop);
        SplitAndUpload(paramVectors, totalPop);

        var mazeConfig = BuildMazeConfig();
        var mazeNNViews = BuildMazeNNViews();

        var totalFitness = new float[totalPop];
        var solvedCounts = new int[totalPop];

        for (int m = 0; m < numMazes; m++)
        {
            var maze = GenerateMaze(baseSeed + m, NumObstacles);
            UploadMaze(totalPop, maze);
            ResetEpisodeState(totalPop);

            var megaConfig = BuildMegaConfig(maze.obstacles.Length);
            var physicsViews = BuildPhysicsViews();
            var episodeViews = BuildEpisodeViews();
            var mazeViews = BuildMazeViews();

            RunSimulationLoop(totalPop, megaConfig, mazeConfig, physicsViews, mazeNNViews, episodeViews, mazeViews);

            _accelerator.Synchronize();
            _fitnessValues!.CopyToCPU(_fitnessCache!);
            _hasLanded!.CopyToCPU(_landedCache!);

            for (int i = 0; i < totalPop; i++)
            {
                totalFitness[i] += _fitnessCache![i];
                if (_landedCache![i] != 0) solvedCounts[i]++;
            }
        }

        var result = new float[totalPop];
        int totalSolved = 0;
        for (int i = 0; i < totalPop; i++)
        {
            float meanFitness = totalFitness[i] / numMazes;
            result[i] = solvedCounts[i] * GoalBonus + meanFitness;
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

    public (int solved, int total) EvaluateChampion(float[] mu, int numMazes, int baseSeed)
    {
        var (_, solved, _, _) = EvaluateMultiMaze(mu, 1, numMazes, baseSeed);
        return (solved, numMazes);
    }

    // === Procedural maze generation ===
    public struct Maze
    {
        public float sx, sy, gx, gy;
        public GPUOBBCollider[] obstacles;
    }

    private Maze GenerateMaze(int seed, int numObstacles)
    {
        var r = new Random(seed * 2 + 1);
        float sx = (float)(r.NextDouble() * 4 - 2);       // [-2, 2]
        float sy = (float)(8 + r.NextDouble() * 2);       // [8, 10]
        float gx = (float)(r.NextDouble() * 20 - 10);     // [-10, 10]
        float gy = (float)(2 + r.NextDouble() * 8);       // [2, 10]

        var obstacles = new GPUOBBCollider[numObstacles];
        int placed = 0, attempts = 0;
        while (placed < numObstacles && attempts < numObstacles * 20)
        {
            attempts++;
            float ox = (float)(r.NextDouble() * 20 - 10);
            float oy = (float)(r.NextDouble() * 10 + 1);
            float hx = (float)(0.5 + r.NextDouble() * 1.5);
            float hy = (float)(0.5 + r.NextDouble() * 1.5);
            // keep clear of start and goal
            if (Dist(ox, oy, sx, sy) < hx + hy + 2f) continue;
            if (Dist(ox, oy, gx, gy) < hx + hy + 2f) continue;
            obstacles[placed++] = new GPUOBBCollider
            {
                CX = ox, CY = oy, UX = 1f, UY = 0f, HalfExtentX = hx, HalfExtentY = hy
            };
        }
        if (placed < numObstacles) Array.Resize(ref obstacles, placed);

        return new Maze { sx = sx, sy = sy, gx = gx, gy = gy, obstacles = obstacles };
    }

    private static float Dist(float ax, float ay, float bx, float by)
    {
        float dx = ax - bx, dy = ay - by;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>Scatter non-overlapping axis-aligned obstacle boxes across the flight region (no fixed start/goal).</summary>
    private GPUOBBCollider[] GenerateObstacleField(int seed, int numObstacles)
    {
        if (numObstacles <= 0) return Array.Empty<GPUOBBCollider>();
        var r = new Random(seed * 2 + 1);
        var obstacles = new GPUOBBCollider[numObstacles];
        int placed = 0, attempts = 0;
        while (placed < numObstacles && attempts < numObstacles * 40)
        {
            attempts++;
            float ox = (float)(r.NextDouble() * 20 - 10);
            float oy = (float)(r.NextDouble() * 10 + 1);
            float hx = (float)(0.5 + r.NextDouble() * 1.5);
            float hy = (float)(0.5 + r.NextDouble() * 1.5);
            bool ok = true;
            for (int k = 0; k < placed; k++)
                if (Dist(ox, oy, obstacles[k].CX, obstacles[k].CY) < hx + obstacles[k].HalfExtentX + 1.5f) { ok = false; break; }
            if (!ok) continue;
            obstacles[placed++] = new GPUOBBCollider { CX = ox, CY = oy, UX = 1f, UY = 0f, HalfExtentX = hx, HalfExtentY = hy };
        }
        if (placed < numObstacles) Array.Resize(ref obstacles, placed);
        return obstacles;
    }

    /// <summary>True if (x,y) is outside every obstacle box expanded by margin (boxes are axis-aligned).</summary>
    private static bool ClearOfObstacles(float x, float y, GPUOBBCollider[] obstacles, float margin)
    {
        for (int k = 0; k < obstacles.Length; k++)
            if (MathF.Abs(x - obstacles[k].CX) < obstacles[k].HalfExtentX + margin &&
                MathF.Abs(y - obstacles[k].CY) < obstacles[k].HalfExtentY + margin)
                return false;
        return true;
    }

    private DenseMazeConfig BuildMazeConfig() => new()
    {
        MazeNumLayers = _mazeTopology.NumLayers,
        MazeTotalWeights = _mazeTopology.TotalWeights,
        MazeTotalBiases = _mazeTopology.TotalBiases,
        MazeInputSize = _mazeTopology.InputSize,
        MazeOutputSize = _mazeTopology.OutputSize,
        CtrlNumLayers = _ctrlTopology.NumLayers,
        CtrlTotalWeights = _ctrlTopology.TotalWeights,
        CtrlTotalBiases = _ctrlTopology.TotalBiases,
        CmdSpeedMax = CmdSpeedMax,
        GoalRadius = GoalRadius,
        PosScale = PosScale,   // note: goal POSITION is per-world via MazeViews.GoalX/GoalY
        ProgressWeight = ProgressWeight,
        GoalBonus = GoalBonus,
        CollisionPenalty = CollisionPenalty,
        TumblePenalty = TumblePenalty,
        StepPenalty = StepPenalty
    };

    private MegaKernelConfig BuildMegaConfig(int obstacleCount) => new()
    {
        BodiesPerWorld = 3,
        GeomsPerWorld = 19,
        JointsPerWorld = 2,
        SharedColliderCount = obstacleCount,    // kernel reads exactly this many obstacle OBBs
        MaxContactsPerWorld = 24 + obstacleCount * 4,
        Dt = 1f / 120f,
        GravityX = 0f,
        GravityY = -9.81f,
        FrictionMu = 0.8f,
        Restitution = 0.0f,
        GlobalDamping = 0.02f,
        AngularDamping = 0.1f,
        SolverIterations = SolverIterations,
        InputSize = _mazeTopology.InputSize,
        OutputSize = _mazeTopology.OutputSize,
        MaxSteps = MaxSteps,
        MaxThrust = MaxThrust,
        MaxGimbalTorque = MaxGimbalTorque,
        SensorCount = SensorCount,
        MaxSensorRange = MaxSensorRange,
        ObstacleDeathEnabled = obstacleCount > 0 ? 1 : 0,
        FirstObstacleIndex = 0   // all shared colliders are obstacles in the maze task
    };

    private DenseNNViews BuildMazeNNViews() => new()
    {
        Weights = _mazeWeights!.View,
        Biases = _mazeBiases!.View,
        LayerSizes = _mazeLayerSizes.View
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

    private MazeViews BuildMazeViews() => new()
    {
        CtrlWeights = _ctrlWeights.View,
        CtrlBiases = _ctrlBiases.View,
        CtrlLayerSizes = _ctrlLayerSizes.View,
        CtrlObs = _ctrlObs!.View,
        CtrlAct = _ctrlAct!.View,
        GoalX = _goalX!.View,
        GoalY = _goalY!.View,
        PrevDist = _prevDist!.View,
        RewardAccum = _rewardAccum!.View
    };

    private void RunSimulationLoop(int worldCount,
        MegaKernelConfig megaConfig, DenseMazeConfig mazeConfig,
        PhysicsViews physicsViews, DenseNNViews mazeNNViews, EpisodeViews episodeViews, MazeViews mazeViews)
    {
        const int batchSize = 10;
        for (int step = 0; step < MaxSteps; step++)
        {
            _fusedStepKernel(
                worldCount, physicsViews, mazeNNViews, episodeViews,
                megaConfig, mazeConfig, mazeViews,
                _mazeObs!.View, _mazeAct!.View);

            if ((step + 1) % batchSize == 0)
            {
                try { _accelerator.Synchronize(); }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"GPU FATAL: Synchronize failed at maze step {step}/{MaxSteps}, {worldCount} worlds. " +
                        $"CUDA context poisoned. maze={_mazeTopology}, ctrl={_ctrlTopology}", ex);
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
        int tw = _mazeTopology.TotalWeights;
        int tb = _mazeTopology.TotalBiases;
        int tp = _mazeTopology.TotalParams;

        if (_mazeWeightsCPU == null || _mazeWeightsCPU.Length != totalPop * tw)
        {
            _mazeWeightsCPU = new float[totalPop * tw];
            _mazeBiasesCPU = new float[totalPop * tb];
        }
        for (int i = 0; i < totalPop; i++)
        {
            Array.Copy(paramVectors, i * tp, _mazeWeightsCPU, i * tw, tw);
            Array.Copy(paramVectors, i * tp + tw, _mazeBiasesCPU!, i * tb, tb);
        }
        _mazeWeights!.CopyFromCPU(_mazeWeightsCPU);
        _mazeBiases!.CopyFromCPU(_mazeBiasesCPU!);
    }

    private void ResetEpisodeState(int worldCount)
    {
        _resetKernel(
            worldCount,
            _stepCounters!.View, _isTerminal!.View, _hasLanded!.View,
            _currentThrottle!.View, _currentGimbal!.View, _fitnessValues!.View);
        _waggleAccum!.MemSetToZero();
        _settledSteps!.MemSetToZero();
        _rewardAccum!.MemSetToZero();
        // _goalX/_goalY/_prevDist are set in UploadMaze (per-world).
        _worldState!.ClearContactCounts();
        _worldState.ClearContactCache();
        _accelerator.Synchronize();
    }

    private void EnsureBuffers(int worldCount)
    {
        if (_lastWorldCount == worldCount && _worldState != null) return;

        _worldState?.Dispose();
        _mazeObs?.Dispose(); _mazeAct?.Dispose();
        _ctrlObs?.Dispose(); _ctrlAct?.Dispose();
        _currentThrottle?.Dispose(); _currentGimbal?.Dispose();
        _isTerminal?.Dispose(); _hasLanded?.Dispose();
        _stepCounters?.Dispose(); _fitnessValues?.Dispose();
        _waggleAccum?.Dispose(); _settledSteps?.Dispose();
        _goalX?.Dispose(); _goalY?.Dispose();
        _prevDist?.Dispose(); _rewardAccum?.Dispose();
        _mazeWeights?.Dispose(); _mazeBiases?.Dispose();

        // Reserve room for the largest obstacle count we might place (so the shared view never resizes mid-run).
        int reserveColliders = Math.Max(1, NumObstacles);
        var worldConfig = new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = reserveColliders,
            TargetsPerWorld = 0,
            MaxContactsPerWorld = 24 + reserveColliders * 4
        };
        _worldState = new GPUBatchedWorldState(_accelerator, worldConfig);

        _mazeObs = _accelerator.Allocate1D<float>(worldCount * _mazeTopology.InputSize);
        _mazeAct = _accelerator.Allocate1D<float>(worldCount * _mazeTopology.OutputSize);
        _ctrlObs = _accelerator.Allocate1D<float>(worldCount * 9);
        _ctrlAct = _accelerator.Allocate1D<float>(worldCount * 2);
        _currentThrottle = _accelerator.Allocate1D<float>(worldCount);
        _currentGimbal = _accelerator.Allocate1D<float>(worldCount);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _hasLanded = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        _waggleAccum = _accelerator.Allocate1D<float>(worldCount);
        _settledSteps = _accelerator.Allocate1D<int>(worldCount);
        _goalX = _accelerator.Allocate1D<float>(worldCount);
        _goalY = _accelerator.Allocate1D<float>(worldCount);
        _prevDist = _accelerator.Allocate1D<float>(worldCount);
        _rewardAccum = _accelerator.Allocate1D<float>(worldCount);
        _mazeWeights = _accelerator.Allocate1D<float>(worldCount * _mazeTopology.TotalWeights);
        _mazeBiases = _accelerator.Allocate1D<float>(worldCount * _mazeTopology.TotalBiases);

        _terminalCache = new byte[worldCount];
        _landedCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _lastWorldCount = worldCount;
    }

    /// <summary>Upload obstacles and build the (shared) rocket at the maze's start, upright and at rest.</summary>
    private void UploadMaze(int worldCount, Maze maze)
    {
        // Obstacles → shared colliders (pad to the reserved size with a far-away dummy).
        var cfg = _worldState!.Config;
        var colliders = new GPUOBBCollider[cfg.SharedColliderCount];
        for (int i = 0; i < colliders.Length; i++)
        {
            colliders[i] = i < maze.obstacles.Length
                ? maze.obstacles[i]
                : new GPUOBBCollider { CX = 0f, CY = -1000f, UX = 1f, UY = 0f, HalfExtentX = 1f, HalfExtentY = 1f };
        }
        _worldState.UploadSharedColliders(colliders);

        CreateAndUploadRocketTemplate(worldCount, maze.sx, maze.sy);

        // Per-world goal (single maze → replicated) + initial distance for the progress reward.
        float d0 = Dist(maze.sx, maze.sy, maze.gx, maze.gy);
        var gx = new float[worldCount];
        var gy = new float[worldCount];
        var pd = new float[worldCount];
        for (int i = 0; i < worldCount; i++) { gx[i] = maze.gx; gy[i] = maze.gy; pd[i] = d0; }
        _goalX!.CopyFromCPU(gx);
        _goalY!.CopyFromCPU(gy);
        _prevDist!.CopyFromCPU(pd);
    }

    /// <summary>Build the rocket at one spawn for all worlds (training: shared maze).</summary>
    private void CreateAndUploadRocketTemplate(int worldCount, float spawnX, float spawnY)
    {
        var sx = new float[worldCount];
        var sy = new float[worldCount];
        for (int i = 0; i < worldCount; i++) { sx[i] = spawnX; sy[i] = spawnY; }
        CreateAndUploadRocketTemplate(worldCount, sx, sy);
    }

    /// <summary>Build the 3-body rocket (fuselage + 2 legs) per-world at (spawnX[w], spawnY[w]), upright, at rest.</summary>
    private void CreateAndUploadRocketTemplate(int worldCount, float[] spawnX, float[] spawnY)
    {
        const float bodyHeight = 1.5f, bodyRadius = 0.2f, bodyMass = 8f;
        const float bodyHalfLength = bodyHeight * 0.5f;
        float bodyInertia = bodyMass * (bodyRadius * bodyRadius * 0.25f + bodyHeight * bodyHeight / 12f);

        const float legLength = 1.0f, legRadius = 0.1f, legMass = 1.5f;
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
            bodyGeoms[i] = new GPURigidBodyGeom { LocalX = -bodyHalfLength + t * bodyHeight, LocalY = 0f, Radius = bodyRadius, BodyIndex = 0 };
        }
        var legGeoms = new GPURigidBodyGeom[legGeomCount];
        for (int i = 0; i < legGeomCount; i++)
        {
            float t = (float)i / (legGeomCount - 1);
            legGeoms[i] = new GPURigidBodyGeom { LocalX = -legHalfLength + t * legLength, LocalY = 0f, Radius = legRadius, BodyIndex = 0 };
        }

        var config = _worldState!.Config;
        var allBodies = new GPURigidBody[config.TotalRigidBodies];
        var allGeoms = new GPURigidBodyGeom[config.TotalGeoms];
        var allJoints = new GPURevoluteJoint[config.TotalJoints];

        float bodyAngleFinal = bodyAngle;

        for (int w = 0; w < worldCount; w++)
        {
            float sX = spawnX[w], sY = spawnY[w];
            float bodyX = sX, bodyY = sY + bodyHalfLength;
            allBodies[config.GetRigidBodyIndex(w, 0)] = new GPURigidBody
            {
                X = bodyX, Y = bodyY, Angle = bodyAngleFinal,
                VelX = 0f, VelY = 0f, AngularVel = 0f,
                PrevX = bodyX, PrevY = bodyY, PrevAngle = bodyAngleFinal,
                InvMass = 1f / bodyMass, InvInertia = 1f / bodyInertia,
                GeomStartIndex = 0, GeomCount = bodyGeomCount
            };

            float leftLegX = sX + leftOffX, leftLegY = sY + leftOffY;
            allBodies[config.GetRigidBodyIndex(w, 1)] = new GPURigidBody
            {
                X = leftLegX, Y = leftLegY, Angle = leftLegAngle,
                VelX = 0f, VelY = 0f, AngularVel = 0f,
                PrevX = leftLegX, PrevY = leftLegY, PrevAngle = leftLegAngle,
                InvMass = 1f / legMass, InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount, GeomCount = legGeomCount
            };

            float rightLegX = sX + rightOffX, rightLegY = sY + rightOffY;
            allBodies[config.GetRigidBodyIndex(w, 2)] = new GPURigidBody
            {
                X = rightLegX, Y = rightLegY, Angle = rightLegAngle,
                VelX = 0f, VelY = 0f, AngularVel = 0f,
                PrevX = rightLegX, PrevY = rightLegY, PrevAngle = rightLegAngle,
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

    // === Replay (visualization): ONE maze policy across `count` worlds, each its own goal+spawn ===
    private GPURigidBody[]? _replayBodies;
    private GPURigidBodyGeom[]? _replayGeoms;
    private byte[]? _replayTerminal;
    private byte[]? _replayLanded;
    private float[]? _replayThrottle;
    private float[]? _replayGimbal;
    private float[]? _replayMazeAct;
    private Maze[]? _replayMazes;
    private int _replayCount;

    public int ReplayCount => _replayCount;
    public Maze[] ReplayMazes => _replayMazes ?? Array.Empty<Maze>();

    public void PrepareReplay(float[] mazePolicyParams, int count, int baseSeed)
    {
        _replayCount = count;
        EnsureBuffers(count);

        int pc = _mazeTopology.TotalParams;
        if (mazePolicyParams.Length != pc)
            throw new ArgumentException($"Maze policy params {mazePolicyParams.Length} != topology {pc} ({_mazeTopology}).");
        var flat = new float[count * pc];
        for (int w = 0; w < count; w++) Array.Copy(mazePolicyParams, 0, flat, w * pc, pc);
        SplitAndUpload(flat, count);

        // The GPU shares ONE obstacle set across all worlds, so the replay grid shows the SAME
        // obstacle field in every cell, with a different start+goal per cell (each sampled clear
        // of the field). Open space (NumObstacles=0) → empty field → just varied start/goal.
        var field = GenerateObstacleField(baseSeed, NumObstacles);
        var rng = new Random(baseSeed * 7 + 13);
        _replayMazes = new Maze[count];
        for (int w = 0; w < count; w++)
        {
            float sxw = 0f, syw = 9f, gxw = 0f, gyw = 6f;
            for (int tries = 0; tries < 200; tries++)
            {
                sxw = (float)(rng.NextDouble() * 4 - 2);
                syw = (float)(8 + rng.NextDouble() * 2);
                if (ClearOfObstacles(sxw, syw, field, 1.5f)) break;
            }
            for (int tries = 0; tries < 200; tries++)
            {
                gxw = (float)(rng.NextDouble() * 20 - 10);
                gyw = (float)(2 + rng.NextDouble() * 8);
                if (ClearOfObstacles(gxw, gyw, field, 1.5f) && Dist(sxw, syw, gxw, gyw) > 4f) break;
            }
            _replayMazes[w] = new Maze { sx = sxw, sy = syw, gx = gxw, gy = gyw, obstacles = field };
        }

        var gx = new float[count]; var gy = new float[count]; var pd = new float[count];
        var sx = new float[count]; var sy = new float[count];
        for (int w = 0; w < count; w++)
        {
            var mz = _replayMazes[w];
            gx[w] = mz.gx; gy[w] = mz.gy; sx[w] = mz.sx; sy[w] = mz.sy;
            pd[w] = Dist(mz.sx, mz.sy, mz.gx, mz.gy);
        }

        // Obstacles are SHARED across worlds — replay uses world 0's set (open space = none).
        var cfg = _worldState!.Config;
        var colliders = new GPUOBBCollider[cfg.SharedColliderCount];
        var m0obs = _replayMazes[0].obstacles;
        for (int i = 0; i < colliders.Length; i++)
            colliders[i] = i < m0obs.Length ? m0obs[i]
                : new GPUOBBCollider { CX = 0f, CY = -1000f, UX = 1f, UY = 0f, HalfExtentX = 1f, HalfExtentY = 1f };
        _worldState.UploadSharedColliders(colliders);

        CreateAndUploadRocketTemplate(count, sx, sy);
        _goalX!.CopyFromCPU(gx);
        _goalY!.CopyFromCPU(gy);
        _prevDist!.CopyFromCPU(pd);
        ResetEpisodeState(count);

        _replayBodies = new GPURigidBody[cfg.TotalRigidBodies];
        _replayGeoms = new GPURigidBodyGeom[cfg.TotalGeoms];
        _replayTerminal = new byte[count];
        _replayLanded = new byte[count];
        _replayThrottle = new float[count];
        _replayGimbal = new float[count];
        _replayMazeAct = new float[count * _mazeTopology.OutputSize];
    }

    public void StepReplay()
    {
        var megaConfig = BuildMegaConfig(_replayMazes![0].obstacles.Length);
        _fusedStepKernel(_replayCount, BuildPhysicsViews(), BuildMazeNNViews(), BuildEpisodeViews(),
            megaConfig, BuildMazeConfig(), BuildMazeViews(), _mazeObs!.View, _mazeAct!.View);
        _accelerator.Synchronize();
    }

    /// <summary>Read replay state. reachedGoal is the HasLanded flag; cmdVx/cmdVy are the scaled velocity command.</summary>
    public void ReadReplayState(
        out GPURigidBody[] bodies, out GPURigidBodyGeom[] geoms,
        out byte[] terminal, out byte[] reachedGoal, out float[] throttle, out float[] gimbal,
        out float[] cmdVx, out float[] cmdVy)
    {
        _worldState!.RigidBodies.CopyToCPU(_replayBodies!);
        _worldState.Geoms.CopyToCPU(_replayGeoms!);
        _isTerminal!.CopyToCPU(_replayTerminal!);
        _hasLanded!.CopyToCPU(_replayLanded!);
        _currentThrottle!.CopyToCPU(_replayThrottle!);
        _currentGimbal!.CopyToCPU(_replayGimbal!);
        _mazeAct!.CopyToCPU(_replayMazeAct!);

        int oc = _mazeTopology.OutputSize;
        var cvx = new float[_replayCount];
        var cvy = new float[_replayCount];
        for (int w = 0; w < _replayCount; w++)
        {
            cvx[w] = Math.Clamp(_replayMazeAct![w * oc + 0], -1f, 1f) * CmdSpeedMax;
            cvy[w] = Math.Clamp(_replayMazeAct![w * oc + 1], -1f, 1f) * CmdSpeedMax;
        }
        bodies = _replayBodies!; geoms = _replayGeoms!;
        terminal = _replayTerminal!; reachedGoal = _replayLanded!;
        throttle = _replayThrottle!; gimbal = _replayGimbal!;
        cmdVx = cvx; cmdVy = cvy;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _worldState?.Dispose();
        _mazeObs?.Dispose(); _mazeAct?.Dispose();
        _ctrlObs?.Dispose(); _ctrlAct?.Dispose();
        _currentThrottle?.Dispose(); _currentGimbal?.Dispose();
        _isTerminal?.Dispose(); _hasLanded?.Dispose();
        _stepCounters?.Dispose(); _fitnessValues?.Dispose();
        _waggleAccum?.Dispose(); _settledSteps?.Dispose();
        _goalX?.Dispose(); _goalY?.Dispose();
        _prevDist?.Dispose(); _rewardAccum?.Dispose();
        _mazeWeights?.Dispose(); _mazeBiases?.Dispose();
        _mazeLayerSizes.Dispose();
        _ctrlWeights.Dispose(); _ctrlBiases.Dispose(); _ctrlLayerSizes.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }
}
