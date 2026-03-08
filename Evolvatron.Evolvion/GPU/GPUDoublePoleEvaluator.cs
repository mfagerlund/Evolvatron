using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Evolvion.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated fitness evaluator for double pole balancing.
/// Fused mega-kernel: observe -> NN -> action -> RK4 physics -> terminal check.
/// No rigid body physics — pure math, extremely lightweight per thread.
/// Auto-scales population to GPU hardware (NumMultiprocessors * WarpSize * occupancy).
/// </summary>
public class GPUDoublePoleEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly int _maxIndividuals;
    private bool _disposed;

    // Neural network state (reused from rocket evaluator pattern)
    private GPUEvolvionState? _neuralState;

    // Double pole state buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _state; // 6 floats per world
    private MemoryBuffer1D<float, Stride1D.Dense>? _observations;
    private MemoryBuffer1D<float, Stride1D.Dense>? _actions;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;
    private MemoryBuffer1D<float, Stride1D.Dense>? _jiggleBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _contextBuffer;

    // Kernels
    private readonly Action<Index1D, NNViews, DoublePoleEpisodeViews,
        DoublePoleConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>> _stepKernel;

    // Host caches
    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private int[]? _stepCache;
    private int _lastWorldCount;
    private int _lastInputSize;
    private int _lastOutputSize;
    private int _lastContextSize;

    // Environment config
    public int MaxSteps { get; set; } = 100_000;
    public int TicksPerLaunch { get; set; } = 500;
    public float TrackLength { get; set; } = 4.8f;
    public float PoleAngleThresholdDegrees { get; set; } = 36f;
    public bool IncludeVelocity { get; set; } = false; // false = hard mode

    // Recurrence config (Sparse Study: Elman ctx=2 optimal for DPNV)
    public int ContextSize { get; set; } = 2;
    public bool IsJordan { get; set; } = false; // true = action feedback, false = Elman

    // Fitness config (Sparse Study: Gruau provides ~10-15% speed boost)
    public bool UseGruauFitness { get; set; } = true;

    // Initial state (matching Colonel: pole1 at 4 degrees)
    private const float InitialPole1Angle = 3.14159265358979f / 45f; // 4 degrees

    // Multi-position evaluation
    private float[][]? _startingPositions; // cached starting positions (N x 6)

    /// <summary>
    /// Computes optimal population size based on GPU hardware.
    /// Targets 4 warps per SM for good occupancy.
    /// </summary>
    public int OptimalPopulationSize
    {
        get
        {
            int sms = _accelerator.NumMultiprocessors;
            int warpSize = _accelerator.WarpSize;
            int warpsPerSM = 4; // occupancy target
            return sms * warpsPerSM * warpSize;
        }
    }

    public string DeviceName => _accelerator.Name;
    public int NumMultiprocessors => _accelerator.NumMultiprocessors;
    public int WarpSize => _accelerator.WarpSize;

    public GPUDoublePoleEvaluator(int maxIndividuals = 0)
    {
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        // Auto-scale if not specified
        if (maxIndividuals <= 0)
            maxIndividuals = OptimalPopulationSize;
        _maxIndividuals = maxIndividuals;

        Console.WriteLine($"GPUDoublePoleEvaluator on: {_accelerator.Name} " +
                          $"({_accelerator.NumMultiprocessors} SMs, warp={_accelerator.WarpSize}, " +
                          $"optimal pop={OptimalPopulationSize})");

        _stepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, NNViews, DoublePoleEpisodeViews,
            DoublePoleConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            DoublePoleStepKernel.StepKernel);
    }

    /// <summary>
    /// Evaluates all individuals. Returns (fitness array, count of solved individuals).
    /// Solved = survived MaxSteps.
    /// </summary>
    public (float[] fitness, int solved) EvaluatePopulation(
        SpeciesSpec spec,
        List<Individual> individuals,
        int seed)
    {
        int worldCount = individuals.Count;
        if (worldCount == 0)
            return (Array.Empty<float>(), 0);

        int inputSize = spec.RowCounts[0];
        int outputSize = spec.RowCounts[^1];

        InitializeNeuralState(spec, individuals);
        EnsureBuffers(worldCount, inputSize, outputSize);
        UploadInitialState(worldCount);

        var config = BuildConfig(spec, inputSize, outputSize);
        var nnViews = BuildNNViews();
        var episodeViews = BuildEpisodeViews();

        RunEvaluationLoop(worldCount, config, nnViews, episodeViews);

        // Read back results
        _accelerator.Synchronize();
        _fitnessValues!.CopyToCPU(_fitnessCache!);
        _stepCounters!.CopyToCPU(_stepCache!);

        int solved = 0;
        var result = new float[worldCount];
        for (int i = 0; i < worldCount; i++)
        {
            result[i] = _fitnessCache![i];
            if (_stepCache![i] >= MaxSteps)
                solved++;
        }

        return (result, solved);
    }

    /// <summary>
    /// Set starting positions for multi-position evaluation.
    /// Each position is a float[6]: [cartPos, cartVel, pole1Angle, pole1AngVel, pole2Angle, pole2AngVel].
    /// </summary>
    public void SetStartingPositions(float[][] positions)
    {
        _startingPositions = positions;
    }

    /// <summary>
    /// Evaluates all individuals across ALL starting positions.
    /// Runs one full evaluation per position, aggregates fitness as mean across positions.
    /// Solved = individual survived MaxSteps on EVERY position.
    /// </summary>
    public (float[] fitness, int allSolved) EvaluateAllPositions(
        SpeciesSpec spec,
        List<Individual> individuals,
        int seed)
    {
        if (_startingPositions == null || _startingPositions.Length == 0)
            throw new InvalidOperationException("Call SetStartingPositions before EvaluateAllPositions");

        int worldCount = individuals.Count;
        if (worldCount == 0)
            return (Array.Empty<float>(), 0);

        int numPositions = _startingPositions.Length;
        int inputSize = spec.RowCounts[0];
        int outputSize = spec.RowCounts[^1];

        // Upload NN weights once
        InitializeNeuralState(spec, individuals);
        EnsureBuffers(worldCount, inputSize, outputSize);

        var config = BuildConfig(spec, inputSize, outputSize);
        var nnViews = BuildNNViews();
        var episodeViews = BuildEpisodeViews();

        // Accumulate fitness across positions
        var totalFitness = new float[worldCount];
        var solvedCounts = new int[worldCount]; // how many positions each individual solved

        for (int posIdx = 0; posIdx < numPositions; posIdx++)
        {
            // Reset episode state and upload this position's initial state
            UploadInitialState(worldCount, _startingPositions[posIdx]);

            // Run evaluation loop
            RunEvaluationLoop(worldCount, config, nnViews, episodeViews);

            // Read back results
            _accelerator.Synchronize();
            _fitnessValues!.CopyToCPU(_fitnessCache!);
            _stepCounters!.CopyToCPU(_stepCache!);

            for (int i = 0; i < worldCount; i++)
            {
                totalFitness[i] += _fitnessCache![i];
                if (_stepCache![i] >= MaxSteps)
                    solvedCounts[i]++;
            }
        }

        // Aggregate fitness: solvedCount * MaxSteps + mean_steps
        // This ensures solving more positions always dominates over surviving longer on fewer.
        // Range: 0 (all fail instantly) to numPositions * MaxSteps + MaxSteps (all solved).
        var result = new float[worldCount];
        int allSolved = 0;
        for (int i = 0; i < worldCount; i++)
        {
            float meanSteps = totalFitness[i] / numPositions;
            result[i] = solvedCounts[i] * MaxSteps + meanSteps;
            if (solvedCounts[i] == numPositions)
                allSolved++;
        }

        return (result, allSolved);
    }

    private DoublePoleConfig BuildConfig(SpeciesSpec spec, int inputSize, int outputSize)
    {
        float angleThreshold = PoleAngleThresholdDegrees * 3.14159265358979f / 180f;
        return new DoublePoleConfig
        {
            MaxSteps = MaxSteps,
            TicksPerLaunch = TicksPerLaunch,
            TrackLengthHalf = TrackLength / 2f,
            PoleAngleThreshold = angleThreshold,
            IncludeVelocity = IncludeVelocity ? 1 : 0,
            TotalNodes = spec.TotalNodes,
            TotalEdges = spec.TotalEdges,
            NumRows = spec.RowPlans.Length,
            InputSize = inputSize,
            OutputSize = outputSize,
            ContextSize = ContextSize,
            IsJordan = IsJordan ? 1 : 0,
            GruauEnabled = UseGruauFitness ? 1 : 0
        };
    }

    private NNViews BuildNNViews()
    {
        return new NNViews
        {
            NodeValues = _neuralState!.NodeValues.View,
            Edges = _neuralState.Edges.View,
            Weights = _neuralState.Individuals.Weights.View,
            Biases = _neuralState.Individuals.Biases.View,
            Activations = _neuralState.Individuals.Activations.View,
            NodeParams = _neuralState.Individuals.NodeParams.View,
            RowPlans = _neuralState.RowPlans.View
        };
    }

    private DoublePoleEpisodeViews BuildEpisodeViews()
    {
        return new DoublePoleEpisodeViews
        {
            IsTerminal = _isTerminal!.View,
            StepCounters = _stepCounters!.View,
            FitnessValues = _fitnessValues!.View,
            JiggleBuffer = _jiggleBuffer!.View,
            ContextBuffer = _contextBuffer!.View
        };
    }

    private void RunEvaluationLoop(int worldCount, DoublePoleConfig config,
        NNViews nnViews, DoublePoleEpisodeViews episodeViews)
    {
        int totalLaunches = (MaxSteps + TicksPerLaunch - 1) / TicksPerLaunch;
        int earlyExitInterval = Math.Max(1, 2000 / TicksPerLaunch);

        for (int launch = 0; launch < totalLaunches; launch++)
        {
            if (launch > 0 && launch % earlyExitInterval == 0)
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

            _stepKernel(
                worldCount,
                nnViews,
                episodeViews,
                config,
                _observations!.View,
                _actions!.View,
                _state!.View);
        }
    }

    private void UploadInitialState(int worldCount, float[] position)
    {
        var initialState = new float[worldCount * 6];
        for (int w = 0; w < worldCount; w++)
        {
            int b = w * 6;
            for (int j = 0; j < 6; j++)
                initialState[b + j] = position[j];
        }
        _state!.CopyFromCPU(initialState);

        _isTerminal!.MemSetToZero();
        _stepCounters!.MemSetToZero();
        _fitnessValues!.MemSetToZero();
        _jiggleBuffer!.MemSetToZero();
        _contextBuffer!.MemSetToZero();
    }

    private void UploadInitialState(int worldCount)
    {
        // All worlds start with same initial condition: pole1 at 4 degrees
        var initialState = new float[worldCount * 6];
        for (int w = 0; w < worldCount; w++)
        {
            int b = w * 6;
            initialState[b]     = 0f;                  // cart position
            initialState[b + 1] = 0f;                  // cart velocity
            initialState[b + 2] = InitialPole1Angle;   // pole1 angle (4 degrees)
            initialState[b + 3] = 0f;                  // pole1 angular velocity
            initialState[b + 4] = 0f;                  // pole2 angle
            initialState[b + 5] = 0f;                  // pole2 angular velocity
        }
        _state!.CopyFromCPU(initialState);

        // Reset episode state (no Synchronize needed — same-stream ordering guarantees)
        _isTerminal!.MemSetToZero();
        _stepCounters!.MemSetToZero();
        _fitnessValues!.MemSetToZero();
        _jiggleBuffer!.MemSetToZero();
        _contextBuffer!.MemSetToZero();
    }

    private void EnsureBuffers(int worldCount, int inputSize, int outputSize)
    {
        if (_lastWorldCount == worldCount &&
            _lastInputSize == inputSize &&
            _lastOutputSize == outputSize &&
            _lastContextSize == ContextSize &&
            _state != null)
            return;

        _state?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _isTerminal?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _jiggleBuffer?.Dispose();
        _contextBuffer?.Dispose();

        _state = _accelerator.Allocate1D<float>(worldCount * 6);
        _observations = _accelerator.Allocate1D<float>(worldCount * inputSize);
        _actions = _accelerator.Allocate1D<float>(worldCount * outputSize);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        _jiggleBuffer = _accelerator.Allocate1D<float>(worldCount * 100);
        _contextBuffer = _accelerator.Allocate1D<float>(Math.Max(1, worldCount * ContextSize));

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _stepCache = new int[worldCount];
        _lastWorldCount = worldCount;
        _lastInputSize = inputSize;
        _lastOutputSize = outputSize;
        _lastContextSize = ContextSize;
    }

    private void InitializeNeuralState(SpeciesSpec spec, List<Individual> individuals)
    {
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

        _state?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _isTerminal?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _jiggleBuffer?.Dispose();
        _contextBuffer?.Dispose();
        _neuralState?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
