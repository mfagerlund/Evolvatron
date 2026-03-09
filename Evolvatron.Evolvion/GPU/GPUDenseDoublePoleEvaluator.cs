using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated fitness evaluator for double pole balancing using dense NN.
/// Takes flat parameter vectors (weights + biases concatenated), splits and uploads to GPU.
/// Fused mega-kernel: observe → dense NN → action → 2x RK4 → terminal check.
/// </summary>
public class GPUDenseDoublePoleEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly DenseTopology _topology;
    private bool _disposed;

    // Dense NN buffers (weights and biases only — no edges, row plans, activations, etc.)
    private MemoryBuffer1D<float, Stride1D.Dense>? _weights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _biases;
    private MemoryBuffer1D<int, Stride1D.Dense>? _layerSizes;

    // Episode state buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _state;
    private MemoryBuffer1D<float, Stride1D.Dense>? _observations;
    private MemoryBuffer1D<float, Stride1D.Dense>? _actions;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;
    private MemoryBuffer1D<float, Stride1D.Dense>? _jiggleBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _contextBuffer;

    // Kernel
    private readonly Action<Index1D, DenseNNViews, DoublePoleEpisodeViews,
        DenseDoublePoleConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>> _stepKernel;

    // Host caches
    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private int[]? _stepCache;
    private float[]? _weightsCPU;
    private float[]? _biasesCPU;
    private int _lastWorldCount;

    // Environment config
    public int MaxSteps { get; set; } = 100_000;
    public int TicksPerLaunch { get; set; } = 500;
    public float TrackLength { get; set; } = 4.8f;
    public float PoleAngleThresholdDegrees { get; set; } = 36f;
    public bool IncludeVelocity { get; set; } = false;

    // Recurrence (Elman ctx=2 is the DPNV default)
    public int ContextSize { get; set; } = 2;
    public bool IsJordan { get; set; } = false;

    // Fitness
    public bool UseGruauFitness { get; set; } = true;

    // Initial state
    private const float InitialPole1Angle = 3.14159265358979f / 45f; // 4 degrees

    // Multi-position evaluation
    private float[][]? _startingPositions;

    public int OptimalPopulationSize
    {
        get
        {
            int sms = _accelerator.NumMultiprocessors;
            int warpSize = _accelerator.WarpSize;
            return sms * 4 * warpSize;
        }
    }

    public string DeviceName => _accelerator.Name;
    public int NumMultiprocessors => _accelerator.NumMultiprocessors;

    public GPUDenseDoublePoleEvaluator(DenseTopology topology)
    {
        _topology = topology;
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPUDenseDoublePoleEvaluator on: {_accelerator.Name} " +
                          $"({_accelerator.NumMultiprocessors} SMs, optimal pop={OptimalPopulationSize}), " +
                          $"topology={topology}");

        _stepKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, DenseNNViews, DoublePoleEpisodeViews,
            DenseDoublePoleConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            DenseDoublePoleStepKernel.StepKernel);

        // Upload layer sizes once (shared across all individuals and generations)
        _layerSizes = _accelerator.Allocate1D<int>(topology.NumLayers);
        _layerSizes.CopyFromCPU(topology.LayerSizes);
    }

    /// <summary>
    /// Evaluate population from flat parameter vectors.
    /// paramVectors layout: [ind0_weights | ind0_biases | ind1_weights | ind1_biases | ...]
    /// Each individual's params = totalWeights + totalBiases floats.
    /// </summary>
    public (float[] fitness, int solved) EvaluatePopulation(float[] paramVectors, int totalPop)
    {
        if (totalPop == 0)
            return (Array.Empty<float>(), 0);

        EnsureBuffers(totalPop);
        SplitAndUpload(paramVectors, totalPop);
        UploadInitialState(totalPop);

        var config = BuildConfig();
        var nnViews = BuildNNViews();
        var episodeViews = BuildEpisodeViews();

        RunEvaluationLoop(totalPop, config, nnViews, episodeViews);

        _accelerator.Synchronize();
        _fitnessValues!.CopyToCPU(_fitnessCache!);
        _stepCounters!.CopyToCPU(_stepCache!);

        int solved = 0;
        var result = new float[totalPop];
        for (int i = 0; i < totalPop; i++)
        {
            result[i] = _fitnessCache![i];
            if (_stepCache![i] >= MaxSteps)
                solved++;
        }

        return (result, solved);
    }

    public void SetStartingPositions(float[][] positions)
    {
        _startingPositions = positions;
    }

    /// <summary>
    /// Evaluate all individuals across ALL starting positions (625-grid benchmark).
    /// </summary>
    public (float[] fitness, int allSolved) EvaluateAllPositions(float[] paramVectors, int totalPop)
    {
        if (_startingPositions == null || _startingPositions.Length == 0)
            throw new InvalidOperationException("Call SetStartingPositions before EvaluateAllPositions");

        if (totalPop == 0)
            return (Array.Empty<float>(), 0);

        int numPositions = _startingPositions.Length;

        EnsureBuffers(totalPop);
        SplitAndUpload(paramVectors, totalPop);

        var config = BuildConfig();
        var nnViews = BuildNNViews();
        var episodeViews = BuildEpisodeViews();

        var totalFitness = new float[totalPop];
        var solvedCounts = new int[totalPop];

        for (int posIdx = 0; posIdx < numPositions; posIdx++)
        {
            UploadInitialState(totalPop, _startingPositions[posIdx]);
            RunEvaluationLoop(totalPop, config, nnViews, episodeViews);

            _accelerator.Synchronize();
            _fitnessValues!.CopyToCPU(_fitnessCache!);
            _stepCounters!.CopyToCPU(_stepCache!);

            for (int i = 0; i < totalPop; i++)
            {
                totalFitness[i] += _fitnessCache![i];
                if (_stepCache![i] >= MaxSteps)
                    solvedCounts[i]++;
            }
        }

        var result = new float[totalPop];
        int allSolved = 0;
        for (int i = 0; i < totalPop; i++)
        {
            float meanSteps = totalFitness[i] / numPositions;
            result[i] = solvedCounts[i] * MaxSteps + meanSteps;
            if (solvedCounts[i] == numPositions)
                allSolved++;
        }

        return (result, allSolved);
    }

    /// <summary>
    /// Evaluate a single parameter vector (e.g., the μ of the best island).
    /// Useful for champion testing on the 625-grid.
    /// </summary>
    public int EvaluateChampionOnGrid(float[] mu)
    {
        if (_startingPositions == null || _startingPositions.Length == 0)
            throw new InvalidOperationException("Call SetStartingPositions first");

        // Create a "population" of 1 individual
        var (fitness, allSolved) = EvaluateAllPositions(mu, 1);
        return allSolved; // 0 or 1 (but for grid scoring, use solvedCounts)
    }

    /// <summary>
    /// Evaluate champion on grid and return number of positions survived.
    /// </summary>
    public int EvaluateChampionGridScore(float[] mu)
    {
        if (_startingPositions == null || _startingPositions.Length == 0)
            throw new InvalidOperationException("Call SetStartingPositions first");

        int numPositions = _startingPositions.Length;

        EnsureBuffers(1);
        SplitAndUpload(mu, 1);

        var config = BuildConfig();
        var nnViews = BuildNNViews();
        var episodeViews = BuildEpisodeViews();

        int gridScore = 0;

        for (int posIdx = 0; posIdx < numPositions; posIdx++)
        {
            UploadInitialState(1, _startingPositions[posIdx]);
            RunEvaluationLoop(1, config, nnViews, episodeViews);

            _accelerator.Synchronize();
            _stepCounters!.CopyToCPU(_stepCache!);

            if (_stepCache![0] >= MaxSteps)
                gridScore++;
        }

        return gridScore;
    }

    private DenseDoublePoleConfig BuildConfig()
    {
        float angleThreshold = PoleAngleThresholdDegrees * 3.14159265358979f / 180f;
        return new DenseDoublePoleConfig
        {
            MaxSteps = MaxSteps,
            TicksPerLaunch = TicksPerLaunch,
            TrackLengthHalf = TrackLength / 2f,
            PoleAngleThreshold = angleThreshold,
            IncludeVelocity = IncludeVelocity ? 1 : 0,
            NumLayers = _topology.NumLayers,
            TotalWeightsPerNet = _topology.TotalWeights,
            TotalBiasesPerNet = _topology.TotalBiases,
            InputSize = _topology.InputSize,
            OutputSize = _topology.OutputSize,
            ContextSize = ContextSize,
            IsJordan = IsJordan ? 1 : 0,
            GruauEnabled = UseGruauFitness ? 1 : 0
        };
    }

    private DenseNNViews BuildNNViews()
    {
        return new DenseNNViews
        {
            Weights = _weights!.View,
            Biases = _biases!.View,
            LayerSizes = _layerSizes!.View
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

    private void RunEvaluationLoop(int worldCount, DenseDoublePoleConfig config,
        DenseNNViews nnViews, DoublePoleEpisodeViews episodeViews)
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

    /// <summary>
    /// Split flat param vectors into separate weights and biases arrays, upload to GPU.
    /// Param layout per individual: [weights (totalWeights) | biases (totalBiases)].
    /// </summary>
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

    private void UploadInitialState(int worldCount)
    {
        var initialState = new float[worldCount * 6];
        for (int w = 0; w < worldCount; w++)
        {
            int b = w * 6;
            initialState[b + 2] = InitialPole1Angle;
        }
        _state!.CopyFromCPU(initialState);

        _isTerminal!.MemSetToZero();
        _stepCounters!.MemSetToZero();
        _fitnessValues!.MemSetToZero();
        _jiggleBuffer!.MemSetToZero();
        _contextBuffer!.MemSetToZero();
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

    private void EnsureBuffers(int worldCount)
    {
        if (_lastWorldCount == worldCount && _state != null)
            return;

        int inputSize = _topology.InputSize;
        int outputSize = _topology.OutputSize;

        _state?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _isTerminal?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _jiggleBuffer?.Dispose();
        _contextBuffer?.Dispose();
        _weights?.Dispose();
        _biases?.Dispose();

        _state = _accelerator.Allocate1D<float>(worldCount * 6);
        _observations = _accelerator.Allocate1D<float>(worldCount * inputSize);
        _actions = _accelerator.Allocate1D<float>(worldCount * outputSize);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);
        _jiggleBuffer = _accelerator.Allocate1D<float>(worldCount * 100);
        _contextBuffer = _accelerator.Allocate1D<float>(Math.Max(1, worldCount * ContextSize));
        _weights = _accelerator.Allocate1D<float>(worldCount * _topology.TotalWeights);
        _biases = _accelerator.Allocate1D<float>(worldCount * _topology.TotalBiases);

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _stepCache = new int[worldCount];
        _lastWorldCount = worldCount;
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
        _weights?.Dispose();
        _biases?.Dispose();
        _layerSizes?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
