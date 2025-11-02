using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated neural network evaluator for Evolvion.
/// Evaluates batches of individuals in parallel on the GPU.
/// </summary>
public class GPUEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private GPUEvolvionState? _gpuState;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPUEdge>, ArrayView<float>, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<GPURowPlan>, int, int, int, int> _evaluateRowKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPUEdge>, ArrayView<float>, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<GPURowPlan>, int, int, int, int, int> _evaluateRowForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> _setInputsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int> _setInputsForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<GPURowPlan>, int, int, int, int> _getOutputsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<GPURowPlan>, int, int, int, int> _getOutputsForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<byte>, GPULandscapeConfig, int, int> _initializeEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, GPULandscapeConfig, int> _computeObservationsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<byte>, GPULandscapeConfig, int> _stepEnvironmentKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, GPULandscapeConfig, int, int> _finalizeFitnessKernel;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int> _initializeXORKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, int> _getXORObservationsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int> _stepXORKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _finalizeXORFitnessKernel;

    private readonly Action<Index1D, ArrayView<float>, int, float> _initializeSpiralPointsKernel;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int> _initializeSpiralEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, int> _getSpiralObservationsKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<byte>, int, int> _stepSpiralKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> _finalizeSpiralFitnessKernel;

    private SpeciesSpec? _currentSpec;
    private bool _initialized = false;

    public GPUEvaluator()
    {
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        Console.WriteLine($"Available devices ({_context.Devices.Length}):");
        foreach (var device in _context.Devices)
        {
            Console.WriteLine($"  - {device.Name} (Type: {device.AcceleratorType}, Memory: {device.MemorySize / (1024 * 1024)} MB)");
        }

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        if (cudaDevice != null)
        {
            _accelerator = cudaDevice.CreateAccelerator(_context);
        }
        else
        {
            Console.WriteLine("WARNING: No CUDA device found, falling back to preferred device");
            _accelerator = _context.GetPreferredDevice(preferCPU: false)
                .CreateAccelerator(_context);
        }

        Console.WriteLine($"\nGPU Evaluator initialized on: {_accelerator.Name}");
        Console.WriteLine($"  Device type: {_accelerator.AcceleratorType}");
        Console.WriteLine($"  Memory: {_accelerator.MemorySize / (1024 * 1024)} MB");

        _evaluateRowKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<GPUEdge>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<byte>,
            ArrayView<float>,
            ArrayView<GPURowPlan>,
            int,
            int,
            int,
            int>(GPUEvolvionKernels.EvaluateRowKernel);

        _evaluateRowForEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<GPUEdge>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<byte>,
            ArrayView<float>,
            ArrayView<GPURowPlan>,
            int,
            int,
            int,
            int,
            int>(GPUEvolvionKernels.EvaluateRowForEpisodesKernel);

        _setInputsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int>(GPUEvolvionKernels.SetInputsKernel);

        _setInputsForEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int>(GPUEvolvionKernels.SetInputsForEpisodesKernel);

        _getOutputsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GPURowPlan>,
            int,
            int,
            int,
            int>(GPUEvolvionKernels.GetOutputsKernel);

        _getOutputsForEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GPURowPlan>,
            int,
            int,
            int,
            int>(GPUEvolvionKernels.GetOutputsForEpisodesKernel);

        _initializeEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<int>,
            ArrayView<byte>,
            GPULandscapeConfig,
            int,
            int>(GPUEvolvionKernels.InitializeEpisodesKernel);

        _computeObservationsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            GPULandscapeConfig,
            int>(GPUEvolvionKernels.ComputeObservationsKernel);

        _stepEnvironmentKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<int>,
            ArrayView<byte>,
            GPULandscapeConfig,
            int>(GPUEvolvionKernels.StepEnvironmentKernel);

        _finalizeFitnessKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            GPULandscapeConfig,
            int,
            int>(GPUEvolvionKernels.FinalizeFitnessKernel);

        _initializeXORKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<byte>,
            int>(GPUEvolvionKernels.InitializeXORKernel);

        _getXORObservationsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<int>,
            int>(GPUEvolvionKernels.GetXORObservationsKernel);

        _stepXORKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<byte>,
            int>(GPUEvolvionKernels.StepXORKernel);

        _finalizeXORFitnessKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int>(GPUEvolvionKernels.FinalizeXORFitnessKernel);

        _initializeSpiralPointsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            int,
            float>(GPUEvolvionKernels.InitializeSpiralPointsKernel);

        _initializeSpiralEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<byte>,
            int>(GPUEvolvionKernels.InitializeSpiralEpisodesKernel);

        _getSpiralObservationsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<int>,
            int,
            int>(GPUEvolvionKernels.GetSpiralObservationsKernel);

        _stepSpiralKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<byte>,
            int,
            int>(GPUEvolvionKernels.StepSpiralKernel);

        _finalizeSpiralFitnessKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int>(GPUEvolvionKernels.FinalizeSpiralFitnessKernel);
    }

    /// <summary>
    /// Initializes GPU state with topology and individuals.
    /// Call this once per species before evaluating.
    /// </summary>
    public void Initialize(SpeciesSpec spec, List<Individual> individuals)
    {
        _currentSpec = spec;

        if (!_initialized)
        {
            _gpuState = new GPUEvolvionState(
                _accelerator,
                maxIndividuals: 1000,
                maxNodes: spec.TotalNodes,
                maxEdges: spec.TotalEdges);
            _initialized = true;
        }

        _gpuState!.UploadTopology(spec);
        _gpuState.UploadIndividuals(individuals, spec);
    }

    /// <summary>
    /// Evaluates all individuals in parallel for given inputs.
    /// Returns output values for all individuals.
    /// </summary>
    public float[,] EvaluateBatch(float[,] inputs)
    {
        if (_gpuState == null || _currentSpec == null)
            throw new InvalidOperationException("Must call Initialize() before EvaluateBatch()");

        int individualCount = inputs.GetLength(0);
        int inputSize = inputs.GetLength(1);
        int outputSize = _currentSpec.RowCounts[^1];

        var flatInputs = new float[individualCount * inputSize];
        for (int i = 0; i < individualCount; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                flatInputs[i * inputSize + j] = inputs[i, j];
            }
        }

        using var inputBuffer = _accelerator.Allocate1D<float>(flatInputs.Length);
        inputBuffer.CopyFromCPU(flatInputs);

        _setInputsKernel(
            individualCount,
            _gpuState.NodeValues.View,
            inputBuffer.View,
            individualCount,
            _currentSpec.TotalNodes,
            inputSize);
        _accelerator.Synchronize();

        for (int rowIdx = 1; rowIdx < _currentSpec.RowPlans.Length; rowIdx++)
        {
            _evaluateRowKernel(
                individualCount,
                _gpuState.NodeValues.View,
                _gpuState.Edges.View,
                _gpuState.Individuals.Weights.View,
                _gpuState.Individuals.Biases.View,
                _gpuState.Individuals.Activations.View,
                _gpuState.Individuals.NodeParams.View,
                _gpuState.RowPlans.View,
                rowIdx,
                individualCount,
                _currentSpec.TotalNodes,
                _currentSpec.TotalEdges);
            _accelerator.Synchronize();
        }

        using var outputBuffer = _accelerator.Allocate1D<float>(individualCount * outputSize);
        _getOutputsKernel(
            individualCount,
            _gpuState.NodeValues.View,
            outputBuffer.View,
            _gpuState.RowPlans.View,
            _currentSpec.RowPlans.Length - 1,
            individualCount,
            _currentSpec.TotalNodes,
            outputSize);
        _accelerator.Synchronize();

        var flatOutputs = outputBuffer.GetAsArray1D();
        var outputs = new float[individualCount, outputSize];
        for (int i = 0; i < individualCount; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputs[i, j] = flatOutputs[i * outputSize + j];
            }
        }

        return outputs;
    }

    /// <summary>
    /// Evaluates a single individual (convenience method for testing).
    /// </summary>
    public ReadOnlySpan<float> Evaluate(Individual individual, ReadOnlySpan<float> inputs)
    {
        if (_currentSpec == null)
            throw new InvalidOperationException("Must call Initialize() before Evaluate()");

        var batchInputs = new float[1, inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            batchInputs[0, i] = inputs[i];
        }

        var batchOutputs = EvaluateBatch(batchInputs);

        var result = new float[batchOutputs.GetLength(1)];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = batchOutputs[0, i];
        }

        return result;
    }

    /// <summary>
    /// Evaluates all individuals with full environment episode loop.
    /// Returns fitness values for all individuals.
    /// </summary>
    public float[] EvaluateWithEnvironment(
        SpeciesSpec spec,
        List<Individual> individuals,
        GPULandscapeConfig landscapeConfig,
        int episodesPerIndividual,
        int seed)
    {
        if (_gpuState == null)
            throw new InvalidOperationException("Must call Initialize() before EvaluateWithEnvironment()");

        if (episodesPerIndividual > GPUEvolvionState.MAX_EPISODES_PER_INDIVIDUAL)
            throw new ArgumentException(
                $"episodesPerIndividual ({episodesPerIndividual}) exceeds maximum ({GPUEvolvionState.MAX_EPISODES_PER_INDIVIDUAL})");

        int individualCount = individuals.Count;
        int totalEpisodes = individualCount * episodesPerIndividual;
        int observationSize = landscapeConfig.Dimensions;
        int actionSize = landscapeConfig.Dimensions;

        _gpuState.InitializeEnvironments(totalEpisodes, observationSize, actionSize);

        _initializeEpisodesKernel(
            totalEpisodes,
            _gpuState.Environments!.Positions.View,
            _gpuState.Environments.Steps.View,
            _gpuState.Environments.IsTerminal.View,
            landscapeConfig,
            totalEpisodes,
            seed);
        _accelerator.Synchronize();

        int outputRowIdx = spec.RowPlans.Length - 1;

        for (int step = 0; step < landscapeConfig.MaxSteps; step++)
        {
            _computeObservationsKernel(
                totalEpisodes,
                _gpuState.Environments.Observations.View,
                _gpuState.Environments.Positions.View,
                landscapeConfig,
                totalEpisodes);
            _accelerator.Synchronize();

            _setInputsForEpisodesKernel(
                totalEpisodes,
                _gpuState.NodeValues.View,
                _gpuState.Environments.Observations.View,
                totalEpisodes,
                episodesPerIndividual,
                spec.TotalNodes,
                observationSize);
            _accelerator.Synchronize();

            for (int rowIdx = 1; rowIdx < spec.RowPlans.Length; rowIdx++)
            {
                _evaluateRowForEpisodesKernel(
                    totalEpisodes,
                    _gpuState.NodeValues.View,
                    _gpuState.Edges.View,
                    _gpuState.Individuals.Weights.View,
                    _gpuState.Individuals.Biases.View,
                    _gpuState.Individuals.Activations.View,
                    _gpuState.Individuals.NodeParams.View,
                    _gpuState.RowPlans.View,
                    rowIdx,
                    totalEpisodes,
                    episodesPerIndividual,
                    spec.TotalNodes,
                    spec.TotalEdges);
                _accelerator.Synchronize();
            }

            _getOutputsForEpisodesKernel(
                totalEpisodes,
                _gpuState.NodeValues.View,
                _gpuState.Environments.Actions.View,
                _gpuState.RowPlans.View,
                outputRowIdx,
                totalEpisodes,
                spec.TotalNodes,
                actionSize);
            _accelerator.Synchronize();

            _stepEnvironmentKernel(
                totalEpisodes,
                _gpuState.Environments.Positions.View,
                _gpuState.Environments.Actions.View,
                _gpuState.Environments.Steps.View,
                _gpuState.Environments.IsTerminal.View,
                landscapeConfig,
                totalEpisodes);
            _accelerator.Synchronize();
        }

        _finalizeFitnessKernel(
            individualCount,
            _gpuState.FitnessValues.View,
            _gpuState.Environments.Positions.View,
            _gpuState.Environments.FinalLandscapeValues.View,
            landscapeConfig,
            individualCount,
            episodesPerIndividual);
        _accelerator.Synchronize();

        return _gpuState.FitnessValues.GetAsArray1D().Take(individualCount).ToArray();
    }

    public float[] EvaluateWithXOR(
        SpeciesSpec spec,
        List<Individual> individuals,
        int episodesPerIndividual,
        int seed)
    {
        if (_gpuState == null)
            throw new InvalidOperationException("Must call Initialize() before EvaluateWithXOR()");

        int individualCount = individuals.Count;
        int totalEpisodes = individualCount * episodesPerIndividual;
        int observationSize = 2;
        int actionSize = 1;

        using var xorEnv = new GPUXOREnvironmentBatch(_accelerator, totalEpisodes);

        _initializeXORKernel(
            totalEpisodes,
            xorEnv.CurrentCases.View,
            xorEnv.TotalErrors.View,
            xorEnv.IsTerminal.View,
            totalEpisodes);
        _accelerator.Synchronize();

        int outputRowIdx = spec.RowPlans.Length - 1;

        for (int testCase = 0; testCase < 4; testCase++)
        {
            _getXORObservationsKernel(
                totalEpisodes,
                xorEnv.Observations.View,
                xorEnv.CurrentCases.View,
                totalEpisodes);
            _accelerator.Synchronize();

            _setInputsForEpisodesKernel(
                totalEpisodes,
                _gpuState.NodeValues.View,
                xorEnv.Observations.View,
                totalEpisodes,
                episodesPerIndividual,
                spec.TotalNodes,
                observationSize);
            _accelerator.Synchronize();

            for (int rowIdx = 1; rowIdx < spec.RowPlans.Length; rowIdx++)
            {
                _evaluateRowForEpisodesKernel(
                    totalEpisodes,
                    _gpuState.NodeValues.View,
                    _gpuState.Edges.View,
                    _gpuState.Individuals.Weights.View,
                    _gpuState.Individuals.Biases.View,
                    _gpuState.Individuals.Activations.View,
                    _gpuState.Individuals.NodeParams.View,
                    _gpuState.RowPlans.View,
                    rowIdx,
                    totalEpisodes,
                    episodesPerIndividual,
                    spec.TotalNodes,
                    spec.TotalEdges);
                _accelerator.Synchronize();
            }

            _getOutputsForEpisodesKernel(
                totalEpisodes,
                _gpuState.NodeValues.View,
                xorEnv.Actions.View,
                _gpuState.RowPlans.View,
                outputRowIdx,
                totalEpisodes,
                spec.TotalNodes,
                actionSize);
            _accelerator.Synchronize();

            _stepXORKernel(
                totalEpisodes,
                xorEnv.Actions.View,
                xorEnv.CurrentCases.View,
                xorEnv.TotalErrors.View,
                xorEnv.IsTerminal.View,
                totalEpisodes);
            _accelerator.Synchronize();
        }

        _finalizeXORFitnessKernel(
            individualCount,
            _gpuState.FitnessValues.View,
            xorEnv.TotalErrors.View,
            individualCount,
            episodesPerIndividual);
        _accelerator.Synchronize();

        return _gpuState.FitnessValues.GetAsArray1D().Take(individualCount).ToArray();
    }

    public float[] EvaluateWithSpiral(
        SpeciesSpec spec,
        List<Individual> individuals,
        int pointsPerSpiral,
        float noise,
        int seed)
    {
        if (_gpuState == null)
            throw new InvalidOperationException("Must call Initialize() before EvaluateWithSpiral()");

        int individualCount = individuals.Count;
        int episodesPerIndividual = 1;
        int totalEpisodes = individualCount * episodesPerIndividual;
        int totalPoints = pointsPerSpiral * 2;
        int observationSize = 2;
        int actionSize = 1;

        using var spiralEnv = new GPUSpiralEnvironmentBatch(_accelerator, totalEpisodes, totalPoints);

        _initializeSpiralPointsKernel(
            totalPoints,
            spiralEnv.SpiralPoints.View,
            pointsPerSpiral,
            noise);
        _accelerator.Synchronize();

        _initializeSpiralEpisodesKernel(
            totalEpisodes,
            spiralEnv.CurrentCases.View,
            spiralEnv.TotalErrors.View,
            spiralEnv.IsTerminal.View,
            totalEpisodes);
        _accelerator.Synchronize();

        int outputRowIdx = spec.RowPlans.Length - 1;

        for (int pointIdx = 0; pointIdx < totalPoints; pointIdx++)
        {
            _getSpiralObservationsKernel(
                totalEpisodes,
                spiralEnv.Observations.View,
                spiralEnv.SpiralPoints.View,
                spiralEnv.CurrentCases.View,
                totalEpisodes,
                totalPoints);
            _accelerator.Synchronize();

            _setInputsForEpisodesKernel(
                totalEpisodes,
                _gpuState.NodeValues.View,
                spiralEnv.Observations.View,
                totalEpisodes,
                episodesPerIndividual,
                spec.TotalNodes,
                observationSize);
            _accelerator.Synchronize();

            for (int rowIdx = 1; rowIdx < spec.RowPlans.Length; rowIdx++)
            {
                _evaluateRowForEpisodesKernel(
                    totalEpisodes,
                    _gpuState.NodeValues.View,
                    _gpuState.Edges.View,
                    _gpuState.Individuals.Weights.View,
                    _gpuState.Individuals.Biases.View,
                    _gpuState.Individuals.Activations.View,
                    _gpuState.Individuals.NodeParams.View,
                    _gpuState.RowPlans.View,
                    rowIdx,
                    totalEpisodes,
                    episodesPerIndividual,
                    spec.TotalNodes,
                    spec.TotalEdges);
                _accelerator.Synchronize();
            }

            _getOutputsForEpisodesKernel(
                totalEpisodes,
                _gpuState.NodeValues.View,
                spiralEnv.Actions.View,
                _gpuState.RowPlans.View,
                outputRowIdx,
                totalEpisodes,
                spec.TotalNodes,
                actionSize);
            _accelerator.Synchronize();

            _stepSpiralKernel(
                totalEpisodes,
                spiralEnv.Actions.View,
                spiralEnv.SpiralPoints.View,
                spiralEnv.CurrentCases.View,
                spiralEnv.TotalErrors.View,
                spiralEnv.IsTerminal.View,
                totalEpisodes,
                totalPoints);
            _accelerator.Synchronize();
        }

        _finalizeSpiralFitnessKernel(
            individualCount,
            _gpuState.FitnessValues.View,
            spiralEnv.TotalErrors.View,
            individualCount,
            episodesPerIndividual,
            totalPoints);
        _accelerator.Synchronize();

        return _gpuState.FitnessValues.GetAsArray1D().Take(individualCount).ToArray();
    }

    public void Dispose()
    {
        _gpuState?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
