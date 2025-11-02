using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// Manages GPU memory allocation and transfers for Evolvion neural network evaluation.
/// Follows the pattern from Evolvatron.Rigidon.GPU.GPUWorldState.
/// </summary>
public class GPUEvolvionState : IDisposable
{
    private readonly Accelerator _accelerator;

    public MemoryBuffer1D<GPUEdge, Stride1D.Dense> Edges { get; private set; }
    public MemoryBuffer1D<GPURowPlan, Stride1D.Dense> RowPlans { get; private set; }

    public GPUIndividualBatch Individuals { get; private set; }

    public MemoryBuffer1D<float, Stride1D.Dense> NodeValues { get; private set; }

    public GPUEnvironmentBatch? Environments { get; private set; }

    public MemoryBuffer1D<float, Stride1D.Dense> FitnessValues { get; private set; }

    public int MaxIndividuals { get; private set; }
    public int MaxNodes { get; private set; }
    public int MaxEdges { get; private set; }

    public GPUEvolvionState(
        Accelerator accelerator,
        int maxIndividuals,
        int maxNodes,
        int maxEdges)
    {
        _accelerator = accelerator;
        MaxIndividuals = maxIndividuals;
        MaxNodes = maxNodes;
        MaxEdges = maxEdges;

        Edges = accelerator.Allocate1D<GPUEdge>(maxEdges);
        RowPlans = accelerator.Allocate1D<GPURowPlan>(100);

        Individuals = new GPUIndividualBatch(
            accelerator,
            maxIndividuals,
            maxEdges,
            maxNodes);

        NodeValues = accelerator.Allocate1D<float>(maxIndividuals * maxNodes * 10);

        FitnessValues = accelerator.Allocate1D<float>(maxIndividuals);
    }

    /// <summary>
    /// Uploads species topology to GPU (edges and row plans).
    /// This is done once per species and remains on GPU during evaluation.
    /// </summary>
    public void UploadTopology(SpeciesSpec spec)
    {
        var gpuEdges = spec.Edges.Select(e => new GPUEdge(e.Source, e.Dest)).ToArray();
        Edges.View.SubView(0, gpuEdges.Length).CopyFromCPU(gpuEdges);

        var gpuRowPlans = spec.RowPlans.Select(p => new GPURowPlan(p)).ToArray();
        RowPlans.View.SubView(0, gpuRowPlans.Length).CopyFromCPU(gpuRowPlans);
    }

    /// <summary>
    /// Uploads individual parameters (weights, biases, activations, node params) to GPU.
    /// This is done at the start of each generation for all individuals in a species.
    /// </summary>
    public void UploadIndividuals(List<Individual> individuals, SpeciesSpec spec)
    {
        int individualCount = individuals.Count;
        int weightsPerIndividual = spec.TotalEdges;
        int nodesPerIndividual = spec.TotalNodes;

        var allWeights = new float[individualCount * weightsPerIndividual];
        var allBiases = new float[individualCount * nodesPerIndividual];
        var allNodeParams = new float[individualCount * nodesPerIndividual * 4];
        var allActivations = new byte[individualCount * nodesPerIndividual];

        for (int i = 0; i < individualCount; i++)
        {
            var individual = individuals[i];

            Array.Copy(
                individual.Weights, 0,
                allWeights, i * weightsPerIndividual,
                weightsPerIndividual);

            Array.Copy(
                individual.Biases, 0,
                allBiases, i * nodesPerIndividual,
                nodesPerIndividual);

            Array.Copy(
                individual.NodeParams, 0,
                allNodeParams, i * nodesPerIndividual * 4,
                nodesPerIndividual * 4);

            for (int j = 0; j < nodesPerIndividual; j++)
            {
                allActivations[i * nodesPerIndividual + j] = (byte)individual.Activations[j];
            }
        }

        Individuals.Weights.View.SubView(0, allWeights.Length).CopyFromCPU(allWeights);
        Individuals.Biases.View.SubView(0, allBiases.Length).CopyFromCPU(allBiases);
        Individuals.NodeParams.View.SubView(0, allNodeParams.Length).CopyFromCPU(allNodeParams);
        Individuals.Activations.View.SubView(0, allActivations.Length).CopyFromCPU(allActivations);
    }

    /// <summary>
    /// Downloads fitness values from GPU back to CPU individuals.
    /// This is done at the end of each generation.
    /// </summary>
    public void DownloadFitness(List<Individual> individuals)
    {
        var fitnessArray = FitnessValues.GetAsArray1D();

        for (int i = 0; i < individuals.Count; i++)
        {
            var individual = individuals[i];
            individual.Fitness = fitnessArray[i];
            individuals[i] = individual;
        }
    }

    /// <summary>
    /// Initializes environment batch for parallel episode execution.
    /// </summary>
    public void InitializeEnvironments(int episodeCount, int observationSize, int actionSize)
    {
        Environments?.Dispose();
        Environments = new GPUEnvironmentBatch(
            _accelerator,
            episodeCount,
            observationSize,
            actionSize);
    }

    /// <summary>
    /// Downloads final landscape values from GPU for fitness calculation.
    /// Returns array of landscape values (one per episode).
    /// </summary>
    public float[] GetEnvironmentResults()
    {
        if (Environments == null)
            throw new InvalidOperationException("Environments not initialized");

        return Environments.FinalLandscapeValues.GetAsArray1D();
    }

    public void Dispose()
    {
        Edges?.Dispose();
        RowPlans?.Dispose();
        Individuals?.Dispose();
        NodeValues?.Dispose();
        Environments?.Dispose();
        FitnessValues?.Dispose();
    }
}
