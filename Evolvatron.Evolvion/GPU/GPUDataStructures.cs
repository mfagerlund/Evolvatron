using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-compatible edge representation (blittable struct).
/// Represents a weighted connection from Source to Dest node.
/// </summary>
public struct GPUEdge
{
    public int Source;
    public int Dest;

    public GPUEdge(int source, int dest)
    {
        Source = source;
        Dest = dest;
    }
}

/// <summary>
/// GPU-compatible row plan (already blittable, copied from RowPlan).
/// Contains metadata for evaluating one row of the neural network.
/// </summary>
public struct GPURowPlan
{
    public int NodeStart;
    public int NodeCount;
    public int EdgeStart;
    public int EdgeCount;

    public GPURowPlan(int nodeStart, int nodeCount, int edgeStart, int edgeCount)
    {
        NodeStart = nodeStart;
        NodeCount = nodeCount;
        EdgeStart = edgeStart;
        EdgeCount = edgeCount;
    }

    public GPURowPlan(RowPlan plan)
    {
        NodeStart = plan.NodeStart;
        NodeCount = plan.NodeCount;
        EdgeStart = plan.EdgeStart;
        EdgeCount = plan.EdgeCount;
    }
}

/// <summary>
/// GPU individual batch representation using ILGPU memory buffers.
/// All individuals' data stored in contiguous SoA layout.
/// </summary>
public class GPUIndividualBatch : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> Weights { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Biases { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> NodeParams { get; private set; }
    public MemoryBuffer1D<byte, Stride1D.Dense> Activations { get; private set; }

    public int IndividualCount { get; private set; }
    public int WeightsPerIndividual { get; private set; }
    public int NodesPerIndividual { get; private set; }

    public GPUIndividualBatch(
        Accelerator accelerator,
        int individualCount,
        int weightsPerIndividual,
        int nodesPerIndividual)
    {
        IndividualCount = individualCount;
        WeightsPerIndividual = weightsPerIndividual;
        NodesPerIndividual = nodesPerIndividual;

        Weights = accelerator.Allocate1D<float>(individualCount * weightsPerIndividual);
        Biases = accelerator.Allocate1D<float>(individualCount * nodesPerIndividual);
        NodeParams = accelerator.Allocate1D<float>(individualCount * nodesPerIndividual * 4);
        Activations = accelerator.Allocate1D<byte>(individualCount * nodesPerIndividual);
    }

    public void Dispose()
    {
        Weights?.Dispose();
        Biases?.Dispose();
        NodeParams?.Dispose();
        Activations?.Dispose();
    }
}

/// <summary>
/// GPU environment batch representation.
/// Stores state for multiple episodes running in parallel.
/// </summary>
public class GPUEnvironmentBatch : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> Observations { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Positions { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Actions { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> FinalLandscapeValues { get; private set; }
    public MemoryBuffer1D<int, Stride1D.Dense> Steps { get; private set; }
    public MemoryBuffer1D<byte, Stride1D.Dense> IsTerminal { get; private set; }

    public int EpisodeCount { get; private set; }
    public int ObservationSize { get; private set; }
    public int ActionSize { get; private set; }

    public GPUEnvironmentBatch(
        Accelerator accelerator,
        int episodeCount,
        int observationSize,
        int actionSize)
    {
        EpisodeCount = episodeCount;
        ObservationSize = observationSize;
        ActionSize = actionSize;

        Observations = accelerator.Allocate1D<float>(episodeCount * observationSize);
        Positions = accelerator.Allocate1D<float>(episodeCount * actionSize);
        Actions = accelerator.Allocate1D<float>(episodeCount * actionSize);
        FinalLandscapeValues = accelerator.Allocate1D<float>(episodeCount);
        Steps = accelerator.Allocate1D<int>(episodeCount);
        IsTerminal = accelerator.Allocate1D<byte>(episodeCount);
    }

    public void Dispose()
    {
        Observations?.Dispose();
        Positions?.Dispose();
        Actions?.Dispose();
        FinalLandscapeValues?.Dispose();
        Steps?.Dispose();
        IsTerminal?.Dispose();
    }
}

/// <summary>
/// GPU XOR environment batch representation.
/// </summary>
public class GPUXOREnvironmentBatch : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> Observations { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Actions { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> TotalErrors { get; private set; }
    public MemoryBuffer1D<int, Stride1D.Dense> CurrentCases { get; private set; }
    public MemoryBuffer1D<byte, Stride1D.Dense> IsTerminal { get; private set; }

    public int EpisodeCount { get; private set; }

    public GPUXOREnvironmentBatch(Accelerator accelerator, int episodeCount)
    {
        EpisodeCount = episodeCount;

        Observations = accelerator.Allocate1D<float>(episodeCount * 2);
        Actions = accelerator.Allocate1D<float>(episodeCount);
        TotalErrors = accelerator.Allocate1D<float>(episodeCount);
        CurrentCases = accelerator.Allocate1D<int>(episodeCount);
        IsTerminal = accelerator.Allocate1D<byte>(episodeCount);
    }

    public void Dispose()
    {
        Observations?.Dispose();
        Actions?.Dispose();
        TotalErrors?.Dispose();
        CurrentCases?.Dispose();
        IsTerminal?.Dispose();
    }
}

/// <summary>
/// GPU Spiral environment batch representation.
/// </summary>
public class GPUSpiralEnvironmentBatch : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> SpiralPoints { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Observations { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Actions { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> TotalErrors { get; private set; }
    public MemoryBuffer1D<int, Stride1D.Dense> CurrentCases { get; private set; }
    public MemoryBuffer1D<byte, Stride1D.Dense> IsTerminal { get; private set; }

    public int EpisodeCount { get; private set; }
    public int TotalPoints { get; private set; }

    public GPUSpiralEnvironmentBatch(Accelerator accelerator, int episodeCount, int totalPoints)
    {
        EpisodeCount = episodeCount;
        TotalPoints = totalPoints;

        SpiralPoints = accelerator.Allocate1D<float>(totalPoints * 3);
        Observations = accelerator.Allocate1D<float>(episodeCount * 2);
        Actions = accelerator.Allocate1D<float>(episodeCount);
        TotalErrors = accelerator.Allocate1D<float>(episodeCount);
        CurrentCases = accelerator.Allocate1D<int>(episodeCount);
        IsTerminal = accelerator.Allocate1D<byte>(episodeCount);
    }

    public void Dispose()
    {
        SpiralPoints?.Dispose();
        Observations?.Dispose();
        Actions?.Dispose();
        TotalErrors?.Dispose();
        CurrentCases?.Dispose();
        IsTerminal?.Dispose();
    }
}

/// <summary>
/// Configuration parameters for landscape environment on GPU.
/// Blittable struct that can be passed to kernels.
/// </summary>
public struct GPULandscapeConfig
{
    public int Dimensions;
    public int MaxSteps;
    public float StepSize;
    public float MinBound;
    public float MaxBound;
    public byte LandscapeType;

    public GPULandscapeConfig(
        int dimensions,
        int maxSteps,
        float stepSize,
        float minBound,
        float maxBound,
        byte landscapeType)
    {
        Dimensions = dimensions;
        MaxSteps = maxSteps;
        StepSize = stepSize;
        MinBound = minBound;
        MaxBound = maxBound;
        LandscapeType = landscapeType;
    }
}

/// <summary>
/// Landscape function types for GPU evaluation.
/// </summary>
public enum LandscapeType : byte
{
    Sphere = 0,
    Rosenbrock = 1,
    Rastrigin = 2,
    Ackley = 3,
    Schwefel = 4
}

/// <summary>
/// Configuration parameters for XOR environment on GPU.
/// Blittable struct that can be passed to kernels.
/// </summary>
public struct GPUXORConfig
{
    public int MaxSteps;

    public GPUXORConfig(int maxSteps)
    {
        MaxSteps = maxSteps;
    }
}

/// <summary>
/// Configuration parameters for Spiral environment on GPU.
/// Blittable struct that can be passed to kernels.
/// </summary>
public struct GPUSpiralConfig
{
    public int MaxSteps;
    public int PointsPerSpiral;
    public float Noise;

    public GPUSpiralConfig(int maxSteps, int pointsPerSpiral, float noise)
    {
        MaxSteps = maxSteps;
        PointsPerSpiral = pointsPerSpiral;
        Noise = noise;
    }
}
