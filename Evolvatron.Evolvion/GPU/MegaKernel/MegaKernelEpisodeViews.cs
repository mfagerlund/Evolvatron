using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Bundles neural network ArrayViews into a single kernel parameter.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct NNViews
{
    public ArrayView<float> NodeValues;
    public ArrayView<GPUEdge> Edges;
    public ArrayView<float> Weights;
    public ArrayView<float> Biases;
    public ArrayView<byte> Activations;
    public ArrayView<float> NodeParams;
    public ArrayView<GPURowPlan> RowPlans;
}

/// <summary>
/// Bundles per-episode state ArrayViews into a single kernel parameter.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct EpisodeViews
{
    public ArrayView<float> CurrentThrottle;
    public ArrayView<float> CurrentGimbal;
    public ArrayView<byte> IsTerminal;
    public ArrayView<byte> HasLanded;
    public ArrayView<int> StepCounters;
    public ArrayView<float> FitnessValues;
    public ArrayView<float> WaggleAccum;
    public ArrayView<int> SettledSteps;
}
