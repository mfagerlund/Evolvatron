using System.Runtime.InteropServices;
using ILGPU;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Minimal NN views for dense fully-connected networks.
/// No Edges, RowPlans, Activations, NodeParams, or NodeValues.
/// ~47% memory reduction vs sparse NNViews.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DenseNNViews
{
    public ArrayView<float> Weights;    // [totalPop x totalWeightsPerNet]
    public ArrayView<float> Biases;     // [totalPop x totalBiasesPerNet]
    public ArrayView<int> LayerSizes;   // [numLayers] — shared across all individuals
}
