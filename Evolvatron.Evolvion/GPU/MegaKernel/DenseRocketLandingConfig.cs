using System.Runtime.InteropServices;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Dense NN layout config for the rocket landing kernel.
/// Passed alongside MegaKernelConfig (which handles physics + task params).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DenseRocketNNConfig
{
    public int NumLayers;
    public int TotalWeightsPerNet;
    public int TotalBiasesPerNet;
}
