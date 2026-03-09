using System.Runtime.InteropServices;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Blittable config for the dense double pole balancing GPU kernel.
/// Replaces TotalNodes/TotalEdges/NumRows with dense NN layout parameters.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DenseDoublePoleConfig
{
    // Environment
    public int MaxSteps;
    public int TicksPerLaunch;
    public float TrackLengthHalf;
    public float PoleAngleThreshold;
    public int IncludeVelocity; // 0 = position-only (hard), 1 = with velocity

    // Dense NN layout
    public int NumLayers;            // total layers including input + output
    public int TotalWeightsPerNet;
    public int TotalBiasesPerNet;
    public int InputSize;            // base observations + contextSize
    public int OutputSize;           // base actions + contextSize

    // Recurrence
    public int ContextSize; // 0 = feedforward, >0 = Elman recurrent
    public int IsJordan;    // 1 = Jordan (action feedback), 0 = Elman (extra outputs)

    // Fitness
    public int GruauEnabled; // 1 = add Gruau anti-jiggle bonus for survivors
}
