using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Blittable config for the double pole balancing GPU kernel.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DoublePoleConfig
{
    // Environment
    public int MaxSteps;
    public int TicksPerLaunch;
    public float TrackLengthHalf;
    public float PoleAngleThreshold;
    public int IncludeVelocity; // 0 = position-only (hard), 1 = with velocity

    // NN layout
    public int TotalNodes;
    public int TotalEdges;
    public int NumRows;
    public int InputSize;  // 3 (no velocity) or 6 (with velocity), plus ContextSize
    public int OutputSize; // 1 (action) plus ContextSize for Elman

    // Recurrence
    public int ContextSize; // 0 = feedforward, >0 = recurrent with N context values
    public int IsJordan;    // 1 = Jordan (action feedback), 0 = Elman (extra outputs)

    // Fitness
    public int GruauEnabled; // 1 = add Gruau anti-jiggle bonus for survivors
}

/// <summary>
/// Minimal episode views for double pole balancing.
/// No throttle/gimbal/waggle/landed — just terminal state and fitness.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DoublePoleEpisodeViews
{
    public ArrayView<byte> IsTerminal;
    public ArrayView<int> StepCounters;
    public ArrayView<float> FitnessValues;
    public ArrayView<float> JiggleBuffer;   // circular buffer: worldCount * 100
    public ArrayView<float> ContextBuffer;  // recurrent context: worldCount * ContextSize
}
