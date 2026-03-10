using System.Runtime.InteropServices;
using ILGPU;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Bundles reward zone ArrayViews into a single kernel parameter.
/// Contains both shared zone definitions and per-world zone state.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ZoneViews
{
    // Zone definitions (shared across all worlds)
    public ArrayView<GPUCheckpoint> Checkpoints;
    public ArrayView<GPUDangerZone> DangerZones;
    public ArrayView<GPUSpeedZone> SpeedZones;
    public ArrayView<GPUAttractor> Attractors;

    // Per-world zone state
    public ArrayView<float> ZoneRewardAccum;
    public ArrayView<int> CheckpointProgress;
    public ArrayView<int> AttractorContacted;
}
