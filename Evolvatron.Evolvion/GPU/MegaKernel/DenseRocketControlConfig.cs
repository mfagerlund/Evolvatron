using System.Runtime.InteropServices;
using ILGPU;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Dense NN layout + tracking-reward config for the Phase-1 maneuvering controller
/// (see docs/phase1_controller_spec.md). Passed alongside MegaKernelConfig (physics).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DenseRocketControlConfig
{
    // NN layout (mirrors DenseRocketNNConfig)
    public int NumLayers;
    public int TotalWeightsPerNet;
    public int TotalBiasesPerNet;

    // Command schedule
    public int SegmentLength;       // K: steps per piecewise-constant command segment
    public int SegmentsPerEpisode;  // number of command segments per episode

    // Tracking reward
    public float VErrScale;          // velocity-error scale (m/s) → track = 1 - min(verr/scale, 1)
    public float RewardTrackWeight;  // per-step weight on tracking quality
    public float RewardEffortWeight; // per-step penalty on (dThrottle^2 + dGimbal^2)
    public float AngVelPenalty;      // per-step penalty on |angVel|/10 (0 = off)
    public float TumblePenalty;      // one-time penalty subtracted on airborne tumble terminal

    // Elman recurrence: ContextSize feedback outputs are written to ContextBuffer each step
    // and read back as the last ContextSize input slots next step. 0 = plain reactive controller.
    public int ContextSize;
}

/// <summary>
/// Per-world command-schedule + reward-accumulator views for the control kernel.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ControlViews
{
    public ArrayView<float> CmdVx;            // [worldCount * SegmentsPerEpisode] desired world-frame vx
    public ArrayView<float> CmdVy;            // [worldCount * SegmentsPerEpisode] desired world-frame vy
    public ArrayView<float> TrackRewardAccum; // [worldCount] running tracking reward
    public ArrayView<float> ContextBuffer;    // [worldCount * ContextSize] Elman feedback state (>=1 elem)
}
