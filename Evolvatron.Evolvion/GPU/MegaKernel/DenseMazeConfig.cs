using System.Runtime.InteropServices;
using ILGPU;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Layout + reward config for the Phase-2 maze navigator (see docs/phase2_maze_spec.md).
/// Carries BOTH network layouts: the evolved maze policy (NN#1, per-world) and the frozen
/// controller (NN#2, shared). Passed alongside MegaKernelConfig (physics + sensors + obstacles).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DenseMazeConfig
{
    // Maze policy NN (NN#1, evolved per-world). Input = 6 + SensorCount, Output = 2 (velocity cmd).
    public int MazeNumLayers;
    public int MazeTotalWeights;
    public int MazeTotalBiases;
    public int MazeInputSize;
    public int MazeOutputSize;

    // Frozen controller NN (NN#2, shared weights). Input = 9, Output = 2 (throttle, gimbal).
    public int CtrlNumLayers;
    public int CtrlTotalWeights;
    public int CtrlTotalBiases;

    // Command interface: maze policy raw output ∈ [-1,1] → velocity = raw * CmdSpeedMax.
    public float CmdSpeedMax;

    // Goal / navigation. Goal POSITION is per-world (MazeViews.GoalX/GoalY) so a replay grid can
    // show different goals per cell; training just replicates one goal across all worlds.
    public float GoalRadius;
    public float PosScale;   // normalizer for goal-relative position observation

    // Reward shaping.
    public float ProgressWeight;    // per-step weight on (prevDist - dist) toward goal
    public float GoalBonus;         // one-time bonus on reaching goal (scaled by remaining-time haste)
    public float CollisionPenalty;  // one-time penalty on obstacle contact
    public float TumblePenalty;     // one-time penalty on airborne tumble
    public float StepPenalty;       // small per-step time cost (encourages reaching goal sooner)
}

/// <summary>
/// Per-world views for the maze kernel: the frozen controller's shared weights, the controller's
/// own observation/action scratch buffers, and the navigation progress/reward accumulators.
/// (The maze policy's per-world weights travel in a standard DenseNNViews.)
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct MazeViews
{
    public ArrayView<float> CtrlWeights;    // shared frozen controller weights (ONE net, indexed with weightWorldIdx=0)
    public ArrayView<float> CtrlBiases;
    public ArrayView<int> CtrlLayerSizes;
    public ArrayView<float> CtrlObs;        // [worldCount * 9] controller observation scratch
    public ArrayView<float> CtrlAct;        // [worldCount * 2] controller action scratch
    public ArrayView<float> GoalX;          // [worldCount] per-world goal X
    public ArrayView<float> GoalY;          // [worldCount] per-world goal Y
    public ArrayView<float> PrevDist;       // [worldCount] previous distance to goal (progress reward)
    public ArrayView<float> RewardAccum;    // [worldCount] running navigation reward
}
