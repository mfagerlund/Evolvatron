using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// GPU state for batched environment logic (targets, rewards, observations, actions).
/// Manages per-world environment state for parallel simulation of multiple worlds.
///
/// Memory layout is designed for efficient GPU kernel access with coalesced memory patterns.
/// All arrays are "flattened" with world index as the outer dimension.
/// </summary>
public class GPUBatchedEnvironmentState : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _worldCount;
    private readonly int _targetsPerWorld;
    private readonly int _observationsPerWorld;
    private readonly int _actionsPerWorld;
    private bool _disposed;

    // Target state: positions and active flags
    // Positions: [world0_target0_x, world0_target0_y, world0_target1_x, ...]
    public MemoryBuffer1D<float, Stride1D.Dense> TargetPositions { get; private set; }
    public MemoryBuffer1D<byte, Stride1D.Dense> TargetActive { get; private set; }

    // Per-world episode state
    public MemoryBuffer1D<float, Stride1D.Dense> CumulativeRewards { get; private set; }
    public MemoryBuffer1D<int, Stride1D.Dense> StepCounters { get; private set; }
    public MemoryBuffer1D<byte, Stride1D.Dense> IsTerminal { get; private set; }
    public MemoryBuffer1D<int, Stride1D.Dense> TargetsCollected { get; private set; }

    // Neural network interface
    // Observations: [world0_obs0..N, world1_obs0..N, ...]
    public MemoryBuffer1D<float, Stride1D.Dense> Observations { get; private set; }
    // Actions: [world0_action0..M, world1_action0..M, ...]
    public MemoryBuffer1D<float, Stride1D.Dense> Actions { get; private set; }

    // Final fitness values (computed at episode end)
    public MemoryBuffer1D<float, Stride1D.Dense> FitnessValues { get; private set; }

    public int WorldCount => _worldCount;
    public int TargetsPerWorld => _targetsPerWorld;
    public int ObservationsPerWorld => _observationsPerWorld;
    public int ActionsPerWorld => _actionsPerWorld;

    public GPUBatchedEnvironmentState(
        Accelerator accelerator,
        int worldCount,
        int targetsPerWorld,
        int observationsPerWorld,
        int actionsPerWorld)
    {
        _accelerator = accelerator;
        _worldCount = worldCount;
        _targetsPerWorld = targetsPerWorld;
        _observationsPerWorld = observationsPerWorld;
        _actionsPerWorld = actionsPerWorld;

        AllocateBuffers();
    }

    private void AllocateBuffers()
    {
        int totalTargets = _worldCount * _targetsPerWorld;

        // Target positions: 2 floats (x,y) per target
        TargetPositions = _accelerator.Allocate1D<float>(totalTargets * 2);
        TargetActive = _accelerator.Allocate1D<byte>(totalTargets);

        // Per-world state
        CumulativeRewards = _accelerator.Allocate1D<float>(_worldCount);
        StepCounters = _accelerator.Allocate1D<int>(_worldCount);
        IsTerminal = _accelerator.Allocate1D<byte>(_worldCount);
        TargetsCollected = _accelerator.Allocate1D<int>(_worldCount);
        FitnessValues = _accelerator.Allocate1D<float>(_worldCount);

        // Neural network buffers
        Observations = _accelerator.Allocate1D<float>(_worldCount * _observationsPerWorld);
        Actions = _accelerator.Allocate1D<float>(_worldCount * _actionsPerWorld);
    }

    /// <summary>
    /// Reset all worlds to initial state.
    /// </summary>
    public void ResetAll()
    {
        CumulativeRewards.MemSetToZero();
        StepCounters.MemSetToZero();
        IsTerminal.MemSetToZero();
        TargetsCollected.MemSetToZero();
        FitnessValues.MemSetToZero();
        Observations.MemSetToZero();
        Actions.MemSetToZero();
    }

    /// <summary>
    /// Upload initial target positions for all worlds.
    /// </summary>
    public void UploadTargetPositions(float[] positions, byte[] active)
    {
        TargetPositions.CopyFromCPU(positions);
        TargetActive.CopyFromCPU(active);
    }

    /// <summary>
    /// Download fitness values after episode completion.
    /// </summary>
    public float[] DownloadFitnessValues()
    {
        return FitnessValues.GetAsArray1D();
    }

    /// <summary>
    /// Download targets collected per world.
    /// </summary>
    public int[] DownloadTargetsCollected()
    {
        return TargetsCollected.GetAsArray1D();
    }

    /// <summary>
    /// Download terminal flags.
    /// </summary>
    public byte[] DownloadTerminalFlags()
    {
        return IsTerminal.GetAsArray1D();
    }

    /// <summary>
    /// Check if all worlds are terminal (for early exit).
    /// </summary>
    public bool AllTerminal()
    {
        var flags = DownloadTerminalFlags();
        foreach (var f in flags)
        {
            if (f == 0) return false;
        }
        return true;
    }

    /// <summary>
    /// Get target position index for a specific world and target.
    /// Returns index into TargetPositions array (multiply by 2 for x, +1 for y).
    /// </summary>
    public int GetTargetPositionIndex(int worldIdx, int targetIdx)
    {
        return (worldIdx * _targetsPerWorld + targetIdx) * 2;
    }

    /// <summary>
    /// Get observation index for a specific world.
    /// </summary>
    public int GetObservationIndex(int worldIdx)
    {
        return worldIdx * _observationsPerWorld;
    }

    /// <summary>
    /// Get action index for a specific world.
    /// </summary>
    public int GetActionIndex(int worldIdx)
    {
        return worldIdx * _actionsPerWorld;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        TargetPositions?.Dispose();
        TargetActive?.Dispose();
        CumulativeRewards?.Dispose();
        StepCounters?.Dispose();
        IsTerminal?.Dispose();
        TargetsCollected?.Dispose();
        FitnessValues?.Dispose();
        Observations?.Dispose();
        Actions?.Dispose();
    }
}
