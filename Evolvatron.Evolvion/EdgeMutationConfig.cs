namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration for edge topology mutation operators
/// </summary>
public class EdgeMutationConfig
{
    // Core topology mutations (Phase 10 Optuna optimized)
    public float EdgeAdd { get; set; } = 0.016f;
    public float EdgeDeleteRandom { get; set; } = 0.004f;
    public float EdgeSplit { get; set; } = 0.043f;

    // Advanced topology mutations (Phase 10 Optuna optimized)
    public float EdgeRedirect { get; set; } = 0.093f;
    public float EdgeSwap { get; set; } = 0.029f;

    // Weak edge pruning configuration
    public WeakEdgePruningConfig WeakEdgePruning { get; set; } = new();
}

/// <summary>
/// Configuration for automatic weak edge pruning
/// </summary>
public class WeakEdgePruningConfig
{
    /// <summary>
    /// Enable weak edge pruning (Phase 10 Optuna: optimized)
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Weight threshold below which edges are considered weak
    /// (Phase 10 Optuna: optimized)
    /// </summary>
    public float Threshold { get; set; } = 0.016f;

    /// <summary>
    /// Base probability of pruning weak edges (scaled by weight magnitude)
    /// (Phase 10 Optuna: optimized)
    /// </summary>
    public float BasePruneRate { get; set; } = 0.799f;

    /// <summary>
    /// Apply pruning when new species is born
    /// (Phase 10 Optuna: disabled - better to prune during evolution)
    /// </summary>
    public bool ApplyOnSpeciesBirth { get; set; } = false;

    /// <summary>
    /// Apply pruning during normal evolution (not just at species birth)
    /// (Phase 10 Optuna: enabled - continuous pruning is better)
    /// </summary>
    public bool ApplyDuringEvolution { get; set; } = true;
}
