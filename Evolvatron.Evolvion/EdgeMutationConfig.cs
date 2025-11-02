namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration for edge topology mutation operators
/// </summary>
public class EdgeMutationConfig
{
    // Core topology mutations (Phase 10 Rosenbrock: Trial 138 optimized for valley navigation)
    public float EdgeAdd { get; set; } = 0.007f;  // Reduced: very conservative addition
    public float EdgeDeleteRandom { get; set; } = 0.042f;  // 10.5x increase: aggressive simplification!
    public float EdgeSplit { get; set; } = 0.001f;  // Minimal: avoid complexity

    // Advanced topology mutations (Phase 10 Rosenbrock: Trial 138 optimized)
    public float EdgeRedirect { get; set; } = 0.132f;  // Increased: rewire working pathways
    public float EdgeSwap { get; set; } = 0.047f;  // Increased: more shuffling

    // Weak edge pruning configuration
    public WeakEdgePruningConfig WeakEdgePruning { get; set; } = new();
}

/// <summary>
/// Configuration for automatic weak edge pruning
/// </summary>
public class WeakEdgePruningConfig
{
    /// <summary>
    /// Enable weak edge pruning
    /// Default: False (Phase 10 Rosenbrock: Trial 138 - DISABLED for valley navigation!)
    /// Previous: True (Phase 10 spiral - but valleys need more network diversity)
    /// Networks retain more structural diversity when pruning is disabled.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Weight threshold below which edges are considered weak
    /// Default: 0.078 (Phase 10 Rosenbrock: Trial 138)
    /// Previous: 0.016 (Phase 10 spiral)
    /// </summary>
    public float Threshold { get; set; } = 0.078f;

    /// <summary>
    /// Base probability of pruning weak edges (scaled by weight magnitude)
    /// Default: 0.676 (Phase 10 Rosenbrock: Trial 138)
    /// Previous: 0.799 (Phase 10 spiral)
    /// </summary>
    public float BasePruneRate { get; set; } = 0.676f;

    /// <summary>
    /// Apply pruning when new species is born
    /// Default: False (Phase 10 Rosenbrock: Trial 138 - same as spiral)
    /// </summary>
    public bool ApplyOnSpeciesBirth { get; set; } = false;

    /// <summary>
    /// Apply pruning during normal evolution (not just at species birth)
    /// Default: True (Phase 10 Rosenbrock: Trial 138 - same as spiral)
    /// </summary>
    public bool ApplyDuringEvolution { get; set; } = true;
}
