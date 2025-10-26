namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration for edge topology mutation operators
/// </summary>
public class EdgeMutationConfig
{
    // Core topology mutations (from Evolvion.md)
    public float EdgeAdd { get; set; } = 0.05f;
    public float EdgeDeleteRandom { get; set; } = 0.01f;
    public float EdgeSplit { get; set; } = 0.01f;

    // Advanced topology mutations
    public float EdgeRedirect { get; set; } = 0.03f;
    public float EdgeDuplicate { get; set; } = 0.01f;
    public float EdgeMerge { get; set; } = 0.02f;
    public float EdgeSwap { get; set; } = 0.02f;

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
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Weight threshold below which edges are considered weak
    /// </summary>
    public float Threshold { get; set; } = 0.01f;

    /// <summary>
    /// Base probability of pruning weak edges (scaled by weight magnitude)
    /// </summary>
    public float BasePruneRate { get; set; } = 0.7f;

    /// <summary>
    /// Apply pruning when new species is born
    /// </summary>
    public bool ApplyOnSpeciesBirth { get; set; } = true;

    /// <summary>
    /// Apply pruning during normal evolution (not just at species birth)
    /// </summary>
    public bool ApplyDuringEvolution { get; set; } = false;
}
