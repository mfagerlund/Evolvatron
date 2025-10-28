namespace Evolvatron.Evolvion;

/// <summary>
/// Provides complexity-based mutation rate adjustments.
/// As networks grow in complexity (more nodes/edges), deletion rates increase
/// to balance growth and prevent unbounded expansion.
/// </summary>
public static class ComplexityBasedMutationRates
{
    /// <summary>
    /// Target complexity thresholds - networks should stabilize around these values.
    /// </summary>
    public class ComplexityTargets
    {
        /// <summary>Target number of active (connected) hidden nodes.</summary>
        public int TargetActiveHiddenNodes { get; set; } = 20;

        /// <summary>Target number of active edges.</summary>
        public int TargetActiveEdges { get; set; } = 50;

        /// <summary>Minimum active edges to maintain (prevent over-pruning).</summary>
        public int MinActiveEdges { get; set; } = 10;

        /// <summary>Base edge deletion probability when at target complexity.</summary>
        public float BaseEdgeDeleteProb { get; set; } = 0.02f;

        /// <summary>Maximum edge deletion probability (when well over target).</summary>
        public float MaxEdgeDeleteProb { get; set; } = 0.15f;

        /// <summary>Base edge addition probability when at target complexity.</summary>
        public float BaseEdgeAddProb { get; set; } = 0.05f;

        /// <summary>Minimum edge addition probability (when over target).</summary>
        public float MinEdgeAddProb { get; set; } = 0.01f;
    }

    /// <summary>
    /// Compute adjusted mutation rates based on current network complexity.
    /// Returns a modified EdgeMutationConfig with balanced add/delete rates.
    /// </summary>
    public static EdgeMutationConfig AdjustMutationRates(
        SpeciesSpec spec,
        EdgeMutationConfig baseConfig,
        ComplexityTargets targets)
    {
        // Compute current complexity
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);
        int activeHiddenCount = CountActiveHidden(spec, activeNodes);
        int activeEdgeCount = spec.Edges.Count;

        // Compute complexity ratios (1.0 = at target, >1.0 = over target, <1.0 = under target)
        float nodeComplexityRatio = (float)activeHiddenCount / targets.TargetActiveHiddenNodes;
        float edgeComplexityRatio = (float)activeEdgeCount / targets.TargetActiveEdges;

        // Use the maximum of the two ratios (most over-budget metric dominates)
        float complexityRatio = Math.Max(nodeComplexityRatio, edgeComplexityRatio);

        // Adjust deletion probability based on complexity
        // Linear scaling: at 1.0x target = base, at 2.0x target = max
        float deleteMultiplier = Math.Clamp(complexityRatio, 0.5f, 2.0f);
        float adjustedDeleteProb = baseConfig.EdgeDeleteRandom * deleteMultiplier;
        adjustedDeleteProb = Math.Clamp(
            adjustedDeleteProb,
            targets.BaseEdgeDeleteProb,
            targets.MaxEdgeDeleteProb);

        // Adjust addition probability inversely
        // When complexity is high, reduce additions
        float addMultiplier = Math.Clamp(2.0f - complexityRatio, 0.2f, 2.0f);
        float adjustedAddProb = baseConfig.EdgeAdd * addMultiplier;
        adjustedAddProb = Math.Clamp(
            adjustedAddProb,
            targets.MinEdgeAddProb,
            targets.BaseEdgeAddProb * 2);

        // If we're below minimum edges, force higher addition rate
        if (activeEdgeCount < targets.MinActiveEdges)
        {
            adjustedAddProb = Math.Max(adjustedAddProb, 0.2f);
            adjustedDeleteProb = 0.0f; // Disable deletions
        }

        // Create adjusted config
        return new EdgeMutationConfig
        {
            EdgeAdd = adjustedAddProb,
            EdgeDeleteRandom = adjustedDeleteProb,
            EdgeSplit = baseConfig.EdgeSplit,
            EdgeRedirect = baseConfig.EdgeRedirect,
            EdgeSwap = baseConfig.EdgeSwap,
            WeakEdgePruning = baseConfig.WeakEdgePruning
        };
    }

    /// <summary>
    /// Compute a single complexity score (0.0 - 2.0+) representing how far from target.
    /// 1.0 = at target, &lt;1.0 = under target, &gt;1.0 = over target.
    /// </summary>
    public static float ComputeComplexityScore(
        SpeciesSpec spec,
        ComplexityTargets targets)
    {
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);
        int activeHiddenCount = CountActiveHidden(spec, activeNodes);
        int activeEdgeCount = spec.Edges.Count;

        float nodeRatio = (float)activeHiddenCount / targets.TargetActiveHiddenNodes;
        float edgeRatio = (float)activeEdgeCount / targets.TargetActiveEdges;

        // Return weighted average (edges count more)
        return (nodeRatio * 0.3f) + (edgeRatio * 0.7f);
    }

    /// <summary>
    /// Get human-readable complexity status.
    /// </summary>
    public static string GetComplexityStatus(
        SpeciesSpec spec,
        ComplexityTargets targets)
    {
        float score = ComputeComplexityScore(spec, targets);

        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);
        int activeHiddenCount = CountActiveHidden(spec, activeNodes);
        int activeEdgeCount = spec.Edges.Count;

        string status = score switch
        {
            < 0.5f => "Very Sparse",
            < 0.8f => "Sparse",
            < 1.2f => "Balanced",
            < 1.5f => "Complex",
            _ => "Very Complex"
        };

        return $"{status} (Hidden: {activeHiddenCount}/{targets.TargetActiveHiddenNodes}, " +
               $"Edges: {activeEdgeCount}/{targets.TargetActiveEdges}, Score: {score:F2})";
    }

    /// <summary>
    /// Count active hidden nodes (excluding inputs and outputs).
    /// </summary>
    private static int CountActiveHidden(SpeciesSpec spec, bool[] activeNodes)
    {
        int count = 0;
        int nodeIdx = 0;

        for (int row = 0; row < spec.RowCounts.Length; row++)
        {
            // Skip input row (row 0) and output row (last row)
            bool isHidden = row > 0 && row < spec.RowCounts.Length - 1;

            for (int i = 0; i < spec.RowCounts[row]; i++)
            {
                if (isHidden && activeNodes[nodeIdx])
                {
                    count++;
                }
                nodeIdx++;
            }
        }

        return count;
    }

    /// <summary>
    /// Create default complexity targets for typical networks.
    /// </summary>
    public static ComplexityTargets DefaultTargets() => new ComplexityTargets();

    /// <summary>
    /// Create complexity targets for small/simple problems.
    /// </summary>
    public static ComplexityTargets SmallTargets() => new ComplexityTargets
    {
        TargetActiveHiddenNodes = 10,
        TargetActiveEdges = 25,
        MinActiveEdges = 5,
        BaseEdgeDeleteProb = 0.01f,
        MaxEdgeDeleteProb = 0.10f
    };

    /// <summary>
    /// Create complexity targets for large/complex problems.
    /// </summary>
    public static ComplexityTargets LargeTargets() => new ComplexityTargets
    {
        TargetActiveHiddenNodes = 40,
        TargetActiveEdges = 100,
        MinActiveEdges = 20,
        BaseEdgeDeleteProb = 0.03f,
        MaxEdgeDeleteProb = 0.20f
    };
}
