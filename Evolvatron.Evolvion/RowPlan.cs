namespace Evolvatron.Evolvion;

/// <summary>
/// Compact metadata for synchronous row-by-row neural network evaluation.
/// Built during species creation and uploaded to GPU for kernel execution.
/// </summary>
public readonly struct RowPlan
{
    /// <summary>
    /// Global index of first node in this row
    /// </summary>
    public readonly int NodeStart;

    /// <summary>
    /// Number of nodes in this row
    /// </summary>
    public readonly int NodeCount;

    /// <summary>
    /// Global index of first edge targeting this row
    /// </summary>
    public readonly int EdgeStart;

    /// <summary>
    /// Number of edges targeting nodes in this row
    /// </summary>
    public readonly int EdgeCount;

    public RowPlan(int nodeStart, int nodeCount, int edgeStart, int edgeCount)
    {
        NodeStart = nodeStart;
        NodeCount = nodeCount;
        EdgeStart = edgeStart;
        EdgeCount = edgeCount;
    }

    public override string ToString()
    {
        return $"RowPlan(Nodes: {NodeStart}+{NodeCount}, Edges: {EdgeStart}+{EdgeCount})";
    }
}
