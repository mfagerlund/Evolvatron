namespace Evolvatron.Evolvion;

/// <summary>
/// Defines the fixed topology and configuration of a species.
/// All individuals within a species share this structure but differ in weights and parameters.
/// </summary>
public class SpeciesSpec
{
    /// <summary>
    /// Number of nodes per row (includes bias row at index 0)
    /// Example: [1, 6, 8, 6, 3] = bias(1), input(6), hidden(8), hidden(6), output(3)
    /// </summary>
    public int[] RowCounts { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Bitmask of allowed activations per row.
    /// Bit i corresponds to ActivationType value i.
    /// </summary>
    public uint[] AllowedActivationsPerRow { get; set; } = Array.Empty<uint>();

    /// <summary>
    /// Maximum number of incoming edges per node
    /// </summary>
    public int MaxInDegree { get; set; } = 6;

    /// <summary>
    /// Edge list: each edge is (sourceNodeIndex, destNodeIndex)
    /// Edges must satisfy: row(source) < row(dest) for acyclic constraint
    /// </summary>
    public List<(int Source, int Dest)> Edges { get; set; } = new();

    /// <summary>
    /// Compiled row evaluation plans (generated from topology)
    /// </summary>
    public RowPlan[] RowPlans { get; set; } = Array.Empty<RowPlan>();

    /// <summary>
    /// Total number of nodes across all rows
    /// </summary>
    public int TotalNodes => RowCounts.Sum();

    /// <summary>
    /// Total number of edges in the topology
    /// </summary>
    public int TotalEdges => Edges.Count;

    /// <summary>
    /// Number of rows (including bias row at index 0)
    /// </summary>
    public int RowCount => RowCounts.Length;

    /// <summary>
    /// Gets the row index for a given node index
    /// </summary>
    public int GetRowForNode(int nodeIndex)
    {
        int cumulative = 0;
        for (int row = 0; row < RowCounts.Length; row++)
        {
            cumulative += RowCounts[row];
            if (nodeIndex < cumulative)
                return row;
        }
        throw new ArgumentException($"Node index {nodeIndex} out of range");
    }

    /// <summary>
    /// Checks if an activation type is allowed for a specific row
    /// </summary>
    public bool IsActivationAllowed(int row, ActivationType activation)
    {
        if (row < 0 || row >= AllowedActivationsPerRow.Length)
            return false;

        uint mask = AllowedActivationsPerRow[row];
        return (mask & (1u << (int)activation)) != 0;
    }

    /// <summary>
    /// Validates the species specification for correctness
    /// </summary>
    public void Validate()
    {
        if (RowCounts.Length == 0)
            throw new InvalidOperationException("RowCounts must not be empty");

        if (RowCounts[0] != 1)
            throw new InvalidOperationException("Row 0 (bias) must have exactly 1 node");

        if (RowCounts.Any(c => c <= 0))
            throw new InvalidOperationException("All row counts must be positive");

        if (AllowedActivationsPerRow.Length != RowCounts.Length)
            throw new InvalidOperationException("AllowedActivationsPerRow length must match RowCounts length");

        // Validate output row only allows Linear or Tanh
        int outputRow = RowCounts.Length - 1;
        uint outputMask = AllowedActivationsPerRow[outputRow];
        uint allowedOutputMask = (1u << (int)ActivationType.Linear) | (1u << (int)ActivationType.Tanh);
        if ((outputMask & ~allowedOutputMask) != 0)
            throw new InvalidOperationException("Output row may only use Linear or Tanh activations");

        // Validate edges
        foreach (var (source, dest) in Edges)
        {
            if (source < 0 || source >= TotalNodes)
                throw new InvalidOperationException($"Invalid source node {source}");
            if (dest < 0 || dest >= TotalNodes)
                throw new InvalidOperationException($"Invalid dest node {dest}");

            int sourceRow = GetRowForNode(source);
            int destRow = GetRowForNode(dest);

            if (sourceRow >= destRow)
                throw new InvalidOperationException($"Edge ({source}, {dest}) violates acyclic constraint: source row {sourceRow} >= dest row {destRow}");
        }

        // Check for excessive parallel edges (allow up to 2 parallel edges)
        var edgeGroups = Edges.GroupBy(e => e).Where(g => g.Count() > 2);
        if (edgeGroups.Any())
            throw new InvalidOperationException("More than 2 parallel edges detected between same nodes");

        // Validate in-degree constraint
        var inDegrees = new int[TotalNodes];
        foreach (var (_, dest) in Edges)
            inDegrees[dest]++;

        if (inDegrees.Any(d => d > MaxInDegree))
            throw new InvalidOperationException($"Some nodes exceed MaxInDegree={MaxInDegree}");
    }

    /// <summary>
    /// Builds RowPlans from the edge topology.
    /// Edges are sorted by (destRow, destNode) for coalesced GPU access.
    /// </summary>
    public void BuildRowPlans()
    {
        // Sort edges by destination for coalesced memory access
        var sortedEdges = Edges
            .Select(e => (Source: e.Source, Dest: e.Dest, DestRow: GetRowForNode(e.Dest)))
            .OrderBy(e => e.DestRow)
            .ThenBy(e => e.Dest)
            .ToList();

        Edges = sortedEdges.Select(e => (e.Source, e.Dest)).ToList();

        // Build row plans
        RowPlans = new RowPlan[RowCounts.Length];
        int nodeOffset = 0;
        int edgeOffset = 0;

        for (int row = 0; row < RowCounts.Length; row++)
        {
            int nodeCount = RowCounts[row];

            // Count edges targeting this row
            int edgeCount = 0;
            for (int i = edgeOffset; i < Edges.Count; i++)
            {
                if (GetRowForNode(Edges[i].Dest) == row)
                    edgeCount++;
                else
                    break;
            }

            RowPlans[row] = new RowPlan(nodeOffset, nodeCount, edgeOffset, edgeCount);

            nodeOffset += nodeCount;
            edgeOffset += edgeCount;
        }
    }
}
