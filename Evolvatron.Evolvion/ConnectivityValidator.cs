namespace Evolvatron.Evolvion;

/// <summary>
/// Validates network connectivity for topology mutations.
/// Ensures all output nodes remain reachable from input nodes.
/// </summary>
public static class ConnectivityValidator
{
    /// <summary>
    /// Checks if removing an edge would disconnect the graph
    /// </summary>
    public static bool CanDeleteEdge(SpeciesSpec spec, int src, int dst)
    {
        // Create temporary edge list without this edge
        var tempEdges = spec.Edges.Where(e => !(e.Source == src && e.Dest == dst)).ToList();

        return ValidateConnectivity(spec, tempEdges);
    }

    /// <summary>
    /// Validates that all output nodes are reachable from input layer
    /// </summary>
    public static bool ValidateConnectivity(SpeciesSpec spec, List<(int Source, int Dest)> edges)
    {
        // Get input and output node ranges
        var inputPlan = spec.RowPlans[0]; // Row 0 is input layer
        var outputPlan = spec.RowPlans[^1]; // Last row is output layer

        var inputNodes = Enumerable.Range(inputPlan.NodeStart, inputPlan.NodeCount).ToHashSet();
        var outputNodes = Enumerable.Range(outputPlan.NodeStart, outputPlan.NodeCount).ToHashSet();

        // Add bias node (always reachable)

        // Compute nodes reachable from input via forward BFS
        var reachableFromInput = ComputeReachableForward(edges, inputNodes, spec.TotalNodes);

        // Check that all output nodes are reachable
        foreach (var outputNode in outputNodes)
        {
            if (!reachableFromInput.Contains(outputNode))
                return false; // Deletion would disconnect graph
        }

        return true;
    }

    /// <summary>
    /// Computes all nodes reachable from source nodes via forward edges
    /// </summary>
    private static HashSet<int> ComputeReachableForward(
        List<(int Source, int Dest)> edges,
        HashSet<int> sourceNodes,
        int totalNodes)
    {
        var reachable = new HashSet<int>(sourceNodes);
        var queue = new Queue<int>(sourceNodes);

        // Build adjacency list
        var adjacency = new Dictionary<int, List<int>>();
        foreach (var (src, dst) in edges)
        {
            if (!adjacency.ContainsKey(src))
                adjacency[src] = new List<int>();
            adjacency[src].Add(dst);
        }

        // BFS
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();

            if (adjacency.TryGetValue(current, out var neighbors))
            {
                foreach (var neighbor in neighbors)
                {
                    if (reachable.Add(neighbor))
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }
        }

        return reachable;
    }

    /// <summary>
    /// Computes all nodes that can reach output nodes via backward edges
    /// </summary>
    public static HashSet<int> ComputeReachableBackward(
        List<(int Source, int Dest)> edges,
        HashSet<int> targetNodes,
        int totalNodes)
    {
        var reachable = new HashSet<int>(targetNodes);
        var queue = new Queue<int>(targetNodes);

        // Build reverse adjacency list
        var reverseAdjacency = new Dictionary<int, List<int>>();
        foreach (var (src, dst) in edges)
        {
            if (!reverseAdjacency.ContainsKey(dst))
                reverseAdjacency[dst] = new List<int>();
            reverseAdjacency[dst].Add(src);
        }

        // BFS backward
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();

            if (reverseAdjacency.TryGetValue(current, out var predecessors))
            {
                foreach (var predecessor in predecessors)
                {
                    if (reachable.Add(predecessor))
                    {
                        queue.Enqueue(predecessor);
                    }
                }
            }
        }

        return reachable;
    }

    /// <summary>
    /// Marks which nodes are active (on paths from input to output)
    /// </summary>
    public static bool[] ComputeActiveNodes(SpeciesSpec spec)
    {
        var inputPlan = spec.RowPlans[0];
        var outputPlan = spec.RowPlans[^1];

        var inputNodes = Enumerable.Range(inputPlan.NodeStart, inputPlan.NodeCount).ToHashSet();
        var outputNodes = Enumerable.Range(outputPlan.NodeStart, outputPlan.NodeCount).ToHashSet();

        // Nodes reachable from input
        var reachableFromInput = ComputeReachableForward(spec.Edges, inputNodes, spec.TotalNodes);

        // Nodes that reach output
        var reachesOutput = ComputeReachableBackward(spec.Edges, outputNodes, spec.TotalNodes);

        // A node is active if it's both reachable from input AND reaches output
        var activeNodes = new bool[spec.TotalNodes];
        for (int i = 0; i < spec.TotalNodes; i++)
        {
            activeNodes[i] = reachableFromInput.Contains(i) && reachesOutput.Contains(i);
        }

        return activeNodes;
    }
}
