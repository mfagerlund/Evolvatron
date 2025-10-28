namespace Evolvatron.Evolvion;

/// <summary>
/// Edge topology mutation operators.
/// These mutations modify the species topology (shared by all individuals).
/// </summary>
public static class EdgeTopologyMutations
{
    #region Core Mutations (from Evolvion.md)

    /// <summary>
    /// EdgeAdd: Add random upward edge if below in-degree cap
    /// </summary>
    public static bool TryEdgeAdd(SpeciesSpec spec, Random random)
    {
        // Pick random destination node (not in bias or input rows)
        if (spec.RowCounts.Length <= 2)
            return false; // No hidden or output rows

        int destRow = random.Next(2, spec.RowCounts.Length);
        var destPlan = spec.RowPlans[destRow];
        int destNode = destPlan.NodeStart + random.Next(destPlan.NodeCount);

        // Check current in-degree
        int currentInDegree = spec.Edges.Count(e => e.Dest == destNode);
        if (currentInDegree >= spec.MaxInDegree)
            return false;

        // Pick random source from earlier rows (acyclic constraint)
        var candidateSources = new List<int>();
        for (int row = 0; row < destRow; row++)
        {
            var plan = spec.RowPlans[row];
            for (int i = 0; i < plan.NodeCount; i++)
            {
                candidateSources.Add(plan.NodeStart + i);
            }
        }

        if (candidateSources.Count == 0)
            return false;

        // Try random sources until we find one that doesn't create duplicate edge
        var shuffled = candidateSources.OrderBy(_ => random.Next()).ToList();
        foreach (var srcNode in shuffled)
        {
            if (!spec.Edges.Contains((srcNode, destNode)))
            {
                spec.Edges.Add((srcNode, destNode));
                spec.BuildRowPlans(); // Rebuild plans with new edge
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// EdgeDelete: Remove random edge (if graph remains connected)
    /// </summary>
    public static bool TryEdgeDeleteRandom(SpeciesSpec spec, Random random)
    {
        if (spec.Edges.Count == 0)
            return false;

        // Try up to 10 random edges
        for (int attempt = 0; attempt < 10; attempt++)
        {
            var edge = spec.Edges[random.Next(spec.Edges.Count)];

            if (ConnectivityValidator.CanDeleteEdge(spec, edge.Source, edge.Dest))
            {
                spec.Edges.Remove(edge);
                spec.BuildRowPlans();
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// EdgeSplit: Insert intermediate node with two edges
    /// </summary>
    public static bool TryEdgeSplit(SpeciesSpec spec, Random random)
    {
        if (spec.Edges.Count == 0)
            return false;

        var edge = spec.Edges[random.Next(spec.Edges.Count)];
        int srcRow = spec.GetRowForNode(edge.Source);
        int dstRow = spec.GetRowForNode(edge.Dest);

        if (dstRow - srcRow <= 1)
            return false; // No intermediate row available

        // Pick intermediate row
        int intermediateRow = random.Next(srcRow + 1, dstRow);
        var intermediatePlan = spec.RowPlans[intermediateRow];

        // For now, add to existing row (creating new rows is complex)
        // Pick random node in intermediate row
        int newNode = intermediatePlan.NodeStart + random.Next(intermediatePlan.NodeCount);

        // Check in-degree constraints
        int newNodeInDegree = spec.Edges.Count(e => e.Dest == newNode);
        if (newNodeInDegree >= spec.MaxInDegree)
            return false;

        // Remove original edge
        spec.Edges.Remove(edge);

        // Add two new edges
        spec.Edges.Add((edge.Source, newNode));
        spec.Edges.Add((newNode, edge.Dest));

        spec.BuildRowPlans();
        return true;
    }

    /// <summary>
    /// EdgeSplitSmart: Insert intermediate INACTIVE node with minimal network disruption.
    ///
    /// Process:
    /// 1. Pick a link separated by at least one layer
    /// 2. Pick an INACTIVE node (not currently connected to output) between the layers
    /// 3. Remove the link and replace with two links through that node
    /// 4. Pick ANOTHER active node above intermediate, connect with very low weight
    /// 5. Pick ANOTHER active node below intermediate, connect with very low weight
    ///
    /// This operator gives us an almost unchanged network (except for the activator function)
    /// that can gradually mutate the new weights.
    ///
    /// NOTE: Weight initialization happens externally - this only modifies topology.
    /// The caller should initialize new edge weights to very low values (~0.01).
    /// </summary>
    /// <param name="spec">Species topology</param>
    /// <param name="random">Random generator</param>
    /// <param name="newEdgeIndices">Output: indices of the 4 new edges added (for weight initialization)</param>
    /// <returns>True if mutation succeeded</returns>
    public static bool TryEdgeSplitSmart(
        SpeciesSpec spec,
        Random random,
        out List<int> newEdgeIndices)
    {
        newEdgeIndices = new List<int>();

        if (spec.Edges.Count == 0)
            return false;

        // Compute active nodes (connected to outputs)
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);

        // Find edges that span at least 2 layers
        var candidateEdges = spec.Edges
            .Where(e =>
            {
                int srcRow = spec.GetRowForNode(e.Source);
                int dstRow = spec.GetRowForNode(e.Dest);
                return dstRow - srcRow >= 2; // At least one intermediate layer
            })
            .ToList();

        if (candidateEdges.Count == 0)
            return false; // No suitable edges

        // Try multiple times to find suitable edge + inactive node
        for (int attempt = 0; attempt < 10; attempt++)
        {
            var edge = candidateEdges[random.Next(candidateEdges.Count)];
            int srcRow = spec.GetRowForNode(edge.Source);
            int dstRow = spec.GetRowForNode(edge.Dest);

            // Pick intermediate row
            int intermediateRow = random.Next(srcRow + 1, dstRow);
            var intermediatePlan = spec.RowPlans[intermediateRow];

            // Find INACTIVE nodes in intermediate row
            var inactiveNodes = new List<int>();
            for (int i = 0; i < intermediatePlan.NodeCount; i++)
            {
                int nodeIdx = intermediatePlan.NodeStart + i;
                if (!activeNodes[nodeIdx])
                {
                    // Check in-degree constraint (will receive 2 edges)
                    int currentInDegree = spec.Edges.Count(e => e.Dest == nodeIdx);
                    if (currentInDegree + 2 <= spec.MaxInDegree)
                    {
                        inactiveNodes.Add(nodeIdx);
                    }
                }
            }

            if (inactiveNodes.Count == 0)
                continue; // Try different edge

            int intermediateNode = inactiveNodes[random.Next(inactiveNodes.Count)];

            // Find active nodes above intermediate (for additional input)
            var activeNodesAbove = new List<int>();
            for (int row = srcRow; row <= intermediateRow; row++)
            {
                var plan = spec.RowPlans[row];
                for (int i = 0; i < plan.NodeCount; i++)
                {
                    int nodeIdx = plan.NodeStart + i;
                    if (activeNodes[nodeIdx] && nodeIdx != edge.Source && nodeIdx != intermediateNode)
                    {
                        activeNodesAbove.Add(nodeIdx);
                    }
                }
            }

            // Find active nodes below intermediate (for additional output)
            var activeNodesBelow = new List<int>();
            for (int row = intermediateRow; row < spec.RowCounts.Length; row++)
            {
                var plan = spec.RowPlans[row];
                for (int i = 0; i < plan.NodeCount; i++)
                {
                    int nodeIdx = plan.NodeStart + i;
                    if (activeNodes[nodeIdx] && nodeIdx != edge.Dest && nodeIdx != intermediateNode)
                    {
                        // Check in-degree constraint
                        int currentInDegree = spec.Edges.Count(e => e.Dest == nodeIdx);
                        if (currentInDegree < spec.MaxInDegree)
                        {
                            activeNodesBelow.Add(nodeIdx);
                        }
                    }
                }
            }

            if (activeNodesAbove.Count == 0 || activeNodesBelow.Count == 0)
                continue; // Try different edge

            // SUCCESS - perform the mutation

            // Store original edge index for tracking
            int originalEdgeIdx = spec.Edges.IndexOf(edge);

            // Remove original edge
            spec.Edges.Remove(edge);

            // Add main split edges
            spec.Edges.Add((edge.Source, intermediateNode));
            spec.Edges.Add((intermediateNode, edge.Dest));

            // Add additional stabilization edges
            int additionalSourceAbove = activeNodesAbove[random.Next(activeNodesAbove.Count)];
            spec.Edges.Add((additionalSourceAbove, intermediateNode));

            int additionalDestBelow = activeNodesBelow[random.Next(activeNodesBelow.Count)];
            spec.Edges.Add((intermediateNode, additionalDestBelow));

            // Rebuild and sort edges
            spec.BuildRowPlans();

            // Find indices of new edges (after sorting)
            // These should be initialized with low weights externally
            newEdgeIndices.Add(spec.Edges.FindIndex(e => e == (edge.Source, intermediateNode)));
            newEdgeIndices.Add(spec.Edges.FindIndex(e => e == (intermediateNode, edge.Dest)));
            newEdgeIndices.Add(spec.Edges.FindIndex(e => e == (additionalSourceAbove, intermediateNode)));
            newEdgeIndices.Add(spec.Edges.FindIndex(e => e == (intermediateNode, additionalDestBelow)));

            return true;
        }

        return false; // Failed to find suitable configuration
    }

    #endregion

    #region Advanced Mutations

    /// <summary>
    /// EdgeRedirect: Change source or destination of edge
    /// </summary>
    public static bool TryEdgeRedirect(SpeciesSpec spec, Random random)
    {
        if (spec.Edges.Count == 0)
            return false;

        var edge = spec.Edges[random.Next(spec.Edges.Count)];
        bool redirectSource = random.Next(2) == 0;

        if (redirectSource)
        {
            // Change source to different node from earlier row
            int dstRow = spec.GetRowForNode(edge.Dest);
            var candidateSources = new List<int>();

            for (int row = 0; row < dstRow; row++)
            {
                var plan = spec.RowPlans[row];
                for (int i = 0; i < plan.NodeCount; i++)
                {
                    int candidate = plan.NodeStart + i;
                    if (candidate != edge.Source && !spec.Edges.Contains((candidate, edge.Dest)))
                    {
                        candidateSources.Add(candidate);
                    }
                }
            }

            if (candidateSources.Count == 0)
                return false;

            int newSource = candidateSources[random.Next(candidateSources.Count)];
            spec.Edges.Remove(edge);
            spec.Edges.Add((newSource, edge.Dest));
        }
        else
        {
            // Change destination to different node from later row
            int srcRow = spec.GetRowForNode(edge.Source);
            var candidateDests = new List<int>();

            for (int row = srcRow + 1; row < spec.RowCounts.Length; row++)
            {
                var plan = spec.RowPlans[row];
                for (int i = 0; i < plan.NodeCount; i++)
                {
                    int candidate = plan.NodeStart + i;
                    int candidateInDegree = spec.Edges.Count(e => e.Dest == candidate);

                    if (candidate != edge.Dest &&
                        !spec.Edges.Contains((edge.Source, candidate)) &&
                        candidateInDegree < spec.MaxInDegree)
                    {
                        candidateDests.Add(candidate);
                    }
                }
            }

            if (candidateDests.Count == 0)
                return false;

            int newDest = candidateDests[random.Next(candidateDests.Count)];
            spec.Edges.Remove(edge);
            spec.Edges.Add((edge.Source, newDest));
        }

        spec.BuildRowPlans();
        return true;
    }

    /// <summary>
    /// EdgeSwap: Swap destinations of two edges
    /// </summary>
    public static bool TryEdgeSwap(SpeciesSpec spec, Random random)
    {
        if (spec.Edges.Count < 2)
            return false;

        // Try multiple times to find swappable pair
        for (int attempt = 0; attempt < 20; attempt++)
        {
            var edge1 = spec.Edges[random.Next(spec.Edges.Count)];
            var edge2 = spec.Edges[random.Next(spec.Edges.Count)];

            if (edge1 == edge2)
                continue;

            int src1Row = spec.GetRowForNode(edge1.Source);
            int dst1Row = spec.GetRowForNode(edge1.Dest);
            int src2Row = spec.GetRowForNode(edge2.Source);
            int dst2Row = spec.GetRowForNode(edge2.Dest);

            // Verify swap preserves acyclic constraint
            if (src1Row >= dst2Row || src2Row >= dst1Row)
                continue;

            // Verify swapped edges don't already exist
            if (spec.Edges.Contains((edge1.Source, edge2.Dest)) ||
                spec.Edges.Contains((edge2.Source, edge1.Dest)))
                continue;

            // Verify in-degree constraints
            int dest2InDegree = spec.Edges.Count(e => e.Dest == edge2.Dest);
            int dest1InDegree = spec.Edges.Count(e => e.Dest == edge1.Dest);

            // After swap, each dest loses one edge and potentially gains one
            // Net change is zero, so we just need to check current state is under limit
            if (dest2InDegree > spec.MaxInDegree || dest1InDegree > spec.MaxInDegree)
                continue;

            // Perform swap
            spec.Edges.Remove(edge1);
            spec.Edges.Remove(edge2);
            spec.Edges.Add((edge1.Source, edge2.Dest));
            spec.Edges.Add((edge2.Source, edge1.Dest));

            spec.BuildRowPlans();
            return true;
        }

        return false;
    }

    #endregion

    #region Weak Edge Pruning

    /// <summary>
    /// Computes mean absolute weight for an edge across all individuals
    /// </summary>
    public static float ComputeMeanAbsWeight(List<Individual> individuals, (int Source, int Dest) edge, SpeciesSpec spec)
    {
        // Find edge index
        int edgeIndex = spec.Edges.IndexOf(edge);
        if (edgeIndex < 0)
            return 0.0f;

        float sum = 0.0f;
        foreach (var individual in individuals)
        {
            if (edgeIndex < individual.Weights.Length)
            {
                sum += MathF.Abs(individual.Weights[edgeIndex]);
            }
        }

        return individuals.Count > 0 ? sum / individuals.Count : 0.0f;
    }

    /// <summary>
    /// Prunes weak edges based on mean weight magnitudes
    /// </summary>
    public static int PruneWeakEdges(
        SpeciesSpec spec,
        List<Individual> individuals,
        WeakEdgePruningConfig config,
        Random random)
    {
        if (!config.Enabled)
            return 0;

        var weakEdges = new List<(int Source, int Dest, float MeanWeight, int Index)>();

        // Identify weak edges
        for (int i = 0; i < spec.Edges.Count; i++)
        {
            var edge = spec.Edges[i];
            float meanWeight = ComputeMeanAbsWeight(individuals, edge, spec);

            if (meanWeight < config.Threshold)
            {
                weakEdges.Add((edge.Source, edge.Dest, meanWeight, i));
            }
        }

        // Sort by mean weight (weakest first)
        weakEdges = weakEdges.OrderBy(e => e.MeanWeight).ToList();

        int prunedCount = 0;
        var edgesToDelete = new List<(int Source, int Dest, int Index)>();

        // Probabilistically select weak edges for deletion
        foreach (var (src, dst, weight, index) in weakEdges)
        {
            // Probability increases as weight approaches zero
            float deleteProb = 1.0f - (weight / config.Threshold);
            deleteProb = MathF.Min(deleteProb * config.BasePruneRate, 0.9f);

            if (random.NextSingle() < deleteProb)
            {
                if (ConnectivityValidator.CanDeleteEdge(spec, src, dst))
                {
                    edgesToDelete.Add((src, dst, index));
                }
            }
        }

        // Sort by index descending so we remove from end first (keeps earlier indices valid)
        edgesToDelete = edgesToDelete.OrderByDescending(e => e.Index).ToList();

        // Delete edges and corresponding weights at the same index
        foreach (var (src, dst, index) in edgesToDelete)
        {
            spec.Edges.RemoveAt(index);  // Remove at exact index, not by value search

            // Remove corresponding weights from all individuals at the same index
            for (int i = 0; i < individuals.Count; i++)
            {
                var individual = individuals[i];
                var newWeights = individual.Weights.ToList();
                newWeights.RemoveAt(index);
                individual.Weights = newWeights.ToArray();
                individuals[i] = individual;
            }

            prunedCount++;
        }

        if (prunedCount > 0)
        {
            spec.BuildRowPlans();
        }

        return prunedCount;
    }

    #endregion

    #region Mutation Applicator

    /// <summary>
    /// Applies all edge mutations based on configuration probabilities
    /// </summary>
    public static void ApplyEdgeMutations(
        SpeciesSpec spec,
        List<Individual> individuals,
        EdgeMutationConfig config,
        Random random)
    {
        // Core mutations
        if (random.NextSingle() < config.EdgeAdd)
            TryEdgeAdd(spec, random);

        if (random.NextSingle() < config.EdgeDeleteRandom)
            TryEdgeDeleteRandom(spec, random);

        if (random.NextSingle() < config.EdgeSplit)
            TryEdgeSplit(spec, random);

        // Advanced mutations
        if (random.NextSingle() < config.EdgeRedirect)
            TryEdgeRedirect(spec, random);

        if (random.NextSingle() < config.EdgeSwap)
            TryEdgeSwap(spec, random);

        // Weak edge pruning (if enabled during evolution)
        if (config.WeakEdgePruning.ApplyDuringEvolution)
        {
            PruneWeakEdges(spec, individuals, config.WeakEdgePruning, random);
        }
    }

    #endregion
}
