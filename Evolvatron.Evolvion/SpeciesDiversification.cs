namespace Evolvatron.Evolvion;

/// <summary>
/// Species diversification for replacing culled species.
/// Creates new species from top performers with topology mutations.
/// </summary>
public static class SpeciesDiversification
{
    /// <summary>
    /// Create a new diversified species from top-performing parent species.
    /// Process:
    /// 1. Select top-2 performing species
    /// 2. Clone topology from one parent
    /// 3. Apply diversification mutations (±nodes, activations, MaxInDegree)
    /// 4. Apply weak edge pruning at birth
    /// 5. Initialize individuals with Glorot/Xavier weights
    /// 6. Assign grace period protection
    /// </summary>
    /// <param name="population">Population to draw parents from.</param>
    /// <param name="config">Evolution configuration.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>New diversified species.</returns>
    public static Species CreateDiversifiedSpecies(
        Population population,
        EvolutionConfig config,
        Random random)
    {
        // Step 1: Select top-2 performing species
        var topSpecies = population.AllSpecies
            .OrderByDescending(s => s.Stats.MedianFitness)
            .Take(2)
            .ToList();

        if (topSpecies.Count == 0)
            throw new InvalidOperationException("Cannot create diversified species from empty population.");

        // Step 2: Clone topology from one parent
        var parentTopology = topSpecies[random.Next(Math.Min(2, topSpecies.Count))].Topology;
        var newTopology = CloneTopology(parentTopology);

        // Step 3: Apply diversification mutations
        ApplyDiversificationMutations(newTopology, config, random);

        // Step 3.5: Rebuild RowPlans and validate after mutations
        newTopology.BuildRowPlans();
        newTopology.Validate();

        // Step 4: Weak edge pruning is SKIPPED during diversification
        // The topology has been structurally changed (RowCounts mutated),
        // so parent individuals' weight arrays no longer match the topology.
        // Edge pruning will happen naturally during normal evolution instead.

        // Step 5: Initialize new individuals by inheriting from parent species
        // Clone individuals from best parent species and adapt to new topology
        var parentSpecies = topSpecies[0];
        var individuals = InheritPopulationFromParent(
            parentSpecies,
            newTopology,
            config.IndividualsPerSpecies,
            random);

        // Step 6: Create new species with grace period protection
        return new Species(newTopology)
        {
            Individuals = individuals,
            Age = 0, // Grace period protection
            Stats = new SpeciesStats()
        };
    }

    /// <summary>
    /// Inherit population from parent species, adapting to new topology.
    /// Preserves weights where topology matches, initializes new weights with Glorot.
    /// </summary>
    private static List<Individual> InheritPopulationFromParent(
        Species parentSpecies,
        SpeciesSpec newTopology,
        int targetPopSize,
        Random random)
    {
        var parentTopology = parentSpecies.Topology;
        var individuals = new List<Individual>(targetPopSize);

        // Rank parents by fitness (best first) so we bias toward top performers
        var rankedParents = parentSpecies.Individuals
            .OrderByDescending(ind => ind.Fitness)
            .ToList();

        bool compatible = TopologiesCompatible(parentTopology, newTopology);

        for (int i = 0; i < targetPopSize; i++)
        {
            // Bias toward top performers: pick from top half more often
            var parent = SelectWeightedParent(rankedParents, random);

            Individual child = compatible
                ? new Individual(parent)
                : AdaptIndividualToTopology(parent, parentTopology, newTopology, random);

            individuals.Add(child);
        }

        // Reconcile: inherited activations may violate new allowed masks
        foreach (var individual in individuals)
        {
            ReconcileActivations(individual, newTopology, random);
        }

        return individuals;
    }

    /// <summary>
    /// Check if two topologies are structurally identical (same rows and same edges).
    /// </summary>
    private static bool TopologiesCompatible(SpeciesSpec oldTopo, SpeciesSpec newTopo)
    {
        if (oldTopo.RowCounts.Length != newTopo.RowCounts.Length)
            return false;

        for (int i = 0; i < oldTopo.RowCounts.Length; i++)
        {
            if (oldTopo.RowCounts[i] != newTopo.RowCounts[i])
                return false;
        }

        if (oldTopo.Edges.Count != newTopo.Edges.Count)
            return false;

        // Check actual edge identity (after BuildRowPlans both are sorted)
        for (int i = 0; i < oldTopo.Edges.Count; i++)
        {
            if (oldTopo.Edges[i].Source != newTopo.Edges[i].Source ||
                oldTopo.Edges[i].Dest != newTopo.Edges[i].Dest)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Adapt an individual from old topology to new topology.
    /// Preserves weights for matching edges using semantic node remapping
    /// (accounts for node index shifts when row sizes change).
    /// </summary>
    private static Individual AdaptIndividualToTopology(
        Individual parent,
        SpeciesSpec oldTopology,
        SpeciesSpec newTopology,
        Random random)
    {
        int newNodeCount = newTopology.RowCounts.Sum();
        int newEdgeCount = newTopology.Edges.Count;

        var child = new Individual
        {
            Weights = new float[newEdgeCount],
            NodeParams = new float[newNodeCount * 4],
            Activations = new ActivationType[newNodeCount],
            Biases = new float[newNodeCount],
            Fitness = 0f,
            Age = 0
        };

        // Build node-index remapping: old global index → new global index
        // When rows change size, nodes in later rows shift. We match nodes
        // by (row, position-within-row) to preserve semantic identity.
        int numRows = Math.Min(oldTopology.RowCounts.Length, newTopology.RowCounts.Length);
        var oldToNewNode = new Dictionary<int, int>();
        int oldOffset = 0;
        int newOffset = 0;

        for (int row = 0; row < numRows; row++)
        {
            int oldRowSize = oldTopology.RowCounts[row];
            int newRowSize = newTopology.RowCounts[row];
            int commonNodes = Math.Min(oldRowSize, newRowSize);

            for (int j = 0; j < commonNodes; j++)
            {
                oldToNewNode[oldOffset + j] = newOffset + j;
            }

            oldOffset += oldRowSize;
            newOffset += newRowSize;
        }

        // Build remapped old edge lookup: (remappedSrc, remappedDst) → old weight index
        var oldEdgeLookup = new Dictionary<(int, int), int>();
        for (int i = 0; i < oldTopology.Edges.Count; i++)
        {
            var (oldSrc, oldDst) = oldTopology.Edges[i];
            if (oldToNewNode.TryGetValue(oldSrc, out int newSrc) &&
                oldToNewNode.TryGetValue(oldDst, out int newDst))
            {
                oldEdgeLookup[(newSrc, newDst)] = i;
            }
        }

        // Match new edges against remapped old edges
        for (int newEdgeIdx = 0; newEdgeIdx < newTopology.Edges.Count; newEdgeIdx++)
        {
            var (newSrc, newDst) = newTopology.Edges[newEdgeIdx];

            if (oldEdgeLookup.TryGetValue((newSrc, newDst), out int oldEdgeIdx) &&
                oldEdgeIdx < parent.Weights.Length)
            {
                child.Weights[newEdgeIdx] = parent.Weights[oldEdgeIdx];
            }
            else
            {
                // New edge - initialize with Glorot
                int fanIn = newTopology.Edges.Count(e => e.Dest == newDst);
                int fanOut = newTopology.Edges.Count(e => e.Source == newSrc);
                float limit = MathF.Sqrt(6f / (fanIn + fanOut));
                child.Weights[newEdgeIdx] = (random.NextSingle() * 2f - 1f) * limit;
            }
        }

        // Copy activations, biases, and node params using per-row semantic mapping
        oldOffset = 0;
        newOffset = 0;

        for (int row = 0; row < numRows; row++)
        {
            int oldRowSize = oldTopology.RowCounts[row];
            int newRowSize = newTopology.RowCounts[row];
            int commonNodes = Math.Min(oldRowSize, newRowSize);

            // Copy matching nodes within this row
            for (int j = 0; j < commonNodes; j++)
            {
                int oldIdx = oldOffset + j;
                int newIdx = newOffset + j;

                child.Activations[newIdx] = parent.Activations[oldIdx];

                for (int p = 0; p < 4; p++)
                    child.NodeParams[newIdx * 4 + p] = parent.NodeParams[oldIdx * 4 + p];

                if (parent.Biases != null && oldIdx < parent.Biases.Length)
                    child.Biases[newIdx] = parent.Biases[oldIdx];
                else
                    child.Biases[newIdx] = (random.NextSingle() * 2f - 1f) * 0.1f;
            }

            // Initialize new nodes in this row (if row grew)
            for (int j = commonNodes; j < newRowSize; j++)
            {
                int newIdx = newOffset + j;
                uint allowedMask = newTopology.AllowedActivationsPerRow[row];
                child.Activations[newIdx] = SelectRandomActivation(allowedMask, random);
                child.NodeParams[newIdx * 4 + 0] = 0.01f;
                child.NodeParams[newIdx * 4 + 1] = 1.0f;
                child.Biases[newIdx] = (random.NextSingle() * 2f - 1f) * 0.1f;
            }

            oldOffset += oldRowSize;
            newOffset += newRowSize;
        }

        // Initialize nodes from entirely new rows (if new topology has more rows)
        for (int i = newOffset; i < newNodeCount; i++)
        {
            int row = newTopology.GetRowForNode(i);
            uint allowedMask = newTopology.AllowedActivationsPerRow[row];
            child.Activations[i] = SelectRandomActivation(allowedMask, random);
            child.NodeParams[i * 4 + 0] = 0.01f;
            child.NodeParams[i * 4 + 1] = 1.0f;
            child.Biases[i] = (random.NextSingle() * 2f - 1f) * 0.1f;
        }

        return child;
    }

    /// <summary>
    /// Clone a species topology.
    /// Creates a deep copy of the topology structure.
    /// </summary>
    /// <param name="source">Source topology to clone.</param>
    /// <returns>Cloned topology.</returns>
    public static SpeciesSpec CloneTopology(SpeciesSpec source)
    {
        var cloned = new SpeciesSpec
        {
            RowCounts = (int[])source.RowCounts.Clone(),
            AllowedActivationsPerRow = (uint[])source.AllowedActivationsPerRow.Clone(),
            MaxInDegree = source.MaxInDegree,
            Edges = new List<(int, int)>(source.Edges)
        };

        // Rebuild RowPlans from cloned edge topology
        cloned.BuildRowPlans();

        return cloned;
    }

    /// <summary>
    /// Apply diversification mutations to topology:
    /// - ±1-2 nodes per hidden row (respecting min/max constraints)
    /// - Randomly toggle 1-3 allowed activations in bitmask
    /// - Adjust MaxInDegree by ±1 (range: 4-12)
    /// </summary>
    /// <param name="topology">Topology to mutate.</param>
    /// <param name="config">Evolution configuration.</param>
    /// <param name="random">Random number generator.</param>
    private static void ApplyDiversificationMutations(
        SpeciesSpec topology,
        EvolutionConfig config,
        Random random)
    {
        // Mutation 1: ±1-2 nodes per hidden row
        MutateHiddenLayerSizes(topology, random);

        // Mutation 2: Toggle 1-3 allowed activations
        MutateAllowedActivations(topology, random);

        // Mutation 3: Adjust MaxInDegree by ±1 (range: 4-12)
        MutateMaxInDegree(topology, random);
    }

    /// <summary>
    /// Mutate hidden layer sizes by ±1-2 nodes.
    /// Row 0 (bias), row 1 (inputs), and last row (outputs) are unchanged.
    /// </summary>
    private static void MutateHiddenLayerSizes(SpeciesSpec topology, Random random)
    {
        // Save old row counts for node-index remapping
        int[] oldRowCounts = (int[])topology.RowCounts.Clone();

        // Skip input (row 0) and output (last row) — only mutate hidden rows
        for (int row = 1; row < topology.RowCounts.Length - 1; row++)
        {
            int delta = random.Next(-2, 3); // -2, -1, 0, 1, 2
            int newSize = topology.RowCounts[row] + delta;

            // Constrain to reasonable range [2, 16]
            newSize = Math.Clamp(newSize, 2, 16);

            topology.RowCounts[row] = newSize;
        }

        // Remap edge indices to account for shifted node boundaries,
        // then remove edges that are now invalid (out-of-bounds, same-row, or backward)
        RemapAndPruneEdges(topology, oldRowCounts);
    }

    /// <summary>
    /// After row sizes change, remap edge (source, dest) indices to the new node
    /// index space and remove any edges that become invalid (out of bounds,
    /// same-row, or backward — violating acyclic constraint).
    /// </summary>
    private static void RemapAndPruneEdges(SpeciesSpec topology, int[] oldRowCounts)
    {
        int numRows = topology.RowCounts.Length;

        // Build old-node-index → (row, localIndex) mapping
        // Then (row, localIndex) → new-node-index mapping
        int[] oldRowStarts = new int[numRows];
        int[] newRowStarts = new int[numRows];
        for (int r = 1; r < numRows; r++)
        {
            oldRowStarts[r] = oldRowStarts[r - 1] + oldRowCounts[r - 1];
            newRowStarts[r] = newRowStarts[r - 1] + topology.RowCounts[r - 1];
        }

        int oldTotal = oldRowStarts[numRows - 1] + oldRowCounts[numRows - 1];
        int newTotal = topology.TotalNodes;

        // For each old node, compute its new index (or -1 if it was trimmed)
        var oldToNew = new int[oldTotal];
        Array.Fill(oldToNew, -1);

        for (int r = 0; r < numRows; r++)
        {
            int commonNodes = Math.Min(oldRowCounts[r], topology.RowCounts[r]);
            for (int j = 0; j < commonNodes; j++)
            {
                oldToNew[oldRowStarts[r] + j] = newRowStarts[r] + j;
            }
        }

        // Remap edges and filter out invalid ones
        var newEdges = new List<(int Source, int Dest)>(topology.Edges.Count);
        foreach (var (oldSrc, oldDst) in topology.Edges)
        {
            if (oldSrc >= oldTotal || oldDst >= oldTotal)
                continue;

            int newSrc = oldToNew[oldSrc];
            int newDst = oldToNew[oldDst];

            // Skip if either node was trimmed
            if (newSrc < 0 || newDst < 0)
                continue;

            // Verify acyclic constraint still holds (source row < dest row)
            int srcRow = topology.GetRowForNode(newSrc);
            int dstRow = topology.GetRowForNode(newDst);
            if (srcRow >= dstRow)
                continue;

            newEdges.Add((newSrc, newDst));
        }

        topology.Edges = newEdges;
    }

    /// <summary>
    /// Toggle 1-3 random activation functions in the allowed set.
    /// Ensures at least one activation remains enabled per row.
    /// </summary>
    private static void MutateAllowedActivations(SpeciesSpec topology, Random random)
    {
        int toggleCount = random.Next(1, 4); // 1, 2, or 3 toggles

        int activationCount = Enum.GetValues<ActivationType>().Length;

        // Only mutate hidden rows: skip input (row 0) and output (last row)
        int firstMutableRow = 1;
        int mutableRowCount = topology.AllowedActivationsPerRow.Length - 2; // exclude input + output
        if (mutableRowCount <= 0)
            return; // No hidden rows to mutate

        for (int i = 0; i < toggleCount; i++)
        {
            int row = firstMutableRow + random.Next(mutableRowCount);
            int activation = random.Next(activationCount);

            uint mask = 1u << activation;
            uint currentMask = topology.AllowedActivationsPerRow[row];

            // Toggle the bit
            uint newMask = currentMask ^ mask;

            // Ensure at least one activation remains enabled
            if (newMask != 0)
            {
                topology.AllowedActivationsPerRow[row] = newMask;
            }
        }
    }

    /// <summary>
    /// Adjust MaxInDegree by ±1, constrained to range [4, 12].
    /// </summary>
    private static void MutateMaxInDegree(SpeciesSpec topology, Random random)
    {
        int delta = random.Next(-1, 2); // -1, 0, 1
        int newMaxInDegree = topology.MaxInDegree + delta;

        topology.MaxInDegree = Math.Clamp(newMaxInDegree, 4, 12);

        // If MaxInDegree decreased, prune excess edges from saturated nodes
        PruneExcessInDegree(topology, random);
    }

    /// <summary>
    /// Remove random edges from nodes that exceed MaxInDegree.
    /// </summary>
    private static void PruneExcessInDegree(SpeciesSpec topology, Random random)
    {
        // Count in-degree per node
        var inDegree = new Dictionary<int, List<int>>(); // dest → list of edge indices
        for (int i = 0; i < topology.Edges.Count; i++)
        {
            int dest = topology.Edges[i].Dest;
            if (!inDegree.ContainsKey(dest))
                inDegree[dest] = new List<int>();
            inDegree[dest].Add(i);
        }

        var edgesToRemove = new HashSet<int>();
        foreach (var (dest, edgeIndices) in inDegree)
        {
            if (edgeIndices.Count > topology.MaxInDegree)
            {
                // Shuffle and remove excess
                for (int i = edgeIndices.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (edgeIndices[i], edgeIndices[j]) = (edgeIndices[j], edgeIndices[i]);
                }
                int excess = edgeIndices.Count - topology.MaxInDegree;
                for (int i = 0; i < excess; i++)
                {
                    edgesToRemove.Add(edgeIndices[i]);
                }
            }
        }

        if (edgesToRemove.Count > 0)
        {
            topology.Edges = topology.Edges
                .Where((_, idx) => !edgesToRemove.Contains(idx))
                .ToList();
        }
    }

    /// <summary>
    /// Initialize a population of individuals for a species.
    /// Uses Glorot/Xavier uniform weight initialization.
    /// </summary>
    /// <param name="topology">Species topology.</param>
    /// <param name="populationSize">Number of individuals to create.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>List of initialized individuals.</returns>
    public static List<Individual> InitializePopulation(
        SpeciesSpec topology,
        int populationSize,
        Random random)
    {
        var individuals = new List<Individual>(populationSize);

        for (int i = 0; i < populationSize; i++)
        {
            var individual = InitializeIndividual(topology, random);
            individuals.Add(individual);
        }

        return individuals;
    }

    /// <summary>
    /// Initialize a single individual with random weights and activations.
    /// </summary>
    /// <param name="topology">Species topology.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Initialized individual.</returns>
    public static Individual InitializeIndividual(
        SpeciesSpec topology,
        Random random)
    {
        int totalNodes = topology.RowCounts.Sum();
        int totalEdges = topology.Edges.Count;

        var individual = new Individual
        {
            Weights = new float[totalEdges],
            NodeParams = new float[totalNodes * 4], // 4 params per node
            Activations = new ActivationType[totalNodes],
            Biases = new float[totalNodes],
            Fitness = 0f,
            Age = 0
        };

        // Initialize weights using Glorot/Xavier uniform
        for (int i = 0; i < totalEdges; i++)
        {
            var (src, dst) = topology.Edges[i];

            // Compute fan-in and fan-out
            int fanIn = topology.Edges.Count(e => e.Dest == dst);
            int fanOut = topology.Edges.Count(e => e.Source == src);

            float limit = MathF.Sqrt(6f / (fanIn + fanOut));
            individual.Weights[i] = (random.NextSingle() * 2f - 1f) * limit;
        }

        // Initialize activations randomly from allowed set
        int nodeIdx = 0;
        for (int row = 0; row < topology.RowCounts.Length; row++)
        {
            uint allowedMask = topology.AllowedActivationsPerRow[row];

            for (int i = 0; i < topology.RowCounts[row]; i++)
            {
                individual.Activations[nodeIdx] = SelectRandomActivation(allowedMask, random);
                nodeIdx++;
            }
        }

        // Initialize node params to default values (will be set by activation requirements)
        for (int i = 0; i < totalNodes; i++)
        {
            // Default LeakyReLU alpha = 0.01, ELU alpha = 1.0, etc.
            individual.NodeParams[i * 4 + 0] = 0.01f; // alpha
            individual.NodeParams[i * 4 + 1] = 1.0f;  // beta
            individual.NodeParams[i * 4 + 2] = 0f;
            individual.NodeParams[i * 4 + 3] = 0f;
        }

        // Initialize biases with small random values
        for (int i = 0; i < totalNodes; i++)
        {
            individual.Biases[i] = (random.NextSingle() * 2f - 1f) * 0.1f;
        }

        return individual;
    }

    /// <summary>
    /// Select a parent from a fitness-ranked list, biased toward top performers.
    /// Uses linear rank weighting: rank 0 (best) gets weight N, rank N-1 gets weight 1.
    /// </summary>
    private static Individual SelectWeightedParent(List<Individual> rankedParents, Random random)
    {
        int n = rankedParents.Count;
        // Total weight = n + (n-1) + ... + 1 = n*(n+1)/2
        int totalWeight = n * (n + 1) / 2;
        int roll = random.Next(totalWeight);

        int cumulative = 0;
        for (int i = 0; i < n; i++)
        {
            cumulative += (n - i); // weight for rank i
            if (roll < cumulative)
                return rankedParents[i];
        }

        return rankedParents[0]; // fallback
    }

    /// <summary>
    /// Ensure all node activations comply with the species' allowed activation masks.
    /// Replaces any disallowed activation with a random allowed one.
    /// </summary>
    private static void ReconcileActivations(Individual individual, SpeciesSpec topology, Random random)
    {
        int nodeIdx = 0;
        for (int row = 0; row < topology.RowCounts.Length; row++)
        {
            uint allowedMask = topology.AllowedActivationsPerRow[row];
            for (int i = 0; i < topology.RowCounts[row]; i++)
            {
                var current = individual.Activations[nodeIdx];
                if ((allowedMask & (1u << (int)current)) == 0)
                {
                    var newActivation = SelectRandomActivation(allowedMask, random);
                    individual.Activations[nodeIdx] = newActivation;

                    // Reset node params to defaults for the new activation type
                    var defaults = newActivation.GetDefaultParameters();
                    individual.NodeParams[nodeIdx * 4 + 0] = defaults.Length > 0 ? defaults[0] : 0.01f;
                    individual.NodeParams[nodeIdx * 4 + 1] = defaults.Length > 1 ? defaults[1] : 1.0f;
                    individual.NodeParams[nodeIdx * 4 + 2] = 0f;
                    individual.NodeParams[nodeIdx * 4 + 3] = 0f;
                }
                nodeIdx++;
            }
        }
    }

    /// <summary>
    /// Select a random activation from the allowed bitmask.
    /// </summary>
    private static ActivationType SelectRandomActivation(uint allowedMask, Random random)
    {
        // Count enabled bits
        var allowed = new List<ActivationType>();
        int activationCount = Enum.GetValues<ActivationType>().Length;
        for (int i = 0; i < activationCount; i++)
        {
            if ((allowedMask & (1u << i)) != 0)
            {
                allowed.Add((ActivationType)i);
            }
        }

        if (allowed.Count == 0)
            return ActivationType.Linear; // Fallback

        return allowed[random.Next(allowed.Count)];
    }
}
