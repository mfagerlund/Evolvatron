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

        // If topologies are identical (no size changes), just deep copy individuals
        if (TopologiesCompatible(parentTopology, newTopology))
        {
            // Simple case: clone parents and mutate
            int parentCount = parentSpecies.Individuals.Count;
            for (int i = 0; i < targetPopSize; i++)
            {
                var parent = parentSpecies.Individuals[i % parentCount];
                var child = new Individual(parent); // Deep copy
                individuals.Add(child);
            }
        }
        else
        {
            // Complex case: topology changed (nodes added/removed)
            // Create new individuals and try to preserve matching weights
            for (int i = 0; i < targetPopSize; i++)
            {
                var parent = parentSpecies.Individuals[i % parentSpecies.Individuals.Count];
                var child = AdaptIndividualToTopology(parent, parentTopology, newTopology, random);
                individuals.Add(child);
            }
        }

        return individuals;
    }

    /// <summary>
    /// Check if two topologies are structurally compatible (same node/edge counts).
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

        return oldTopo.Edges.Count == newTopo.Edges.Count;
    }

    /// <summary>
    /// Adapt an individual from old topology to new topology.
    /// Preserves weights for matching edges, initializes new edges with Glorot.
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
            Fitness = 0f,
            Age = 0
        };

        // Try to match edges between old and new topology
        for (int newEdgeIdx = 0; newEdgeIdx < newTopology.Edges.Count; newEdgeIdx++)
        {
            var (newSrc, newDst) = newTopology.Edges[newEdgeIdx];

            // Look for matching edge in old topology
            int oldEdgeIdx = oldTopology.Edges.FindIndex(e => e.Source == newSrc && e.Dest == newDst);

            if (oldEdgeIdx >= 0 && oldEdgeIdx < parent.Weights.Length)
            {
                // Edge exists in parent - inherit weight
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

        // Copy activations for matching nodes
        int minNodes = Math.Min(newNodeCount, parent.Activations.Length);
        for (int i = 0; i < minNodes; i++)
        {
            child.Activations[i] = parent.Activations[i];
            for (int p = 0; p < 4; p++)
            {
                child.NodeParams[i * 4 + p] = parent.NodeParams[i * 4 + p];
            }
        }

        // Initialize new nodes (if topology grew)
        for (int i = minNodes; i < newNodeCount; i++)
        {
            int row = newTopology.GetRowForNode(i);
            uint allowedMask = newTopology.AllowedActivationsPerRow[row];
            child.Activations[i] = SelectRandomActivation(allowedMask, random);

            // Default params
            child.NodeParams[i * 4 + 0] = 0.01f; // alpha
            child.NodeParams[i * 4 + 1] = 1.0f;  // beta
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
        // Skip bias (row 0), input (row 1), and output (last row)
        for (int row = 2; row < topology.RowCounts.Length - 1; row++)
        {
            int delta = random.Next(-2, 3); // -2, -1, 0, 1, 2
            int newSize = topology.RowCounts[row] + delta;

            // Constrain to reasonable range [2, 16]
            newSize = Math.Clamp(newSize, 2, 16);

            topology.RowCounts[row] = newSize;
        }

        // Remove edges that now reference invalid node indices
        int totalNodes = topology.TotalNodes;
        topology.Edges.RemoveAll(e => e.Source >= totalNodes || e.Dest >= totalNodes);
    }

    /// <summary>
    /// Toggle 1-3 random activation functions in the allowed set.
    /// Ensures at least one activation remains enabled per row.
    /// </summary>
    private static void MutateAllowedActivations(SpeciesSpec topology, Random random)
    {
        int toggleCount = random.Next(1, 4); // 1, 2, or 3 toggles

        for (int i = 0; i < toggleCount; i++)
        {
            int row = random.Next(topology.AllowedActivationsPerRow.Length);
            int activation = random.Next(11); // 11 activation types

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

        return individual;
    }

    /// <summary>
    /// Select a random activation from the allowed bitmask.
    /// </summary>
    private static ActivationType SelectRandomActivation(uint allowedMask, Random random)
    {
        // Count enabled bits
        var allowed = new List<ActivationType>();
        for (int i = 0; i < 11; i++)
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
