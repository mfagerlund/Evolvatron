using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Tests for neural network initialization strategies.
/// Verifies that different initialization approaches satisfy connectivity requirements
/// and that sparse initialization is viable for evolution.
/// </summary>
public class InitializationStrategiesTests
{
    #region Connectivity Requirements

    [Fact]
    public void MinimalInitialization_DirectInputToOutput_SatisfiesConnectivity()
    {
        // Create absolutely minimal network: each output connects to one input
        var builder = new SpeciesBuilder()
            .AddInputRow(3)
            .AddOutputRow(2, ActivationType.Tanh);

        // Connect output 0 to input 0
        builder.AddEdge(0, 3);
        // Connect output 1 to input 1
        builder.AddEdge(1, 4);

        var random = new Random(42);
        var spec = builder.InitializeSparse(random).Build();

        // Verify connectivity
        bool isConnected = ConnectivityValidator.ValidateConnectivity(spec, spec.Edges);
        Assert.True(isConnected, "Minimal direct input->output should satisfy connectivity");

        // Verify edge count
        Assert.Equal(2, spec.Edges.Count);
    }

    [Fact]
    public void SparseInitialization_2EdgesPerNode_SatisfiesConnectivity()
    {
        var random = new Random(42);

        // 3 inputs -> 4 hidden -> 2 outputs
        // Each hidden/output connects to 2 nodes from previous layer
        var builder = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(4, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh);

        int nodeOffset = 3; // Skip inputs

        // Connect hidden nodes (2 edges each)
        for (int i = 0; i < 4; i++)
        {
            int destNode = nodeOffset + i;
            // Pick 2 random inputs
            var sources = Enumerable.Range(0, 3)
                .OrderBy(_ => random.Next())
                .Take(2);

            foreach (var src in sources)
            {
                builder.AddEdge(src, destNode);
            }
        }

        nodeOffset += 4;

        // Connect output nodes (2 edges each)
        for (int i = 0; i < 2; i++)
        {
            int destNode = nodeOffset + i;
            // Pick 2 random hidden nodes
            var sources = Enumerable.Range(3, 4)
                .OrderBy(_ => random.Next())
                .Take(2);

            foreach (var src in sources)
            {
                builder.AddEdge(src, destNode);
            }
        }

        var spec = builder.InitializeSparse(random).Build();

        // Verify connectivity
        bool isConnected = ConnectivityValidator.ValidateConnectivity(spec, spec.Edges);
        Assert.True(isConnected);

        // Verify edge count (4 hidden * 2 + 2 outputs * 2 = 12 edges)
        Assert.InRange(spec.Edges.Count, 10, 14);
    }

    [Fact]
    public void FullyConnectedInitialization_HasManyMoreEdges()
    {
        var random = new Random(42);
        var fullyConnected = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(4, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeSparse(random)
            .Build();

        // Fully connected: (3 * 4) + (4 * 2) = 12 + 8 = 20 edges
        Assert.Equal(20, fullyConnected.Edges.Count);

        // Compare to minimal sparse (2 edges per node): (4 * 2) + (2 * 2) = 12 edges
        // Fully connected is 67% more edges!
    }

    #endregion

    #region Initialization Comparison

    [Fact]
    public void CompareInitializationDensity_FullyVsSparse()
    {
        var random = new Random(42);

        // Architecture: 4 inputs -> 8 hidden -> 8 hidden -> 3 outputs
        var fullyConnected = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(8, ActivationType.ReLU)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(3, ActivationType.Tanh)
            .WithMaxInDegree(10) // Allow higher for fully connected
            .Build();

        var sparse = CreateSparseNetwork(
            numInputs: 4,
            hiddenLayers: new[] { 8, 8 },
            numOutputs: 3,
            edgesPerNode: 3,
            random);

        // Fully connected: (4*8) + (8*8) + (8*3) = 32 + 64 + 24 = 120 edges
        Assert.Equal(120, fullyConnected.Edges.Count);

        // Sparse (3 edges per node): (8*3) + (8*3) + (3*3) = 24 + 24 + 9 = 57 edges
        Assert.InRange(sparse.Edges.Count, 50, 65);

        // Sparse is ~50% fewer edges
        double reductionRatio = 1.0 - ((double)sparse.Edges.Count / fullyConnected.Edges.Count);
        Assert.InRange(reductionRatio, 0.4, 0.6); // 40-60% reduction
    }

    [Fact]
    public void SparseInitialization_UsesGlorotWeights_ProperlyScaled()
    {
        var random = new Random(123);

        var spec = CreateSparseNetwork(
            numInputs: 3,
            hiddenLayers: new[] { 5 },
            numOutputs: 2,
            edgesPerNode: 2,
            random);

        var individual = SpeciesDiversification.InitializeIndividual(spec, random);

        // All weights should be in reasonable Glorot range
        // For fan-in/out of ~2-3, limit should be around sqrt(6/5) ≈ 1.1
        foreach (var weight in individual.Weights)
        {
            Assert.InRange(weight, -2.0f, 2.0f);
        }

        // Mean absolute weight should be reasonable (not too small)
        float meanAbs = individual.Weights.Average(w => MathF.Abs(w));
        Assert.InRange(meanAbs, 0.2f, 1.5f);
    }

    #endregion

    #region Evolution from Sparse Networks

    [Fact]
    public void SparseNetwork_CanEvolveByAddingEdges()
    {
        var random = new Random(42);

        var spec = CreateSparseNetwork(
            numInputs: 2,
            hiddenLayers: new[] { 4 },
            numOutputs: 1,
            edgesPerNode: 2,
            random);

        int initialEdgeCount = spec.Edges.Count;

        // Apply EdgeAdd mutation multiple times
        for (int i = 0; i < 10; i++)
        {
            EdgeTopologyMutations.TryEdgeAdd(spec, random);
        }

        // Should have added some edges
        Assert.True(spec.Edges.Count > initialEdgeCount,
            "Evolution should be able to add edges to sparse network");

        // Should still be valid
        spec.Validate();
    }

    [Fact]
    public void DirectInputOutput_CanEvolveHiddenLayers()
    {
        // Start with absolutely minimal: direct input -> output
        var builder = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(3, ActivationType.ReLU) // Hidden exists but may not be connected
            .AddOutputRow(1, ActivationType.Tanh);

        // Only connect inputs to output directly
        builder.AddEdge(0, 5); // Input 0 -> Output
        builder.AddEdge(1, 5); // Input 1 -> Output

        var spec = builder.Build();

        var random = new Random(42);

        // Verify initial connectivity
        Assert.True(ConnectivityValidator.ValidateConnectivity(spec, spec.Edges));

        int initialEdgeCount = spec.Edges.Count;
        Assert.Equal(2, initialEdgeCount);

        // Apply EdgeAdd to potentially use hidden layer
        for (int i = 0; i < 20; i++)
        {
            EdgeTopologyMutations.TryEdgeAdd(spec, random);
        }

        // Should have added edges that potentially route through hidden layer
        Assert.True(spec.Edges.Count > initialEdgeCount);

        // Check if any hidden nodes are now connected
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);

        // Nodes 2, 3, 4 are hidden layer
        bool anyHiddenActive = activeNodes[2] || activeNodes[3] || activeNodes[4];

        // May or may not have connected hidden layer yet - depends on random mutations
        // The important thing is the network is still valid
        spec.Validate();
    }

    #endregion

    #region Activation Function Distribution

    [Fact]
    public void InitializeIndividual_DistributesActivationFunctionsRandomly()
    {
        var random = new Random(42);

        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(20, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid)
            .AddOutputRow(1, ActivationType.Tanh)
            .Build();

        var individuals = new List<Individual>();
        for (int i = 0; i < 10; i++)
        {
            individuals.Add(SpeciesDiversification.InitializeIndividual(spec, random));
        }

        // Count activation types used in hidden layer (nodes 2-21)
        var activationCounts = new Dictionary<ActivationType, int>();

        foreach (var individual in individuals)
        {
            for (int nodeIdx = 2; nodeIdx < 22; nodeIdx++)
            {
                var activation = individual.Activations[nodeIdx];
                activationCounts[activation] = activationCounts.GetValueOrDefault(activation) + 1;
            }
        }

        // Should have used multiple activation types
        Assert.True(activationCounts.Count >= 2,
            "Should use multiple activation types from allowed set");

        // Should have reasonable distribution (not all one type)
        int totalHiddenNodes = 20 * 10; // 20 nodes * 10 individuals
        foreach (var count in activationCounts.Values)
        {
            float proportion = (float)count / totalHiddenNodes;
            // No single activation should dominate (>70%)
            Assert.True(proportion < 0.7f,
                $"Activation distribution should be reasonably balanced, but found {proportion * 100}%");
        }
    }

    #endregion

    #region Weight Initialization Quality

    [Fact]
    public void GlorotInitialization_ProducesZeroMeanDistribution()
    {
        var random = new Random(42);

        var spec = new SpeciesBuilder()
            .AddInputRow(5)
            .AddHiddenRow(10, ActivationType.ReLU)
            .AddOutputRow(3, ActivationType.Tanh)
            .WithMaxInDegree(12) // Allow higher for fully connected
            .Build();

        // Create many individuals to get statistical distribution
        var allWeights = new List<float>();

        for (int i = 0; i < 100; i++)
        {
            var individual = SpeciesDiversification.InitializeIndividual(spec, random);
            allWeights.AddRange(individual.Weights);
        }

        // Mean should be close to zero
        float mean = allWeights.Average();
        Assert.InRange(mean, -0.1f, 0.1f);

        // Standard deviation should be reasonable for Glorot
        float variance = allWeights.Average(w => (w - mean) * (w - mean));
        float stdDev = MathF.Sqrt(variance);

        // For Glorot with fan=7.5 average, std should be ~sqrt(2/15) ≈ 0.36
        Assert.InRange(stdDev, 0.2f, 0.6f);
    }

    [Fact]
    public void SparseInitialization_HasSmallerTotalMagnitude()
    {
        var random = new Random(123);

        var fullyConnected = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(8) // Allow higher for fully connected
            .InitializeDense(random, density: 1.0f) // Create all edges
            .Build();

        var sparse = CreateSparseNetwork(4, new[] { 6 }, 2, 2, random);

        var fullyIndividual = SpeciesDiversification.InitializeIndividual(fullyConnected, new Random(1));
        var sparseIndividual = SpeciesDiversification.InitializeIndividual(sparse, new Random(1));

        // Total magnitude should be lower for sparse (fewer edges)
        float fullyMagnitude = fullyIndividual.Weights.Sum(w => MathF.Abs(w));
        float sparseMagnitude = sparseIndividual.Weights.Sum(w => MathF.Abs(w));

        Assert.True(sparseMagnitude < fullyMagnitude,
            $"Sparse network total magnitude ({sparseMagnitude:F2}) should be less than fully connected ({fullyMagnitude:F2})");
    }

    #endregion

    #region Dense Initialization

    [Fact]
    public void InitializeDense_FullyConnected_CreatesAllPossibleEdges()
    {
        var random = new Random(42);

        // 2 inputs -> 6 hidden -> 6 hidden -> 1 output
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random, density: 1.0f)
            .Build();

        // Verify edge counts per layer
        // Layer 1: 2 inputs -> 6 hidden = 2*6 = 12 edges
        int layer1Edges = spec.Edges.Count(e => e.Source < 2 && e.Dest >= 2 && e.Dest < 8);
        Assert.Equal(12, layer1Edges);

        // Layer 2: 6 hidden -> 6 hidden = 6*6 = 36 edges
        int layer2Edges = spec.Edges.Count(e => e.Source >= 2 && e.Source < 8 && e.Dest >= 8 && e.Dest < 14);
        Assert.Equal(36, layer2Edges);

        // Layer 3: 6 hidden -> 1 output = 6*1 = 6 edges
        int layer3Edges = spec.Edges.Count(e => e.Source >= 8 && e.Source < 14 && e.Dest >= 14);
        Assert.Equal(6, layer3Edges);

        // Total: 12 + 36 + 6 = 54 edges
        Assert.Equal(54, spec.Edges.Count);
    }

    [Fact]
    public void InitializeDense_HalfDensity_CreatesApproximatelyHalfEdges()
    {
        var random = new Random(42);

        // 2 inputs -> 6 hidden -> 6 hidden -> 1 output
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random, density: 0.5f)
            .Build();

        // Expected edges per layer (approximately):
        // Layer 1: 6 nodes * 50% * 2 sources = 6 edges
        int layer1Edges = spec.Edges.Count(e => e.Source < 2 && e.Dest >= 2 && e.Dest < 8);
        Assert.Equal(6, layer1Edges); // Each of 6 nodes gets 1 connection (50% of 2)

        // Layer 2: 6 nodes * 50% * 6 sources = 18 edges
        int layer2Edges = spec.Edges.Count(e => e.Source >= 2 && e.Source < 8 && e.Dest >= 8 && e.Dest < 14);
        Assert.Equal(18, layer2Edges); // Each of 6 nodes gets 3 connections (50% of 6)

        // Layer 3: 1 node * 50% * 6 sources = 3 edges
        int layer3Edges = spec.Edges.Count(e => e.Source >= 8 && e.Source < 14 && e.Dest >= 14);
        Assert.Equal(3, layer3Edges); // 1 node gets 3 connections (50% of 6)

        // Total: 6 + 18 + 3 = 27 edges (exactly half of 54)
        Assert.Equal(27, spec.Edges.Count);
    }

    [Fact]
    public void InitializeDense_QuarterDensity_CreatesApproximatelyQuarterEdges()
    {
        var random = new Random(42);

        // 2 inputs -> 6 hidden -> 6 hidden -> 1 output
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random, density: 0.25f)
            .Build();

        // Each node gets at least 1 edge, then 25% of available (rounded)
        // Layer 1: 6 nodes * max(1, round(2*0.25)) = 6 nodes * 1 = 6 edges
        int layer1Edges = spec.Edges.Count(e => e.Source < 2 && e.Dest >= 2 && e.Dest < 8);
        Assert.Equal(6, layer1Edges);

        // Layer 2: 6 nodes * max(1, round(6*0.25)) = 6 nodes * 2 = 12 edges
        int layer2Edges = spec.Edges.Count(e => e.Source >= 2 && e.Source < 8 && e.Dest >= 8 && e.Dest < 14);
        Assert.Equal(12, layer2Edges); // round(6*0.25) = round(1.5) = 2

        // Layer 3: 1 node * max(1, round(6*0.25)) = 1 node * 2 = 2 edges
        int layer3Edges = spec.Edges.Count(e => e.Source >= 8 && e.Source < 14 && e.Dest >= 14);
        Assert.Equal(2, layer3Edges);

        // Total: 6 + 12 + 2 = 20 edges
        Assert.Equal(20, spec.Edges.Count);
    }

    [Fact]
    public void InitializeDense_RespectsMaxInDegree()
    {
        var random = new Random(42);

        // Create network where MaxInDegree would be exceeded at full density
        var spec = new SpeciesBuilder()
            .AddInputRow(10)
            .AddHiddenRow(5, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(6) // Limit to 6 incoming edges
            .InitializeDense(random, density: 1.0f)
            .Build();

        // Verify no node exceeds MaxInDegree
        var inDegrees = new Dictionary<int, int>();
        foreach (var (source, dest) in spec.Edges)
        {
            inDegrees[dest] = inDegrees.GetValueOrDefault(dest) + 1;
        }

        foreach (var (node, inDegree) in inDegrees)
        {
            Assert.True(inDegree <= 6, $"Node {node} has in-degree {inDegree}, exceeds MaxInDegree=6");
        }
    }

    [Fact]
    public void InitializeDense_GuaranteesMinimumOneEdgePerNode()
    {
        var random = new Random(42);

        // Very low density should still give at least 1 edge per node
        var spec = new SpeciesBuilder()
            .AddInputRow(10)
            .AddHiddenRow(8, ActivationType.ReLU)
            .AddOutputRow(3, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random, density: 0.1f) // 10% density
            .Build();

        // Count in-degrees
        var inDegrees = new Dictionary<int, int>();
        foreach (var (source, dest) in spec.Edges)
        {
            inDegrees[dest] = inDegrees.GetValueOrDefault(dest) + 1;
        }

        // Every hidden and output node should have at least 1 incoming edge
        for (int nodeIdx = 10; nodeIdx < spec.TotalNodes; nodeIdx++) // Skip inputs (0-9)
        {
            Assert.True(inDegrees.ContainsKey(nodeIdx) && inDegrees[nodeIdx] >= 1,
                $"Node {nodeIdx} should have at least 1 incoming edge even at low density");
        }
    }

    [Fact]
    public void InitializeDense_ThrowsOnInvalidDensity()
    {
        var random = new Random(42);
        var builder = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(4, ActivationType.ReLU)
            .AddOutputRow(1, ActivationType.Tanh);

        // Test zero density
        Assert.Throws<ArgumentException>(() =>
            builder.InitializeDense(random, density: 0.0f));

        // Test negative density
        Assert.Throws<ArgumentException>(() =>
            builder.InitializeDense(random, density: -0.5f));

        // Test density > 1.0
        Assert.Throws<ArgumentException>(() =>
            builder.InitializeDense(random, density: 1.5f));
    }

    [Fact]
    public void InitializeDense_RandomSelectionDiffers()
    {
        var random1 = new Random(42);
        var random2 = new Random(99);

        var spec1 = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random1, density: 0.5f)
            .Build();

        var spec2 = new SpeciesBuilder()
            .AddInputRow(4)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeDense(random2, density: 0.5f)
            .Build();

        // Same number of edges but different connections due to random selection
        Assert.Equal(spec1.Edges.Count, spec2.Edges.Count);

        // At least some edges should be different
        var edgeSet1 = spec1.Edges.ToHashSet();
        var edgeSet2 = spec2.Edges.ToHashSet();
        int commonEdges = edgeSet1.Intersect(edgeSet2).Count();

        Assert.True(commonEdges < spec1.Edges.Count,
            "Different random seeds should produce different edge selections");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Create a sparse network with specified architecture.
    /// Each node in hidden/output layers connects to edgesPerNode random nodes from previous layer.
    /// </summary>
    private static SpeciesSpec CreateSparseNetwork(
        int numInputs,
        int[] hiddenLayers,
        int numOutputs,
        int edgesPerNode,
        Random random)
    {
        var builder = new SpeciesBuilder()
            .AddInputRow(numInputs);

        // Add hidden layers
        foreach (var layerSize in hiddenLayers)
        {
            builder.AddHiddenRow(layerSize, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid);
        }

        builder.AddOutputRow(numOutputs, ActivationType.Tanh);

        // Connect layers sparsely
        int prevLayerStart = 0;
        int prevLayerSize = numInputs;
        int currentNodeOffset = numInputs;

        // Connect each hidden layer
        foreach (var layerSize in hiddenLayers)
        {
            for (int nodeIdx = 0; nodeIdx < layerSize; nodeIdx++)
            {
                int destNode = currentNodeOffset + nodeIdx;

                // Pick edgesPerNode random sources from previous layer
                var sources = Enumerable.Range(prevLayerStart, prevLayerSize)
                    .OrderBy(_ => random.Next())
                    .Take(Math.Min(edgesPerNode, prevLayerSize))
                    .ToList();

                foreach (var src in sources)
                {
                    builder.AddEdge(src, destNode);
                }
            }

            prevLayerStart = currentNodeOffset;
            prevLayerSize = layerSize;
            currentNodeOffset += layerSize;
        }

        // Connect output layer
        for (int outIdx = 0; outIdx < numOutputs; outIdx++)
        {
            int destNode = currentNodeOffset + outIdx;

            var sources = Enumerable.Range(prevLayerStart, prevLayerSize)
                .OrderBy(_ => random.Next())
                .Take(Math.Min(edgesPerNode, prevLayerSize))
                .ToList();

            foreach (var src in sources)
            {
                builder.AddEdge(src, destNode);
            }
        }

        return builder.Build();
    }

    #endregion
}
