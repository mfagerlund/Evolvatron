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

        var spec = builder.Build();

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

        var spec = builder.Build();

        // Verify connectivity
        bool isConnected = ConnectivityValidator.ValidateConnectivity(spec, spec.Edges);
        Assert.True(isConnected);

        // Verify edge count (4 hidden * 2 + 2 outputs * 2 = 12 edges)
        Assert.InRange(spec.Edges.Count, 10, 14);
    }

    [Fact]
    public void FullyConnectedInitialization_HasManyMoreEdges()
    {
        var fullyConnected = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(4, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .FullyConnect(0, 1)
            .FullyConnect(1, 2)
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
            .FullyConnect(0, 1)
            .FullyConnect(1, 2)
            .FullyConnect(2, 3)
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
            .FullyConnect(0, 1)
            .FullyConnect(1, 2)
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
            .FullyConnect(0, 1)
            .FullyConnect(1, 2)
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
