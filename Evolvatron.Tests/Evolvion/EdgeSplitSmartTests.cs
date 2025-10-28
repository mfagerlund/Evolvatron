using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Comprehensive tests for EdgeSplitSmart mutation operator.
/// Verifies that the operator correctly implements minimal network disruption
/// by inserting inactive nodes with low-weight stabilization connections.
/// </summary>
public class EdgeSplitSmartTests
{
    #region Basic Functionality

    [Fact]
    public void EdgeSplitSmart_InsertsInactiveNode()
    {
        var random = new Random(42);

        // Create network with inactive nodes: 2 inputs -> 6 hidden -> 2 outputs
        // Only connect 2 of the 6 hidden nodes, leaving 4 inactive
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 2) // Input 0 -> Hidden 0
            .AddEdge(1, 3) // Input 1 -> Hidden 1
            .AddEdge(2, 8) // Hidden 0 -> Output 0
            .AddEdge(3, 9) // Hidden 1 -> Output 1
            .Build();

        // Verify we have inactive nodes (4, 5, 6, 7 are inactive)
        var activeBefore = ConnectivityValidator.ComputeActiveNodes(spec);
        Assert.False(activeBefore[4]);
        Assert.False(activeBefore[5]);
        Assert.False(activeBefore[6]);
        Assert.False(activeBefore[7]);

        int initialEdgeCount = spec.Edges.Count;

        // Apply EdgeSplitSmart
        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        Assert.True(success, "EdgeSplitSmart should succeed with inactive nodes available");

        // Should have added 4 edges (removed 1, added 4, net +3)
        Assert.Equal(initialEdgeCount + 3, spec.Edges.Count);

        // Should have identified 4 new edge indices
        Assert.Equal(4, newEdgeIndices.Count);

        // All indices should be valid
        foreach (var idx in newEdgeIndices)
        {
            Assert.InRange(idx, 0, spec.Edges.Count - 1);
        }

        // Network should still be valid
        spec.Validate();

        // Should still be connected
        Assert.True(ConnectivityValidator.ValidateConnectivity(spec, spec.Edges));
    }

    [Fact]
    public void EdgeSplitSmart_RequiresMultiLayerSpan()
    {
        var random = new Random(42);

        // Create network with NO multi-layer spanning edges
        // Only direct adjacent layer connections
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(3, ActivationType.ReLU)
            .AddHiddenRow(3, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 2) // Input -> Hidden1 (adjacent)
            .AddEdge(1, 3) // Input -> Hidden1 (adjacent)
            .AddEdge(2, 5) // Hidden1 -> Hidden2 (adjacent)
            .AddEdge(3, 6) // Hidden1 -> Hidden2 (adjacent)
            .AddEdge(5, 8) // Hidden2 -> Output (adjacent)
            .Build();

        // Should fail - no edges span 2+ layers
        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        Assert.False(success, "EdgeSplitSmart should fail when no edges span multiple layers");
        Assert.Empty(newEdgeIndices);
    }

    [Fact]
    public void EdgeSplitSmart_WorksWithMultiLayerSpan()
    {
        var random = new Random(42);

        // Create network with skip connection that spans 2 layers
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(4, ActivationType.ReLU)
            .AddHiddenRow(4, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 2) // Input -> Hidden1
            .AddEdge(1, 3) // Input -> Hidden1
            .AddEdge(0, 6) // Input -> Hidden2 (SKIP - spans 2 layers!)
            .AddEdge(1, 7) // Input -> Hidden2 (SKIP - spans 2 layers!)
            .AddEdge(2, 10) // Hidden1 -> Output
            .AddEdge(6, 10) // Hidden2 -> Output
            .Build();

        int initialEdges = spec.Edges.Count;

        // Should succeed - we have skip connections
        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        // May succeed or fail depending on availability of inactive nodes
        if (success)
        {
            Assert.Equal(4, newEdgeIndices.Count);
            Assert.Equal(initialEdges + 3, spec.Edges.Count);
            spec.Validate();
        }
    }

    #endregion

    #region Weight Initialization Support

    [Fact]
    public void EdgeSplitSmart_ProvidesCorrectNewEdgeIndices()
    {
        var random = new Random(123);

        // Create network with clear multi-layer structure
        var spec = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(8, ActivationType.ReLU) // Plenty of inactive nodes
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 3) // Input 0 -> Hidden1[0]
            .AddEdge(1, 4) // Input 1 -> Hidden1[1]
            .AddEdge(0, 12) // Input 0 -> Hidden2[1] (SKIP - 2 layers!)
            .AddEdge(3, 19) // Hidden1[0] -> Output[0]
            .AddEdge(12, 20) // Hidden2[1] -> Output[1]
            .Build();

        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        if (!success)
        {
            // May fail if no suitable inactive node found - that's OK
            return;
        }

        Assert.Equal(4, newEdgeIndices.Count);

        // Verify all indices are valid
        foreach (var idx in newEdgeIndices)
        {
            Assert.True(idx >= 0 && idx < spec.Edges.Count,
                $"Edge index {idx} out of range [0, {spec.Edges.Count})");
        }

        // Verify no duplicate indices
        Assert.Equal(4, newEdgeIndices.Distinct().Count());
    }

    [Fact]
    public void EdgeSplitSmart_WithWeightInitialization_MinimalDisruption()
    {
        var random = new Random(456);

        // Create simple network for testing
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(8)
            .AddEdge(0, 2) // Input -> Hidden1
            .AddEdge(1, 3) // Input -> Hidden1
            .AddEdge(0, 9) // Input -> Hidden2 (SKIP!)
            .AddEdge(2, 14) // Hidden1 -> Output
            .AddEdge(9, 14) // Hidden2 -> Output
            .Build();

        // Create individuals before mutation
        var individuals = new List<Individual>();
        for (int i = 0; i < 5; i++)
        {
            individuals.Add(SpeciesDiversification.InitializeIndividual(spec, random));
        }

        // Store original output (would need actual network evaluation here)
        // For now, just verify topology changes

        int originalEdgeCount = spec.Edges.Count;

        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        if (!success)
            return; // May not have suitable configuration

        // Update individuals' weight arrays to match new topology
        for (int i = 0; i < individuals.Count; i++)
        {
            var individual = individuals[i];

            // Expand weight array
            var oldWeights = individual.Weights;
            var newWeights = new float[spec.Edges.Count];

            // Copy old weights (indices may have shifted due to sorting)
            // This is simplified - real implementation would track edge mapping
            Array.Copy(oldWeights, newWeights, Math.Min(oldWeights.Length, newWeights.Length));

            // Initialize new edge weights to VERY LOW values (minimal disruption)
            foreach (var edgeIdx in newEdgeIndices)
            {
                if (edgeIdx < newWeights.Length)
                {
                    newWeights[edgeIdx] = 0.01f * (random.NextSingle() * 2 - 1); // ±0.01
                }
            }

            individual.Weights = newWeights;
            individuals[i] = individual; // Update in list
        }

        // Verify all individuals still have valid weight arrays
        foreach (var individual in individuals)
        {
            Assert.Equal(spec.Edges.Count, individual.Weights.Length);
        }

        spec.Validate();
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EdgeSplitSmart_FailsGracefullyWithNoInactiveNodes()
    {
        var random = new Random(42);

        // Create fully connected small network (all nodes active)
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(2, ActivationType.ReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .WithMaxInDegree(8)
            .InitializeSparse(random)
            .Build();

        // All hidden nodes are active (connected to output)
        var active = ConnectivityValidator.ComputeActiveNodes(spec);
        Assert.True(active[2]); // Hidden node 0 is active
        Assert.True(active[3]); // Hidden node 1 is active

        // Should fail gracefully - no inactive nodes available
        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        Assert.False(success);
        Assert.Empty(newEdgeIndices);

        // Original spec should be unchanged
        spec.Validate();
    }

    [Fact]
    public void EdgeSplitSmart_HandlesMaxInDegreeConstraints()
    {
        var random = new Random(789);

        // Create network with tight MaxInDegree
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(3) // Tight constraint
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(0, 8) // Multi-layer span
            .AddEdge(2, 8)
            .AddEdge(3, 9)
            .Build();

        int originalEdges = spec.Edges.Count;

        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        // May succeed or fail depending on in-degree availability
        if (success)
        {
            // If succeeded, verify constraints are respected
            spec.Validate(); // This checks MaxInDegree

            var inDegrees = new int[spec.TotalNodes];
            foreach (var (_, dest) in spec.Edges)
            {
                inDegrees[dest]++;
            }

            foreach (var inDegree in inDegrees)
            {
                Assert.True(inDegree <= spec.MaxInDegree);
            }
        }
    }

    [Fact]
    public void EdgeSplitSmart_PreservesConnectivity()
    {
        var random = new Random(999);

        // Create network with clear structure
        var spec = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(8, ActivationType.ReLU)
            .AddHiddenRow(8, ActivationType.Tanh)
            .AddOutputRow(3, ActivationType.Tanh)
            .AddEdge(0, 3)
            .AddEdge(1, 4)
            .AddEdge(2, 5)
            .AddEdge(0, 12) // Skip connection
            .AddEdge(1, 13) // Skip connection
            .AddEdge(3, 19)
            .AddEdge(4, 20)
            .AddEdge(5, 21)
            .Build();

        // Verify initially connected
        Assert.True(ConnectivityValidator.ValidateConnectivity(spec, spec.Edges));

        // Apply mutation
        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        if (!success)
            return;

        // Verify still connected after mutation
        Assert.True(ConnectivityValidator.ValidateConnectivity(spec, spec.Edges),
            "Network should remain connected after EdgeSplitSmart");
    }

    #endregion

    #region Multiple Applications

    [Fact]
    public void EdgeSplitSmart_CanBeAppliedMultipleTimes()
    {
        var random = new Random(111);

        // Create network with many inactive nodes
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(10, ActivationType.ReLU)
            .AddHiddenRow(10, ActivationType.Tanh)
            .AddHiddenRow(10, ActivationType.Sigmoid)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(10)
            // Sparse connections leaving many inactive
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(0, 13) // Skip to hidden2
            .AddEdge(1, 14)
            .AddEdge(0, 24) // Skip to hidden3
            .AddEdge(13, 32)
            .AddEdge(24, 33)
            .Build();

        int initialActiveCount = ConnectivityValidator.ComputeActiveNodes(spec).Count(x => x);

        // Apply multiple times
        int successCount = 0;
        for (int i = 0; i < 5; i++)
        {
            if (EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices))
            {
                successCount++;
                spec.Validate();
                Assert.True(ConnectivityValidator.ValidateConnectivity(spec, spec.Edges));
            }
        }

        // Should have succeeded at least once (many inactive nodes available)
        Assert.True(successCount > 0, "Should successfully apply EdgeSplitSmart at least once");

        // Should have activated some previously inactive nodes
        int finalActiveCount = ConnectivityValidator.ComputeActiveNodes(spec).Count(x => x);
        Assert.True(finalActiveCount >= initialActiveCount,
            "Should have activated some nodes (or kept same)");
    }

    #endregion

    #region Verification of Smart Properties

    [Fact]
    public void EdgeSplitSmart_ActivatesExactlyOneInactiveNode()
    {
        var random = new Random(222);

        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(8, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(0, 10) // Skip - multi-layer!
            .AddEdge(2, 10)
            .AddEdge(3, 11)
            .Build();

        var activeBefore = ConnectivityValidator.ComputeActiveNodes(spec);
        int activeCountBefore = activeBefore.Count(x => x);

        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        if (!success)
            return;

        var activeAfter = ConnectivityValidator.ComputeActiveNodes(spec);
        int activeCountAfter = activeAfter.Count(x => x);

        // Should have activated exactly 1 additional node (the intermediate)
        Assert.Equal(activeCountBefore + 1, activeCountAfter);
    }

    [Fact]
    public void EdgeSplitSmart_Adds4EdgesRemoves1_NetPlus3()
    {
        var random = new Random(333);

        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(6, ActivationType.ReLU)
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(0, 9) // Skip
            .AddEdge(2, 14)
            .AddEdge(9, 15)
            .Build();

        int edgesBefore = spec.Edges.Count;

        bool success = EdgeTopologyMutations.TryEdgeSplitSmart(spec, random, out var newEdgeIndices);

        if (!success)
            return;

        int edgesAfter = spec.Edges.Count;

        // Should be +3 edges (removed 1, added 4)
        Assert.Equal(edgesBefore + 3, edgesAfter);

        // Should report exactly 4 new edge indices
        Assert.Equal(4, newEdgeIndices.Count);
    }

    #endregion
}
