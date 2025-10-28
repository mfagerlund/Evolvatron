using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Tests for complexity-based mutation rate balancing.
/// Verifies that deletion rates increase and addition rates decrease
/// as networks approach target complexity.
/// </summary>
public class ComplexityBasedMutationRatesTests
{
    #region Basic Complexity Measurement

    [Fact]
    public void ComputeComplexityScore_SparseNetwork_ReturnsLowScore()
    {
        // Create sparse network: 2 inputs -> 3 hidden -> 1 output with minimal edges
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(3, ActivationType.ReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 2) // Min connectivity
            .AddEdge(1, 3)
            .AddEdge(2, 5)
            .Build();

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        float score = ComplexityBasedMutationRates.ComputeComplexityScore(spec, targets);

        // With 2 active hidden nodes (target 20) and 3 edges (target 50), score should be << 1.0
        Assert.True(score < 0.5f, $"Sparse network should have score < 0.5, got {score}");
    }

    [Fact]
    public void ComputeComplexityScore_BalancedNetwork_ReturnsNearOne()
    {
        // Create network near target: ~20 active hidden, ~50 edges
        var spec = CreateNetworkWithComplexity(activeHidden: 18, edges: 45);

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        float score = ComplexityBasedMutationRates.ComputeComplexityScore(spec, targets);

        // Should be close to 1.0
        Assert.InRange(score, 0.8f, 1.2f);
    }

    [Fact]
    public void ComputeComplexityScore_OverComplexNetwork_ReturnsHighScore()
    {
        // Create network well over target: 40 hidden, 100 edges
        var spec = CreateNetworkWithComplexity(activeHidden: 40, edges: 100);

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        float score = ComplexityBasedMutationRates.ComputeComplexityScore(spec, targets);

        // Should be > 1.5
        Assert.True(score > 1.5f, $"Over-complex network should have score > 1.5, got {score}");
    }

    #endregion

    #region Mutation Rate Adjustment

    [Fact]
    public void AdjustMutationRates_SparseNetwork_IncreasesAddition_DecreasesDeletion()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(3, ActivationType.ReLU)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(2, 5)
            .Build();

        var baseConfig = new EdgeMutationConfig
        {
            EdgeAdd = 0.05f,
            EdgeDeleteRandom = 0.02f
        };

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        var adjusted = ComplexityBasedMutationRates.AdjustMutationRates(spec, baseConfig, targets);

        // For sparse network, addition should be boosted, deletion reduced
        Assert.True(adjusted.EdgeAdd >= baseConfig.EdgeAdd,
            $"Sparse network should have EdgeAdd >= base ({adjusted.EdgeAdd} vs {baseConfig.EdgeAdd})");
        Assert.True(adjusted.EdgeDeleteRandom <= baseConfig.EdgeDeleteRandom,
            $"Sparse network should have EdgeDelete <= base ({adjusted.EdgeDeleteRandom} vs {baseConfig.EdgeDeleteRandom})");
    }

    [Fact]
    public void AdjustMutationRates_OverComplexNetwork_DecreasesAddition_IncreasesDeletion()
    {
        // Network with 40 active hidden, 100 edges (2x over target)
        var spec = CreateNetworkWithComplexity(activeHidden: 40, edges: 100);

        var baseConfig = new EdgeMutationConfig
        {
            EdgeAdd = 0.05f,
            EdgeDeleteRandom = 0.02f
        };

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        var adjusted = ComplexityBasedMutationRates.AdjustMutationRates(spec, baseConfig, targets);

        // For over-complex network, deletion should be boosted, addition reduced
        Assert.True(adjusted.EdgeDeleteRandom > baseConfig.EdgeDeleteRandom,
            $"Over-complex network should have EdgeDelete > base ({adjusted.EdgeDeleteRandom} vs {baseConfig.EdgeDeleteRandom})");
        Assert.True(adjusted.EdgeAdd < baseConfig.EdgeAdd,
            $"Over-complex network should have EdgeAdd < base ({adjusted.EdgeAdd} vs {baseConfig.EdgeAdd})");
    }

    [Fact]
    public void AdjustMutationRates_BalancedNetwork_MaintainsSimilarRates()
    {
        var spec = CreateNetworkWithComplexity(activeHidden: 20, edges: 50);

        var baseConfig = new EdgeMutationConfig
        {
            EdgeAdd = 0.05f,
            EdgeDeleteRandom = 0.02f
        };

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        var adjusted = ComplexityBasedMutationRates.AdjustMutationRates(spec, baseConfig, targets);

        // For balanced network, rates should be close to base
        Assert.InRange(adjusted.EdgeAdd, baseConfig.EdgeAdd * 0.7f, baseConfig.EdgeAdd * 1.3f);
        Assert.InRange(adjusted.EdgeDeleteRandom, baseConfig.EdgeDeleteRandom * 0.7f, baseConfig.EdgeDeleteRandom * 1.3f);
    }

    [Fact]
    public void AdjustMutationRates_BelowMinEdges_DisablesDeletion()
    {
        // Very sparse network with only 5 edges (below min threshold of 10)
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(5, ActivationType.ReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(2, 7)
            .AddEdge(3, 8)
            .AddEdge(4, 8)
            .Build();

        var baseConfig = new EdgeMutationConfig
        {
            EdgeAdd = 0.05f,
            EdgeDeleteRandom = 0.02f
        };

        var targets = ComplexityBasedMutationRates.DefaultTargets();
        var adjusted = ComplexityBasedMutationRates.AdjustMutationRates(spec, baseConfig, targets);

        // Deletion should be disabled (0.0)
        Assert.Equal(0.0f, adjusted.EdgeDeleteRandom);

        // Addition should be boosted significantly
        Assert.True(adjusted.EdgeAdd >= 0.2f,
            $"Below min edges should boost addition to at least 0.2, got {adjusted.EdgeAdd}");
    }

    #endregion

    #region Complexity Status Reporting

    [Fact]
    public void GetComplexityStatus_ProvidesReadableOutput()
    {
        var spec = CreateNetworkWithComplexity(activeHidden: 15, edges: 40);
        var targets = ComplexityBasedMutationRates.DefaultTargets();

        string status = ComplexityBasedMutationRates.GetComplexityStatus(spec, targets);

        // Should contain key information
        Assert.Contains("Hidden:", status);
        Assert.Contains("Edges:", status);
        Assert.Contains("Score:", status);

        // Should have a status word
        Assert.True(
            status.Contains("Sparse") ||
            status.Contains("Balanced") ||
            status.Contains("Complex"),
            $"Status should contain complexity descriptor: {status}");
    }

    #endregion

    #region Target Presets

    [Fact]
    public void DefaultTargets_HasReasonableValues()
    {
        var targets = ComplexityBasedMutationRates.DefaultTargets();

        Assert.Equal(20, targets.TargetActiveHiddenNodes);
        Assert.Equal(50, targets.TargetActiveEdges);
        Assert.Equal(10, targets.MinActiveEdges);
        Assert.True(targets.BaseEdgeDeleteProb > 0);
        Assert.True(targets.MaxEdgeDeleteProb > targets.BaseEdgeDeleteProb);
    }

    [Fact]
    public void SmallTargets_HasLowerThresholds()
    {
        var small = ComplexityBasedMutationRates.SmallTargets();
        var def = ComplexityBasedMutationRates.DefaultTargets();

        Assert.True(small.TargetActiveHiddenNodes < def.TargetActiveHiddenNodes);
        Assert.True(small.TargetActiveEdges < def.TargetActiveEdges);
    }

    [Fact]
    public void LargeTargets_HasHigherThresholds()
    {
        var large = ComplexityBasedMutationRates.LargeTargets();
        var def = ComplexityBasedMutationRates.DefaultTargets();

        Assert.True(large.TargetActiveHiddenNodes > def.TargetActiveHiddenNodes);
        Assert.True(large.TargetActiveEdges > def.TargetActiveEdges);
    }

    #endregion

    #region Evolution Simulation

    [Fact]
    public void ComplexityBalancing_PreventsUnboundedGrowth()
    {
        var random = new Random(42);
        var spec = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh)
            .AddHiddenRow(8, ActivationType.Sigmoid)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeSparse(random)
            .Build();

        // Start sparse
        for (int i = 0; i < 10; i++)
        {
            spec.Edges.Add((i % 3, 3 + (i % 8)));
        }
        spec.BuildRowPlans();

        int initialEdges = spec.Edges.Count;

        var baseConfig = new EdgeMutationConfig
        {
            EdgeAdd = 0.1f,
            EdgeDeleteRandom = 0.01f
        };

        var targets = ComplexityBasedMutationRates.DefaultTargets();

        // Simulate many generations
        for (int gen = 0; gen < 100; gen++)
        {
            var adjusted = ComplexityBasedMutationRates.AdjustMutationRates(spec, baseConfig, targets);

            // Apply mutations
            if (random.NextSingle() < adjusted.EdgeAdd)
                EdgeTopologyMutations.TryEdgeAdd(spec, random);

            if (random.NextSingle() < adjusted.EdgeDeleteRandom)
                EdgeTopologyMutations.TryEdgeDeleteRandom(spec, random);
        }

        // Network should have grown but not excessively
        // With balancing, should stabilize around target (50 edges)
        Assert.InRange(spec.Edges.Count, initialEdges, targets.TargetActiveEdges * 2);

        // Should not have grown to absurd size (without balancing could reach 100+)
        Assert.True(spec.Edges.Count < 100,
            $"With complexity balancing, network should not exceed 100 edges, got {spec.Edges.Count}");
    }

    #endregion

    #region Helper Methods

    private SpeciesSpec CreateNetworkWithComplexity(int activeHidden, int edges)
    {
        // Calculate architecture to achieve target complexity
        // Use multiple hidden layers to have enough nodes
        int hiddenPerLayer = (int)Math.Ceiling(activeHidden / 2.0);

        var builder = new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(hiddenPerLayer, ActivationType.ReLU, ActivationType.Tanh)
            .AddHiddenRow(hiddenPerLayer, ActivationType.Sigmoid)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(Math.Max(10, edges / 5));

        var random = new Random(123);

        // Add edges to reach target count
        int nodeOffset = 3;
        int prevLayerStart = 0;
        int prevLayerSize = 3;

        // Connect hidden layer 1
        for (int i = 0; i < hiddenPerLayer && builder.Build().Edges.Count < edges; i++)
        {
            int dest = nodeOffset + i;
            int sources = Math.Min(3, edges / hiddenPerLayer);
            for (int s = 0; s < sources; s++)
            {
                int src = random.Next(prevLayerStart, prevLayerStart + prevLayerSize);
                builder.AddEdge(src, dest);
            }
        }

        prevLayerStart = nodeOffset;
        prevLayerSize = hiddenPerLayer;
        nodeOffset += hiddenPerLayer;

        // Connect hidden layer 2
        for (int i = 0; i < hiddenPerLayer; i++)
        {
            int dest = nodeOffset + i;
            int sources = Math.Min(4, edges / hiddenPerLayer);
            for (int s = 0; s < sources; s++)
            {
                int src = random.Next(prevLayerStart, prevLayerStart + prevLayerSize);
                builder.AddEdge(src, dest);
            }
        }

        prevLayerStart = nodeOffset;
        prevLayerSize = hiddenPerLayer;
        nodeOffset += hiddenPerLayer;

        // Connect outputs
        for (int i = 0; i < 2; i++)
        {
            int dest = nodeOffset + i;
            int sources = Math.Min(5, edges / 2);
            for (int s = 0; s < sources; s++)
            {
                int src = random.Next(prevLayerStart, prevLayerStart + prevLayerSize);
                builder.AddEdge(src, dest);
            }
        }

        return builder.InitializeSparse(random)
.Build();
    }

    #endregion
}
