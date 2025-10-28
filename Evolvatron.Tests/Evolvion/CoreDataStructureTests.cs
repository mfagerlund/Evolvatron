using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Tests for Individual, SpeciesSpec, and RowPlan core data structures
/// </summary>
public class CoreDataStructureTests
{
    #region Individual Tests

    [Fact]
    public void Individual_Constructor_InitializesArraysCorrectly()
    {
        int edgeCount = 20;
        int nodeCount = 10;

        var individual = new Individual(edgeCount, nodeCount);

        Assert.Equal(edgeCount, individual.Weights.Length);
        Assert.Equal(nodeCount * 4, individual.NodeParams.Length);
        Assert.Equal(nodeCount, individual.Activations.Length);
        Assert.Equal(nodeCount, individual.ActiveNodes.Length);
        Assert.Equal(0.0f, individual.Fitness);
        Assert.Equal(0, individual.Age);
    }

    [Fact]
    public void Individual_CopyConstructor_CreatesDeepCopy()
    {
        var original = new Individual(5, 3);
        original.Weights[0] = 1.5f;
        original.NodeParams[0] = 2.5f;
        original.Activations[0] = ActivationType.ReLU;
        original.ActiveNodes[0] = true;
        original.Fitness = 100.0f;
        original.Age = 5;

        var copy = new Individual(original);

        // Verify values copied
        Assert.Equal(original.Fitness, copy.Fitness);
        Assert.Equal(original.Age, copy.Age);
        Assert.Equal(original.Weights[0], copy.Weights[0]);
        Assert.Equal(original.NodeParams[0], copy.NodeParams[0]);
        Assert.Equal(original.Activations[0], copy.Activations[0]);
        Assert.Equal(original.ActiveNodes[0], copy.ActiveNodes[0]);

        // Verify deep copy (modifying copy doesn't affect original)
        copy.Weights[0] = 999.0f;
        copy.NodeParams[0] = 888.0f;
        copy.Activations[0] = ActivationType.Tanh;
        copy.ActiveNodes[0] = false;

        Assert.Equal(1.5f, original.Weights[0]);
        Assert.Equal(2.5f, original.NodeParams[0]);
        Assert.Equal(ActivationType.ReLU, original.Activations[0]);
        Assert.True(original.ActiveNodes[0]);
    }

    [Fact]
    public void Individual_GetNodeParams_ReturnsCorrectSpan()
    {
        var individual = new Individual(5, 3);
        individual.NodeParams[0] = 1.0f;
        individual.NodeParams[1] = 2.0f;
        individual.NodeParams[2] = 3.0f;
        individual.NodeParams[3] = 4.0f;

        individual.NodeParams[4] = 5.0f;
        individual.NodeParams[5] = 6.0f;
        individual.NodeParams[6] = 7.0f;
        individual.NodeParams[7] = 8.0f;

        var node0Params = individual.GetNodeParams(0);
        Assert.Equal(4, node0Params.Length);
        Assert.Equal(1.0f, node0Params[0]);
        Assert.Equal(2.0f, node0Params[1]);
        Assert.Equal(3.0f, node0Params[2]);
        Assert.Equal(4.0f, node0Params[3]);

        var node1Params = individual.GetNodeParams(1);
        Assert.Equal(5.0f, node1Params[0]);
        Assert.Equal(6.0f, node1Params[1]);
    }

    [Fact]
    public void Individual_SetNodeParams_UpdatesCorrectly()
    {
        var individual = new Individual(5, 3);
        var newParams = new float[] { 10.0f, 20.0f, 30.0f, 40.0f };

        individual.SetNodeParams(1, newParams);

        Assert.Equal(10.0f, individual.NodeParams[4]);
        Assert.Equal(20.0f, individual.NodeParams[5]);
        Assert.Equal(30.0f, individual.NodeParams[6]);
        Assert.Equal(40.0f, individual.NodeParams[7]);
    }

    #endregion

    #region RowPlan Tests

    [Fact]
    public void RowPlan_Constructor_StoresValuesCorrectly()
    {
        var plan = new RowPlan(nodeStart: 10, nodeCount: 5, edgeStart: 20, edgeCount: 15);

        Assert.Equal(10, plan.NodeStart);
        Assert.Equal(5, plan.NodeCount);
        Assert.Equal(20, plan.EdgeStart);
        Assert.Equal(15, plan.EdgeCount);
    }

    [Fact]
    public void RowPlan_ToString_ProducesReadableOutput()
    {
        var plan = new RowPlan(10, 5, 20, 15);
        var str = plan.ToString();

        Assert.Contains("10", str);
        Assert.Contains("5", str);
        Assert.Contains("20", str);
        Assert.Contains("15", str);
    }

    #endregion

    #region SpeciesSpec Tests

    [Fact]
    public void SpeciesSpec_BasicProperties_CalculatedCorrectly()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 8, 6, 3 }, // bias, input, hidden, hidden, output
            MaxInDegree = 6
        };

        Assert.Equal(24, spec.TotalNodes); // 1+6+8+6+3
        Assert.Equal(5, spec.RowCount);
        Assert.Equal(0, spec.TotalEdges); // No edges added yet
    }

    [Fact]
    public void SpeciesSpec_GetRowForNode_ReturnsCorrectRow()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 8, 6, 3 }
        };

        // Row 0: node 0
        Assert.Equal(0, spec.GetRowForNode(0));

        // Row 1: nodes 1-6
        Assert.Equal(1, spec.GetRowForNode(1));
        Assert.Equal(1, spec.GetRowForNode(6));

        // Row 2: nodes 7-14
        Assert.Equal(2, spec.GetRowForNode(7));
        Assert.Equal(2, spec.GetRowForNode(14));

        // Row 3: nodes 15-20
        Assert.Equal(3, spec.GetRowForNode(15));
        Assert.Equal(3, spec.GetRowForNode(20));

        // Row 4: nodes 21-23
        Assert.Equal(4, spec.GetRowForNode(21));
        Assert.Equal(4, spec.GetRowForNode(23));
    }

    [Fact]
    public void SpeciesSpec_GetRowForNode_ThrowsForInvalidNode()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 8, 6, 3 }
        };

        Assert.Throws<ArgumentException>(() => spec.GetRowForNode(24));
        Assert.Throws<ArgumentException>(() => spec.GetRowForNode(100));
    }

    [Fact]
    public void SpeciesSpec_IsActivationAllowed_ChecksBitmaskCorrectly()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 3 },
            AllowedActivationsPerRow = new uint[]
            {
                0, // Row 0 (bias): no activations allowed
                (1u << (int)ActivationType.ReLU) | (1u << (int)ActivationType.Tanh), // Row 1: ReLU and Tanh
                (1u << (int)ActivationType.Linear) // Row 2: Linear only
            }
        };

        // Row 0
        Assert.False(spec.IsActivationAllowed(0, ActivationType.Linear));

        // Row 1
        Assert.True(spec.IsActivationAllowed(1, ActivationType.ReLU));
        Assert.True(spec.IsActivationAllowed(1, ActivationType.Tanh));
        Assert.False(spec.IsActivationAllowed(1, ActivationType.Linear));

        // Row 2
        Assert.True(spec.IsActivationAllowed(2, ActivationType.Linear));
        Assert.False(spec.IsActivationAllowed(2, ActivationType.Tanh));
    }

    [Fact]
    public void SpeciesSpec_Validate_AcceptsValidSpec()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 8, 3 },
            AllowedActivationsPerRow = new uint[]
            {
                0, // Bias
                0xFFFFFFFF, // Input: all activations
                0xFFFFFFFF, // Hidden: all activations
                (1u << (int)ActivationType.Linear) | (1u << (int)ActivationType.Tanh) // Output: Linear or Tanh only
            },
            MaxInDegree = 6,
            Edges = new List<(int, int)>
            {
                (0, 7), // Bias to hidden
                (1, 7), // Input to hidden
                (7, 15) // Hidden to output
            }
        };

        // Should not throw
        spec.Validate();
    }

    [Fact]
    public void SpeciesSpec_Validate_RejectsEmptyRowCounts()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = Array.Empty<int>(),
            AllowedActivationsPerRow = Array.Empty<uint>()
        };

        var ex = Assert.Throws<InvalidOperationException>(() => spec.Validate());
        Assert.Contains("empty", ex.Message.ToLower());
    }

    [Fact]
    public void SpeciesSpec_Validate_RejectsNegativeRowCounts()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, -3 },
            AllowedActivationsPerRow = new uint[] { 0, 0, 0 }
        };

        var ex = Assert.Throws<InvalidOperationException>(() => spec.Validate());
        Assert.Contains("positive", ex.Message.ToLower());
    }

    [Fact]
    public void SpeciesSpec_Validate_RejectsInvalidOutputActivations()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 3 },
            AllowedActivationsPerRow = new uint[]
            {
                0,
                0xFFFFFFFF,
                (1u << (int)ActivationType.ReLU) // Invalid: output row allows ReLU
            }
        };

        var ex = Assert.Throws<InvalidOperationException>(() => spec.Validate());
        Assert.Contains("output", ex.Message.ToLower());
        Assert.Contains("linear", ex.Message.ToLower(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SpeciesSpec_Validate_RejectsNonAcyclicEdges()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 8, 3 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 0xFFFFFFFF, 3 },
            Edges = new List<(int, int)>
            {
                (7, 1) // Invalid: goes from row 2 back to row 1
            }
        };

        var ex = Assert.Throws<InvalidOperationException>(() => spec.Validate());
        Assert.Contains("acyclic", ex.Message.ToLower());
    }

    [Fact]
    public void SpeciesSpec_Validate_RejectsExcessiveInDegree()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 6, 8, 3 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 0xFFFFFFFF, 3 },
            MaxInDegree = 2,
            Edges = new List<(int, int)>
            {
                (0, 7),
                (1, 7),
                (2, 7) // Third edge to node 7, exceeds MaxInDegree=2
            }
        };

        var ex = Assert.Throws<InvalidOperationException>(() => spec.Validate());
        Assert.Contains("maxindegree", ex.Message.ToLower());
    }

    [Fact]
    public void SpeciesSpec_BuildRowPlans_CreatesCorrectPlans()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 2, 3, 2 }, // bias(1), input(2), hidden(3), output(2)
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 0xFFFFFFFF, 3 },
            Edges = new List<(int, int)>
            {
                // Edges to row 2 (nodes 3, 4, 5)
                (0, 3), // bias -> hidden[0]
                (1, 3), // input[0] -> hidden[0]
                (2, 4), // input[1] -> hidden[1]

                // Edges to row 3 (nodes 6, 7)
                (3, 6), // hidden[0] -> output[0]
                (4, 7), // hidden[1] -> output[1]
                (5, 7)  // hidden[2] -> output[1]
            }
        };

        spec.BuildRowPlans();

        Assert.Equal(4, spec.RowPlans.Length);

        // Row 0 (bias): 1 node, 0 incoming edges
        Assert.Equal(0, spec.RowPlans[0].NodeStart);
        Assert.Equal(1, spec.RowPlans[0].NodeCount);
        Assert.Equal(0, spec.RowPlans[0].EdgeCount);

        // Row 1 (input): 2 nodes, 0 incoming edges
        Assert.Equal(1, spec.RowPlans[1].NodeStart);
        Assert.Equal(2, spec.RowPlans[1].NodeCount);
        Assert.Equal(0, spec.RowPlans[1].EdgeCount);

        // Row 2 (hidden): 3 nodes, 3 incoming edges
        Assert.Equal(3, spec.RowPlans[2].NodeStart);
        Assert.Equal(3, spec.RowPlans[2].NodeCount);
        Assert.Equal(0, spec.RowPlans[2].EdgeStart);
        Assert.Equal(3, spec.RowPlans[2].EdgeCount);

        // Row 3 (output): 2 nodes, 3 incoming edges
        Assert.Equal(6, spec.RowPlans[3].NodeStart);
        Assert.Equal(2, spec.RowPlans[3].NodeCount);
        Assert.Equal(3, spec.RowPlans[3].EdgeStart);
        Assert.Equal(3, spec.RowPlans[3].EdgeCount);
    }

    [Fact]
    public void SpeciesSpec_BuildRowPlans_SortsEdgesByDestination()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 2, 3 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 3 },
            Edges = new List<(int, int)>
            {
                (1, 4), // Unsorted order
                (0, 3),
                (2, 5),
                (1, 3)
            }
        };

        spec.BuildRowPlans();

        // After sorting by (destRow, destNode), edges should be grouped by destination row
        // and sorted by destination node within each row
        var sortedEdges = spec.Edges;

        // All edges target row 2, so they should be sorted by destination node
        Assert.Equal(3, sortedEdges[0].Dest); // First two edges should target node 3
        Assert.Equal(3, sortedEdges[1].Dest);
        Assert.Equal(4, sortedEdges[2].Dest); // Then node 4
        Assert.Equal(5, sortedEdges[3].Dest); // Then node 5
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void FullSpeciesSetup_ValidatesAndBuildsCorrectly()
    {
        // Create a realistic species spec for XOR problem
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 2, 4, 1 }, // bias, 2 inputs, 4 hidden, 1 output
            AllowedActivationsPerRow = new uint[]
            {
                0, // Bias
                0xFFFFFFFF, // Input layer: any activation
                0xFFFFFFFF, // Hidden layer: any activation
                (1u << (int)ActivationType.Tanh) // Output: Tanh only
            },
            MaxInDegree = 6
        };

        // Add fully connected edges from bias+input to hidden
        for (int src = 0; src < 3; src++) // bias(0) + inputs(1,2)
        {
            for (int dst = 3; dst < 7; dst++) // hidden nodes
            {
                spec.Edges.Add((src, dst));
            }
        }

        // Add fully connected edges from hidden to output
        for (int src = 3; src < 7; src++) // hidden nodes
        {
            spec.Edges.Add((src, 7)); // to output node
        }

        // Validate
        spec.Validate();

        // Build row plans
        spec.BuildRowPlans();

        // Verify structure
        Assert.Equal(12 + 4, spec.TotalEdges); // 3*4 + 4*1 = 16
        Assert.Equal(8, spec.TotalNodes);
        Assert.Equal(4, spec.RowPlans.Length);

        // Create individual matching this spec
        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);

        // Initialize with random weights
        var random = new Random(42);
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(random.NextDouble() * 2 - 1);
        }

        // Set activations
        individual.Activations[0] = ActivationType.Linear; // Bias
        individual.Activations[1] = ActivationType.Linear; // Input
        individual.Activations[2] = ActivationType.Linear; // Input
        individual.Activations[3] = ActivationType.ReLU;   // Hidden
        individual.Activations[4] = ActivationType.ReLU;   // Hidden
        individual.Activations[5] = ActivationType.Tanh;   // Hidden
        individual.Activations[6] = ActivationType.Sigmoid; // Hidden
        individual.Activations[7] = ActivationType.Tanh;    // Output

        Assert.Equal(spec.TotalEdges, individual.Weights.Length);
        Assert.Equal(spec.TotalNodes, individual.Activations.Length);
    }

    #endregion
}
