using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Comprehensive tests for CPU-based neural network evaluation with RowPlan execution
/// </summary>
public class CPUEvaluatorTests
{
    #region Basic Functionality Tests

    [Fact]
    public void CPUEvaluator_Constructor_Succeeds()
    {
        var spec = CreateSimpleSpec();
        var evaluator = new CPUEvaluator(spec);
        Assert.NotNull(evaluator);
    }

    [Fact]
    public void CPUEvaluator_Evaluate_ReturnsCorrectOutputSize()
    {
        var spec = CreateSimpleSpec(); // 1 bias, 2 inputs, 3 outputs
        spec.BuildRowPlans();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var evaluator = new CPUEvaluator(spec);

        var inputs = new float[] { 0.5f, 0.5f };
        var outputs = evaluator.Evaluate(individual, inputs);

        Assert.Equal(3, outputs.Length);
    }

    [Fact]
    public void CPUEvaluator_Evaluate_ThrowsOnInvalidInputSize()
    {
        var spec = CreateSimpleSpec(); // Expects 2 inputs
        spec.BuildRowPlans();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var evaluator = new CPUEvaluator(spec);

        var invalidInputs = new float[] { 0.5f }; // Wrong size

        Assert.Throws<ArgumentException>(() =>
            evaluator.Evaluate(individual, invalidInputs));
    }

    #endregion

    #region Pass-Through Network Tests

    [Fact]
    public void CPUEvaluator_PassThroughNetwork_ProducesCorrectOutput()
    {
        // Create simple pass-through: input -> output
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddOutputRow(2, ActivationType.Tanh)
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);

        // Set weights to 1.0 (identity)
        individual.Weights[0] = 1.0f;
        individual.Weights[1] = 1.0f;

        // Use linear activation for outputs
        individual.Activations[2] = ActivationType.Linear;
        individual.Activations[3] = ActivationType.Linear;

        var evaluator = new CPUEvaluator(spec);
        var inputs = new float[] { 0.5f, 0.8f };
        var outputs = evaluator.Evaluate(individual, inputs);

        Assert.Equal(0.5f, outputs[0], precision: 6);
        Assert.Equal(0.8f, outputs[1], precision: 6);
    }

    [Fact]
    public void CPUEvaluator_PassThroughWithWeights_ScalesCorrectly()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(1)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 1)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        individual.Weights[0] = 2.5f; // Scale by 2.5
        individual.Activations[1] = ActivationType.Linear;

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 1.0f });

        Assert.Equal(2.5f, outputs[0], precision: 6);
    }

    #endregion

    #region Bias Tests

    [Fact]
    public void CPUEvaluator_IntrinsicBias_AddsCorrectly()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(1)
            .AddOutputRow(1, ActivationType.Linear)
            .AddEdge(0, 1)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        individual.Weights[0] = 2.0f;
        individual.Biases[1] = 3.0f; // Set intrinsic bias for output node
        individual.Activations[1] = ActivationType.Linear;

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 1.0f });

        // output = 1.0 * 2.0 + 3.0 = 5.0
        Assert.Equal(5.0f, outputs[0], precision: 6);
    }

    #endregion

    #region Multi-Layer Network Tests

    [Fact]
    public void CPUEvaluator_TwoLayerNetwork_ComputesCorrectly()
    {
        // Network: input -> hidden -> output
        var spec = new SpeciesBuilder()
            .AddInputRow(1)
            .AddHiddenRow(2, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 1)
            .AddEdge(0, 2)
            .AddEdge(1, 3)
            .AddEdge(2, 3)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);

        // Set weights
        individual.Weights[0] = 2.0f;  // input -> hidden[0]
        individual.Weights[1] = 3.0f;  // input -> hidden[1]
        individual.Weights[2] = 0.5f;  // hidden[0] -> output
        individual.Weights[3] = 0.5f;  // hidden[1] -> output

        // Use linear activations for simplicity
        individual.Activations[1] = ActivationType.Linear;
        individual.Activations[2] = ActivationType.Linear;
        individual.Activations[3] = ActivationType.Linear;

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 1.0f });

        // hidden[0] = 1.0 * 2.0 = 2.0
        // hidden[1] = 1.0 * 3.0 = 3.0
        // output = 2.0 * 0.5 + 3.0 * 0.5 = 1.0 + 1.5 = 2.5
        Assert.Equal(2.5f, outputs[0], precision: 6);
    }

    [Fact]
    public void CPUEvaluator_MultipleInputsToSameNode_AccumulatesCorrectly()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddOutputRow(1, ActivationType.Linear)
            .AddEdge(0, 2)  // input[0] -> output
            .AddEdge(1, 2)  // input[1] -> output
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        // After BuildRowPlans, edges are sorted by destination, then source
        // All edges go to node 2, so they're sorted by source: 0, 1
        individual.Weights[0] = 2.0f;  // input[0] weight (edge 0->2)
        individual.Weights[1] = 3.0f;  // input[1] weight (edge 1->2)
        individual.Biases[2] = 0.5f;   // intrinsic bias
        individual.Activations[2] = ActivationType.Linear;

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 1.0f, 1.0f });

        // output = 1.0*2.0 + 1.0*3.0 + 0.5 = 2.0 + 3.0 + 0.5 = 5.5
        Assert.Equal(5.5f, outputs[0], precision: 6);
    }

    #endregion

    #region Activation Function Tests

    [Fact]
    public void CPUEvaluator_ReLUActivation_ClipsNegative()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(1)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 1)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        individual.Weights[0] = -2.0f; // Negative weight
        individual.Activations[1] = ActivationType.ReLU;

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 1.0f });

        // -2.0 * 1.0 = -2.0, ReLU(-2.0) = 0.0
        Assert.Equal(0.0f, outputs[0], precision: 6);
    }

    [Fact]
    public void CPUEvaluator_TanhActivation_BoundsOutput()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(1)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 1)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        individual.Weights[0] = 100.0f; // Very large weight
        individual.Activations[1] = ActivationType.Tanh;

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 1.0f });

        // Tanh of large number approaches 1.0
        Assert.True(outputs[0] > 0.99f);
        Assert.True(outputs[0] <= 1.0f);
    }

    [Fact]
    public void CPUEvaluator_LeakyReLUWithParameters_UsesAlpha()
    {
        var spec = new SpeciesBuilder()
            .AddInputRow(1)
            .AddOutputRow(1, ActivationType.Tanh)
            .AddEdge(0, 1)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        individual.Weights[0] = -1.0f;
        individual.Activations[1] = ActivationType.LeakyReLU;
        individual.SetNodeParams(1, new[] { 0.1f, 0f, 0f, 0f }); // α = 0.1

        var evaluator = new CPUEvaluator(spec);
        var outputs = evaluator.Evaluate(individual, new[] { 5.0f });

        // -1.0 * 5.0 = -5.0, LeakyReLU(-5.0, α=0.1) = -5.0 * 0.1 = -0.5
        Assert.Equal(-0.5f, outputs[0], precision: 6);
    }

    #endregion

    #region XOR Problem Test

    [Fact]
    public void CPUEvaluator_ComplexNetwork_ExecutesWithoutError()
    {
        // Test that a complex multi-layer network can be evaluated
        // Architecture: 2 inputs, 2 hidden (ReLU), 1 output (Tanh)
        var spec = new SpeciesBuilder()
            .AddInputRow(2)
            .AddHiddenRow(2, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(1, ActivationType.Tanh)
            .Build();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        var random = new Random(42);

        // Initialize with random weights
        MutationOperators.InitializeWeights(individual, spec, random);

        individual.Activations[2] = ActivationType.ReLU;
        individual.Activations[3] = ActivationType.ReLU;
        individual.Activations[4] = ActivationType.Tanh;

        var evaluator = new CPUEvaluator(spec);

        // Test that we can evaluate without errors
        var test00 = evaluator.Evaluate(individual, new[] { 0.0f, 0.0f });
        var test01 = evaluator.Evaluate(individual, new[] { 0.0f, 1.0f });
        var test10 = evaluator.Evaluate(individual, new[] { 1.0f, 0.0f });
        var test11 = evaluator.Evaluate(individual, new[] { 1.0f, 1.0f });

        // Just verify outputs are bounded by Tanh
        Assert.InRange(test00[0], -1.0f, 1.0f);
        Assert.InRange(test01[0], -1.0f, 1.0f);
        Assert.InRange(test10[0], -1.0f, 1.0f);
        Assert.InRange(test11[0], -1.0f, 1.0f);

        // Verify outputs are finite
        Assert.True(float.IsFinite(test00[0]));
        Assert.True(float.IsFinite(test01[0]));
        Assert.True(float.IsFinite(test10[0]));
        Assert.True(float.IsFinite(test11[0]));
    }

    #endregion

    #region Determinism Tests

    [Fact]
    public void CPUEvaluator_SameInputs_ProducesSameOutputs()
    {
        var spec = CreateSimpleSpec();
        spec.BuildRowPlans();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        InitializeRandomIndividual(individual, new Random(42));

        var evaluator = new CPUEvaluator(spec);
        var inputs = new[] { 0.3f, 0.7f };

        var outputs1 = evaluator.Evaluate(individual, inputs).ToArray();
        var outputs2 = evaluator.Evaluate(individual, inputs).ToArray();

        for (int i = 0; i < outputs1.Length; i++)
        {
            Assert.Equal(outputs1[i], outputs2[i]);
        }
    }

    [Fact]
    public void CPUEvaluator_MultipleEvaluations_AreIndependent()
    {
        var spec = CreateSimpleSpec();
        spec.BuildRowPlans();

        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);
        InitializeRandomIndividual(individual, new Random(42));

        var evaluator = new CPUEvaluator(spec);

        var outputs1 = evaluator.Evaluate(individual, new[] { 1.0f, 0.0f }).ToArray();
        var outputs2 = evaluator.Evaluate(individual, new[] { 0.0f, 1.0f }).ToArray();

        // Different inputs should produce different outputs (unless network is degenerate)
        bool anyDifferent = false;
        for (int i = 0; i < outputs1.Length; i++)
        {
            if (MathF.Abs(outputs1[i] - outputs2[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Different inputs should typically produce different outputs");
    }

    #endregion

    #region Helper Methods

    private static SpeciesSpec CreateSimpleSpec()
    {
        var random = new Random(42);
        return new SpeciesBuilder()
            .AddInputRow(2)
            .AddOutputRow(3, ActivationType.Tanh)
            .InitializeSparse(random)
            .Build();
    }

    private static void InitializeRandomIndividual(Individual individual, Random random)
    {
        for (int i = 0; i < individual.Weights.Length; i++)
            individual.Weights[i] = random.NextSingle() * 2.0f - 1.0f;

        for (int i = 0; i < individual.Activations.Length; i++)
            individual.Activations[i] = ActivationType.Linear;
    }

    #endregion
}
