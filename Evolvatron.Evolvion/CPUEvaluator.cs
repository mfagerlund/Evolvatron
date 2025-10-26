namespace Evolvatron.Evolvion;

/// <summary>
/// CPU-based neural network forward pass evaluator using RowPlan execution.
/// Serves as reference implementation for GPU parity testing.
/// </summary>
public class CPUEvaluator
{
    private readonly SpeciesSpec _spec;
    private float[] _nodeValues;

    public CPUEvaluator(SpeciesSpec spec)
    {
        _spec = spec;
        _nodeValues = new float[spec.TotalNodes];
    }

    /// <summary>
    /// Evaluates the neural network for a single individual with given inputs.
    /// Returns the output layer activations.
    /// </summary>
    public ReadOnlySpan<float> Evaluate(Individual individual, ReadOnlySpan<float> inputs)
    {
        // Validate inputs
        int inputRowSize = _spec.RowCounts[1]; // Row 1 is input layer
        if (inputs.Length != inputRowSize)
        {
            throw new ArgumentException(
                $"Input size mismatch: expected {inputRowSize}, got {inputs.Length}");
        }

        // Clear node values
        Array.Fill(_nodeValues, 0.0f);

        // Row 0: Bias (always 1.0)
        _nodeValues[0] = 1.0f;

        // Row 1: Copy input values
        RowPlan inputPlan = _spec.RowPlans[1];
        for (int i = 0; i < inputRowSize; i++)
        {
            _nodeValues[inputPlan.NodeStart + i] = inputs[i];
        }

        // Evaluate remaining rows (2 onwards)
        for (int rowIdx = 2; rowIdx < _spec.RowPlans.Length; rowIdx++)
        {
            EvaluateRow(rowIdx, individual);
        }

        // Return output layer values
        RowPlan outputPlan = _spec.RowPlans[^1];
        return _nodeValues.AsSpan(outputPlan.NodeStart, outputPlan.NodeCount);
    }

    /// <summary>
    /// Evaluates a single row by computing weighted sums and applying activations
    /// </summary>
    private void EvaluateRow(int rowIdx, Individual individual)
    {
        RowPlan plan = _spec.RowPlans[rowIdx];

        // Initialize row nodes to zero
        for (int i = 0; i < plan.NodeCount; i++)
        {
            _nodeValues[plan.NodeStart + i] = 0.0f;
        }

        // Accumulate weighted inputs from edges
        for (int edgeIdx = plan.EdgeStart; edgeIdx < plan.EdgeStart + plan.EdgeCount; edgeIdx++)
        {
            var (source, dest) = _spec.Edges[edgeIdx];
            float weight = individual.Weights[edgeIdx];
            float sourceValue = _nodeValues[source];

            _nodeValues[dest] += weight * sourceValue;
        }

        // Apply activations
        for (int i = 0; i < plan.NodeCount; i++)
        {
            int nodeIdx = plan.NodeStart + i;
            float preActivation = _nodeValues[nodeIdx];
            ActivationType activation = individual.Activations[nodeIdx];
            ReadOnlySpan<float> parameters = individual.GetNodeParams(nodeIdx);

            _nodeValues[nodeIdx] = activation.Evaluate(preActivation, parameters);
        }
    }

    /// <summary>
    /// Gets the current node values (for debugging/inspection)
    /// </summary>
    public ReadOnlySpan<float> GetNodeValues() => _nodeValues;

    /// <summary>
    /// Gets the value of a specific node (for debugging)
    /// </summary>
    public float GetNodeValue(int nodeIndex)
    {
        if (nodeIndex < 0 || nodeIndex >= _nodeValues.Length)
            throw new ArgumentOutOfRangeException(nameof(nodeIndex));

        return _nodeValues[nodeIndex];
    }
}
