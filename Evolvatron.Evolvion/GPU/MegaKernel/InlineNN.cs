using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Evolvatron.Core.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Inline neural network forward pass for a single world/episode.
/// Ported from GPUEvolvionKernels.EvaluateRowForEpisodesKernel.
/// One thread runs the full NN for one individual: set inputs → evaluate rows → extract outputs.
/// </summary>
public static class InlineNN
{
    /// <summary>
    /// Run full NN forward pass for one world/episode.
    /// Writes outputs to the provided output span in nn.NodeValues.
    /// Returns output values via the actions array.
    /// </summary>
    /// <param name="nn">Neural network views (node values, edges, weights, biases, activations, row plans).</param>
    /// <param name="observations">Observation buffer (8 floats per world).</param>
    /// <param name="actions">Action buffer (2 floats per world).</param>
    /// <param name="worldIdx">Index of the current world/episode.</param>
    /// <param name="cfg">Mega kernel config with NN layout parameters.</param>
    public static void ForwardPassOneWorld(
        NNViews nn,
        ArrayView<float> observations,
        ArrayView<float> actions,
        int worldIdx,
        MegaKernelConfig cfg)
    {
        ForwardPass(nn, observations, actions, worldIdx,
            cfg.TotalNodes, cfg.TotalEdges, cfg.NumRows, cfg.InputSize, cfg.OutputSize);
    }

    /// <summary>
    /// Generic NN forward pass usable by any environment kernel.
    /// Takes explicit NN layout parameters instead of a domain-specific config struct.
    /// </summary>
    public static void ForwardPass(
        NNViews nn,
        ArrayView<float> observations,
        ArrayView<float> actions,
        int worldIdx,
        int totalNodes, int totalEdges, int numRows, int inputSize, int outputSize)
    {
        int episodeIdx = worldIdx;
        int individualIdx = episodeIdx;
        int nodesPerIndividual = totalNodes;
        int weightsPerIndividual = totalEdges;

        int nodeOffset = episodeIdx * nodesPerIndividual;
        int weightOffset = individualIdx * weightsPerIndividual;
        int biasOffset = individualIdx * nodesPerIndividual;
        int inputOffset = episodeIdx * inputSize;

        // 1. Set inputs (copy observations to input nodes)
        for (int i = 0; i < inputSize; i++)
        {
            nn.NodeValues[nodeOffset + i] = observations[inputOffset + i];
        }

        // 2. Evaluate rows 1..NumRows-1 (row 0 is inputs)
        for (int rowIdx = 1; rowIdx < numRows; rowIdx++)
        {
            var rowPlan = nn.RowPlans[rowIdx];

            // Clear row nodes
            for (int i = 0; i < rowPlan.NodeCount; i++)
            {
                nn.NodeValues[nodeOffset + rowPlan.NodeStart + i] = 0.0f;
            }

            // Accumulate weighted inputs
            for (int edgeIdx = rowPlan.EdgeStart; edgeIdx < rowPlan.EdgeStart + rowPlan.EdgeCount; edgeIdx++)
            {
                var edge = nn.Edges[edgeIdx];
                float weight = nn.Weights[weightOffset + edgeIdx];
                float sourceValue = nn.NodeValues[nodeOffset + edge.Source];
                nn.NodeValues[nodeOffset + edge.Dest] += weight * sourceValue;
            }

            // Add biases
            for (int i = 0; i < rowPlan.NodeCount; i++)
            {
                int nodeIdx = rowPlan.NodeStart + i;
                nn.NodeValues[nodeOffset + nodeIdx] += nn.Biases[biasOffset + nodeIdx];
            }

            // Apply activations
            for (int i = 0; i < rowPlan.NodeCount; i++)
            {
                int nodeIdx = rowPlan.NodeStart + i;
                int globalNodeIdx = nodeOffset + nodeIdx;
                float preActivation = nn.NodeValues[globalNodeIdx];
                byte activationType = nn.Activations[biasOffset + nodeIdx];

                int paramOff = (biasOffset + nodeIdx) * 4;
                float param0 = nn.NodeParams[paramOff];
                float param1 = nn.NodeParams[paramOff + 1];
                float param2 = nn.NodeParams[paramOff + 2];
                float param3 = nn.NodeParams[paramOff + 3];

                nn.NodeValues[globalNodeIdx] = EvaluateActivation(
                    activationType, preActivation, param0, param1, param2, param3);
            }
        }

        // 3. Extract outputs from the last row
        int outputRowIdx = numRows - 1;
        var outputRow = nn.RowPlans[outputRowIdx];
        int outputOffset = episodeIdx * outputSize;

        for (int i = 0; i < outputSize; i++)
        {
            actions[outputOffset + i] = nn.NodeValues[nodeOffset + outputRow.NodeStart + i];
        }
    }

    /// <summary>
    /// Evaluates activation function on GPU.
    /// Duplicated from GPUEvolvionKernels to be reachable from the fused kernel.
    /// </summary>
    private static float EvaluateActivation(
        byte activationType,
        float x,
        float param0,
        float param1,
        float param2,
        float param3)
    {
        if (activationType == 0) return x;

        if (activationType == 1)
        {
            float exp2x = XMath.Exp(2.0f * x);
            return (exp2x - 1.0f) / (exp2x + 1.0f);
        }

        if (activationType == 2)
            return 1.0f / (1.0f + XMath.Exp(-x));

        if (activationType == 3) return x > 0.0f ? x : 0.0f;
        if (activationType == 4) return x > 0.0f ? x : param0 * x;

        if (activationType == 5)
            return x > 0.0f ? x : param0 * (XMath.Exp(x) - 1.0f);

        if (activationType == 6)
            return x / (1.0f + XMath.Abs(x));

        if (activationType == 7)
        {
            float clampX = x > 20.0f ? 20.0f : (x < -20.0f ? -20.0f : x);
            return x > 20.0f ? x : XMath.Log(1.0f + XMath.Exp(clampX));
        }

        if (activationType == 8) return XMath.Sin(x);
        if (activationType == 9) return XMath.Exp(-x * x);

        if (activationType == 10)
        {
            float arg = 0.7978845608f * (x + 0.044715f * x * x * x);
            float exp2arg = XMath.Exp(2.0f * arg);
            float tanhArg = (exp2arg - 1.0f) / (exp2arg + 1.0f);
            return 0.5f * x * (1.0f + tanhArg);
        }

        return x;
    }
}
