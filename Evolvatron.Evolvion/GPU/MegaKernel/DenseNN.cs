using ILGPU;
using ILGPU.Algorithms;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Dense fully-connected NN forward pass for GPU kernel.
/// No edge arrays, no RowPlans, no scatter writes — just contiguous matmul.
/// Supports variable layer widths (e.g., 5→4→4→3).
/// Uses ping-pong local memory buffers for hidden activations.
/// </summary>
public static class DenseNN
{
    private const int MaxLayerWidth = 64;

    /// <summary>
    /// Dense forward pass using DenseDoublePoleConfig for layout parameters.
    /// </summary>
    public static void ForwardPass(
        DenseNNViews nn,
        ArrayView<float> observations,
        ArrayView<float> actions,
        int worldIdx,
        DenseDoublePoleConfig cfg)
    {
        ForwardPass(nn.Weights, nn.Biases, nn.LayerSizes,
            observations, actions, worldIdx,
            cfg.NumLayers, cfg.TotalWeightsPerNet, cfg.TotalBiasesPerNet,
            cfg.InputSize, cfg.OutputSize);
    }

    /// <summary>
    /// Generic dense forward pass with explicit layout parameters.
    /// Weight layout: layers stored contiguously, dst-major (row-major) within each layer.
    ///   Layer i→i+1: layerSizes[i] × layerSizes[i+1] weights
    ///   w[dst=0,src=0], w[dst=0,src=1], ..., w[dst=1,src=0], ...
    /// Bias layout: one per non-input node, in layer order.
    /// </summary>
    public static void ForwardPass(
        ArrayView<float> weights,
        ArrayView<float> biases,
        ArrayView<int> layerSizes,
        ArrayView<float> observations,
        ArrayView<float> actions,
        int worldIdx,
        int numLayers,
        int totalWeightsPerNet,
        int totalBiasesPerNet,
        int inputSize,
        int outputSize)
    {
        int wBase = worldIdx * totalWeightsPerNet;
        int bBase = worldIdx * totalBiasesPerNet;
        int obsBase = worldIdx * inputSize;
        int actBase = worldIdx * outputSize;

        // Single local buffer, two halves for ping-pong
        var buf = LocalMemory.Allocate1D<float>(MaxLayerWidth * 2);
        int readOff = 0;
        int writeOff = MaxLayerWidth;

        int wOff = wBase;
        int bOff = bBase;

        // --- Handle 2-layer net (input → output, no hidden) ---
        if (numLayers == 2)
        {
            int prevSz = inputSize;
            int currSz = outputSize;
            for (int dst = 0; dst < currSz; dst++)
            {
                float sum = biases[bOff + dst];
                int wRow = wOff + dst * prevSz;
                for (int src = 0; src < prevSz; src++)
                    sum += observations[obsBase + src] * weights[wRow + src];
                actions[actBase + dst] = Tanh(sum);
            }
            return;
        }

        // --- First layer: observations → buffer ---
        {
            int prevSize = layerSizes[0];
            int currSize = layerSizes[1];
            for (int dst = 0; dst < currSize; dst++)
            {
                float sum = biases[bOff + dst];
                int wRow = wOff + dst * prevSize;
                for (int src = 0; src < prevSize; src++)
                    sum += observations[obsBase + src] * weights[wRow + src];
                buf[writeOff + dst] = Tanh(sum);
            }
            wOff += prevSize * currSize;
            bOff += currSize;
            int tmp = readOff; readOff = writeOff; writeOff = tmp;
        }

        // --- Middle layers: buffer → buffer (ping-pong) ---
        for (int layerIdx = 1; layerIdx < numLayers - 2; layerIdx++)
        {
            int prevSize = layerSizes[layerIdx];
            int currSize = layerSizes[layerIdx + 1];
            for (int dst = 0; dst < currSize; dst++)
            {
                float sum = biases[bOff + dst];
                int wRow = wOff + dst * prevSize;
                for (int src = 0; src < prevSize; src++)
                    sum += buf[readOff + src] * weights[wRow + src];
                buf[writeOff + dst] = Tanh(sum);
            }
            wOff += prevSize * currSize;
            bOff += currSize;
            int tmp = readOff; readOff = writeOff; writeOff = tmp;
        }

        // --- Last layer: buffer → actions ---
        {
            int prevSize = layerSizes[numLayers - 2];
            int currSize = layerSizes[numLayers - 1];
            for (int dst = 0; dst < currSize; dst++)
            {
                float sum = biases[bOff + dst];
                int wRow = wOff + dst * prevSize;
                for (int src = 0; src < prevSize; src++)
                    sum += buf[readOff + src] * weights[wRow + src];
                actions[actBase + dst] = Tanh(sum);
            }
        }
    }

    private static float Tanh(float x)
    {
        // Clamp to prevent Exp overflow → Inf/Inf = NaN → GPU memory corruption.
        // tanh(10) = 0.99999999587... ≈ 1.0f in float32.
        if (x > 10f) return 1f;
        if (x < -10f) return -1f;
        float exp2x = XMath.Exp(2f * x);
        return (exp2x - 1f) / (exp2x + 1f);
    }
}
