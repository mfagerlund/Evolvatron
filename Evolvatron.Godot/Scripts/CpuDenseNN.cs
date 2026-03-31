using System;

namespace Evolvatron.Godot;

/// <summary>
/// CPU port of GPU DenseNN.ForwardPass — same weight layout (dst-major per layer).
/// Weights: [layer0→1 weights][layer1→2 weights]...
/// Biases:  [layer1 biases][layer2 biases]...
/// </summary>
public static class CpuDenseNN
{
    public static void ForwardPass(
        float[] weights,
        float[] biases,
        int[] layerSizes,
        ReadOnlySpan<float> observations,
        Span<float> actions)
    {
        int numLayers = layerSizes.Length;

        if (numLayers == 2)
        {
            // Direct input → output (no hidden layers)
            int inSz = layerSizes[0];
            int outSz = layerSizes[1];
            for (int dst = 0; dst < outSz; dst++)
            {
                float sum = biases[dst];
                for (int src = 0; src < inSz; src++)
                    sum += observations[src] * weights[dst * inSz + src];
                actions[dst] = Tanh(sum);
            }
            return;
        }

        // 3+ layers: ping-pong buffers
        const int MaxWidth = 64;
        Span<float> buf = stackalloc float[MaxWidth * 2];
        int readOff = 0;
        int writeOff = MaxWidth;
        int wOff = 0;
        int bOff = 0;

        // First layer: observations → buffer
        {
            int prevSz = layerSizes[0];
            int currSz = layerSizes[1];
            for (int dst = 0; dst < currSz; dst++)
            {
                float sum = biases[bOff + dst];
                for (int src = 0; src < prevSz; src++)
                    sum += observations[src] * weights[wOff + dst * prevSz + src];
                buf[writeOff + dst] = Tanh(sum);
            }
            wOff += prevSz * currSz;
            bOff += currSz;
            (readOff, writeOff) = (writeOff, readOff);
        }

        // Middle layers: buffer → buffer
        for (int layer = 1; layer < numLayers - 2; layer++)
        {
            int prevSz = layerSizes[layer];
            int currSz = layerSizes[layer + 1];
            for (int dst = 0; dst < currSz; dst++)
            {
                float sum = biases[bOff + dst];
                for (int src = 0; src < prevSz; src++)
                    sum += buf[readOff + src] * weights[wOff + dst * prevSz + src];
                buf[writeOff + dst] = Tanh(sum);
            }
            wOff += prevSz * currSz;
            bOff += currSz;
            (readOff, writeOff) = (writeOff, readOff);
        }

        // Last layer: buffer → actions
        {
            int prevSz = layerSizes[numLayers - 2];
            int currSz = layerSizes[numLayers - 1];
            for (int dst = 0; dst < currSz; dst++)
            {
                float sum = biases[bOff + dst];
                for (int src = 0; src < prevSz; src++)
                    sum += buf[readOff + src] * weights[wOff + dst * prevSz + src];
                actions[dst] = Tanh(sum);
            }
        }
    }

    public static (float[] weights, float[] biases) SplitParams(float[] allParams, int[] layerSizes)
    {
        int totalWeights = 0;
        int totalBiases = 0;
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            totalWeights += layerSizes[i] * layerSizes[i + 1];
            totalBiases += layerSizes[i + 1];
        }

        var weights = new float[totalWeights];
        var biases = new float[totalBiases];
        Array.Copy(allParams, 0, weights, 0, totalWeights);
        Array.Copy(allParams, totalWeights, biases, 0, totalBiases);
        return (weights, biases);
    }

    private static float Tanh(float x)
    {
        if (x > 10f) return 1f;
        if (x < -10f) return -1f;
        float exp2x = MathF.Exp(2f * x);
        return (exp2x - 1f) / (exp2x + 1f);
    }
}
