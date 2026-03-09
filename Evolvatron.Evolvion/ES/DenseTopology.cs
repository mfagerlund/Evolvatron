namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Describes a fixed fully-connected neural network topology.
/// Supports variable layer widths (e.g., 5→4→4→3).
/// No edge arrays, no row plans — just layer sizes.
/// </summary>
public class DenseTopology
{
    public int[] LayerSizes { get; }
    public int TotalWeights { get; }
    public int TotalBiases { get; }
    public int TotalParams => TotalWeights + TotalBiases;
    public int InputSize => LayerSizes[0];
    public int OutputSize => LayerSizes[^1];
    public int MaxLayerWidth { get; }
    public int NumLayers => LayerSizes.Length;
    public int NumHiddenLayers => LayerSizes.Length - 2;

    public DenseTopology(params int[] layerSizes)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("Need at least input and output layers");

        LayerSizes = (int[])layerSizes.Clone();

        int totalWeights = 0;
        int totalBiases = 0;
        int maxWidth = 0;

        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            totalWeights += layerSizes[i] * layerSizes[i + 1];
            totalBiases += layerSizes[i + 1];
        }

        for (int i = 1; i < layerSizes.Length - 1; i++)
            maxWidth = Math.Max(maxWidth, layerSizes[i]);
        maxWidth = Math.Max(maxWidth, layerSizes[^1]); // include output width

        TotalWeights = totalWeights;
        TotalBiases = totalBiases;
        MaxLayerWidth = maxWidth;
    }

    /// <summary>
    /// Creates a topology for the DPNV benchmark with Elman recurrence.
    /// Adds contextSize to both input and output layer sizes.
    /// </summary>
    public static DenseTopology ForDPNV(int[] hiddenSizes, int contextSize = 2, bool includeVelocity = false)
    {
        int baseInput = includeVelocity ? 6 : 3;
        int baseOutput = 1; // single force action

        var layers = new int[hiddenSizes.Length + 2];
        layers[0] = baseInput + contextSize;
        for (int i = 0; i < hiddenSizes.Length; i++)
            layers[i + 1] = hiddenSizes[i];
        layers[^1] = baseOutput + contextSize;

        return new DenseTopology(layers);
    }

    public override string ToString()
        => string.Join("→", LayerSizes) + $" ({TotalParams} params, {TotalWeights}w+{TotalBiases}b)";
}
