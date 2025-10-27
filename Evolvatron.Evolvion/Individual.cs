namespace Evolvatron.Evolvion;

/// <summary>
/// Represents a single individual within a species.
/// Stores weights, node parameters, activation functions, and fitness.
/// </summary>
public struct Individual
{
    /// <summary>
    /// Per-edge weights (length = number of edges in species topology)
    /// </summary>
    public float[] Weights;

    /// <summary>
    /// Per-node bias values (length = number of nodes)
    /// Each node has its own independent bias
    /// </summary>
    public float[] Biases;

    /// <summary>
    /// Per-node activation parameters (length = number of nodes × 4)
    /// Only first N params used per node based on activation type
    /// </summary>
    public float[] NodeParams;

    /// <summary>
    /// Per-node activation function (length = number of nodes)
    /// </summary>
    public ActivationType[] Activations;

    /// <summary>
    /// Bitset marking which nodes are active (connected to input and output)
    /// Used for analytics and debugging
    /// </summary>
    public bool[]? ActiveNodes;

    /// <summary>
    /// Current fitness value
    /// </summary>
    public float Fitness;

    /// <summary>
    /// Number of generations this individual has survived
    /// </summary>
    public int Age;

    /// <summary>
    /// Creates a new individual with specified array sizes
    /// </summary>
    public Individual(int edgeCount, int nodeCount)
    {
        Weights = new float[edgeCount];
        Biases = new float[nodeCount];
        NodeParams = new float[nodeCount * 4];
        Activations = new ActivationType[nodeCount];
        ActiveNodes = new bool[nodeCount];
        Fitness = 0.0f;
        Age = 0;
    }

    /// <summary>
    /// Deep copy constructor
    /// </summary>
    public Individual(Individual other)
    {
        Weights = other.Weights != null ? (float[])other.Weights.Clone() : Array.Empty<float>();
        Biases = other.Biases != null ? (float[])other.Biases.Clone() : Array.Empty<float>();
        NodeParams = other.NodeParams != null ? (float[])other.NodeParams.Clone() : Array.Empty<float>();
        Activations = other.Activations != null ? (ActivationType[])other.Activations.Clone() : Array.Empty<ActivationType>();
        ActiveNodes = other.ActiveNodes != null ? (bool[])other.ActiveNodes.Clone() : null;
        Fitness = other.Fitness;
        Age = other.Age;
    }

    /// <summary>
    /// Gets the parameters for a specific node
    /// </summary>
    public ReadOnlySpan<float> GetNodeParams(int nodeIndex)
    {
        return NodeParams.AsSpan(nodeIndex * 4, 4);
    }

    /// <summary>
    /// Sets the parameters for a specific node
    /// </summary>
    public void SetNodeParams(int nodeIndex, ReadOnlySpan<float> parameters)
    {
        parameters.CopyTo(NodeParams.AsSpan(nodeIndex * 4, parameters.Length));
    }
}
