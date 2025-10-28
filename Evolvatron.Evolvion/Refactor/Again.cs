namespace Evolvatron.Evolvion.Refactor;

public enum ActivationTypeX
{
    Linear,
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU,
    ELU,
    Swish,
    Gaussian
}

public class SpeciesEx
{
    public float[] Weights { get; set; } = Array.Empty<float>();
    public float[] Biases { get; set; } = Array.Empty<float>();
}

public class NodeEx
{
    
}

public class IndividualEx
{
    public List<NodeEx> Nodes { get; set; } = new();
}