namespace Evolvatron.Evolvion;

/// <summary>
/// Activation functions available for neural network nodes.
/// Each activation may require specific parameters stored in node.param[].
/// </summary>
public enum ActivationType : byte
{
    Linear = 0,      // f(x) = x
    Tanh = 1,        // f(x) = tanh(x)
    Sigmoid = 2,     // f(x) = 1 / (1 + exp(-x))
    ReLU = 3,        // f(x) = max(0, x)
    LeakyReLU = 4,   // f(x) = x > 0 ? x : α*x (requires param[0] = α)
    ELU = 5,         // f(x) = x > 0 ? x : α*(exp(x) - 1) (requires param[0] = α)
    Softsign = 6,    // f(x) = x / (1 + |x|)
    Softplus = 7,    // f(x) = log(1 + exp(x))
    Sin = 8,         // f(x) = sin(x)
    Gaussian = 9,    // f(x) = exp(-x²)
    GELU = 10        // f(x) = x * Φ(x) where Φ is standard normal CDF
}

/// <summary>
/// Helper methods for activation functions
/// </summary>
public static class ActivationTypeExtensions
{
    /// <summary>
    /// Returns the number of parameters required for this activation type
    /// </summary>
    public static int RequiredParamCount(this ActivationType type)
    {
        return type switch
        {
            ActivationType.LeakyReLU => 1, // α parameter
            ActivationType.ELU => 1,       // α parameter
            _ => 0
        };
    }

    /// <summary>
    /// Returns whether this activation is allowed for output nodes
    /// </summary>
    public static bool IsValidForOutput(this ActivationType type)
    {
        return type is ActivationType.Linear or ActivationType.Tanh;
    }

    /// <summary>
    /// Evaluates the activation function
    /// </summary>
    public static float Evaluate(this ActivationType type, float x, ReadOnlySpan<float> parameters)
    {
        return type switch
        {
            ActivationType.Linear => x,
            ActivationType.Tanh => MathF.Tanh(x),
            ActivationType.Sigmoid => 1.0f / (1.0f + MathF.Exp(-x)),
            ActivationType.ReLU => MathF.Max(0.0f, x),
            ActivationType.LeakyReLU => x > 0 ? x : parameters[0] * x,
            ActivationType.ELU => x > 0 ? x : parameters[0] * (MathF.Exp(x) - 1.0f),
            ActivationType.Softsign => x / (1.0f + MathF.Abs(x)),
            ActivationType.Softplus => MathF.Log(1.0f + MathF.Exp(x)),
            ActivationType.Sin => MathF.Sin(x),
            ActivationType.Gaussian => MathF.Exp(-x * x),
            ActivationType.GELU => x * 0.5f * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x))),
            _ => throw new ArgumentException($"Unknown activation type: {type}")
        };
    }

    /// <summary>
    /// Returns default parameter values for this activation type
    /// </summary>
    public static float[] GetDefaultParameters(this ActivationType type)
    {
        return type switch
        {
            ActivationType.LeakyReLU => new[] { 0.01f }, // Standard α for Leaky ReLU
            ActivationType.ELU => new[] { 1.0f },        // Standard α for ELU
            _ => Array.Empty<float>()
        };
    }
}
