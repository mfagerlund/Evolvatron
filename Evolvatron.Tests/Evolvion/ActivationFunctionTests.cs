using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Comprehensive tests for all 11 activation functions
/// </summary>
public class ActivationFunctionTests
{
    [Fact]
    public void Linear_ReturnsInputUnchanged()
    {
        var activation = ActivationType.Linear;
        var result = activation.Evaluate(2.5f, Array.Empty<float>());
        Assert.Equal(2.5f, result, precision: 6);

        result = activation.Evaluate(-1.3f, Array.Empty<float>());
        Assert.Equal(-1.3f, result, precision: 6);
    }

    [Fact]
    public void Tanh_ProducesBoundedOutput()
    {
        var activation = ActivationType.Tanh;

        // Tanh(0) = 0
        Assert.Equal(0.0f, activation.Evaluate(0.0f, Array.Empty<float>()), precision: 6);

        // Tanh approaches ±1 for large inputs
        Assert.True(activation.Evaluate(10.0f, Array.Empty<float>()) > 0.999f);
        Assert.True(activation.Evaluate(-10.0f, Array.Empty<float>()) < -0.999f);

        // Known value: tanh(1) ≈ 0.7616
        Assert.Equal(0.7616f, activation.Evaluate(1.0f, Array.Empty<float>()), precision: 3);
    }

    [Fact]
    public void Sigmoid_ProducesBoundedOutput()
    {
        var activation = ActivationType.Sigmoid;

        // Sigmoid(0) = 0.5
        Assert.Equal(0.5f, activation.Evaluate(0.0f, Array.Empty<float>()), precision: 6);

        // Sigmoid approaches 1 for large positive, 0 for large negative
        Assert.True(activation.Evaluate(10.0f, Array.Empty<float>()) > 0.9999f);
        Assert.True(activation.Evaluate(-10.0f, Array.Empty<float>()) < 0.0001f);

        // Known value: sigmoid(1) ≈ 0.7311
        Assert.Equal(0.7311f, activation.Evaluate(1.0f, Array.Empty<float>()), precision: 3);
    }

    [Fact]
    public void ReLU_ZerosNegativeInputs()
    {
        var activation = ActivationType.ReLU;

        Assert.Equal(0.0f, activation.Evaluate(-5.0f, Array.Empty<float>()));
        Assert.Equal(0.0f, activation.Evaluate(-0.1f, Array.Empty<float>()));
        Assert.Equal(0.0f, activation.Evaluate(0.0f, Array.Empty<float>()));
        Assert.Equal(5.0f, activation.Evaluate(5.0f, Array.Empty<float>()));
    }

    [Fact]
    public void LeakyReLU_UsesAlphaParameter()
    {
        var activation = ActivationType.LeakyReLU;
        float alpha = 0.01f;
        var parameters = new[] { alpha };

        Assert.Equal(5.0f, activation.Evaluate(5.0f, parameters));
        Assert.Equal(0.0f, activation.Evaluate(0.0f, parameters));
        Assert.Equal(-0.05f, activation.Evaluate(-5.0f, parameters), precision: 6);
    }

    [Fact]
    public void LeakyReLU_RequiresOneParameter()
    {
        Assert.Equal(1, ActivationType.LeakyReLU.RequiredParamCount());
        var defaults = ActivationType.LeakyReLU.GetDefaultParameters();
        Assert.Single(defaults);
        Assert.Equal(0.01f, defaults[0]); // Standard Leaky ReLU alpha
    }

    [Fact]
    public void ELU_UsesAlphaParameter()
    {
        var activation = ActivationType.ELU;
        float alpha = 1.0f;
        var parameters = new[] { alpha };

        // ELU(x) = x for x > 0
        Assert.Equal(5.0f, activation.Evaluate(5.0f, parameters));

        // ELU(0) = 0
        Assert.Equal(0.0f, activation.Evaluate(0.0f, parameters), precision: 6);

        // ELU(x) = α*(exp(x) - 1) for x < 0
        // ELU(-1) ≈ 1.0 * (0.3679 - 1) ≈ -0.6321
        Assert.Equal(-0.6321f, activation.Evaluate(-1.0f, parameters), precision: 3);
    }

    [Fact]
    public void ELU_RequiresOneParameter()
    {
        Assert.Equal(1, ActivationType.ELU.RequiredParamCount());
        var defaults = ActivationType.ELU.GetDefaultParameters();
        Assert.Single(defaults);
        Assert.Equal(1.0f, defaults[0]); // Standard ELU alpha
    }

    [Fact]
    public void Softsign_ProducesBoundedOutput()
    {
        var activation = ActivationType.Softsign;

        // Softsign(0) = 0
        Assert.Equal(0.0f, activation.Evaluate(0.0f, Array.Empty<float>()));

        // Softsign(x) = x / (1 + |x|)
        // Softsign(1) = 1/2 = 0.5
        Assert.Equal(0.5f, activation.Evaluate(1.0f, Array.Empty<float>()), precision: 6);

        // Softsign(-1) = -1/2 = -0.5
        Assert.Equal(-0.5f, activation.Evaluate(-1.0f, Array.Empty<float>()), precision: 6);

        // Softsign approaches ±1 for large inputs
        Assert.True(activation.Evaluate(100.0f, Array.Empty<float>()) > 0.99f);
        Assert.True(activation.Evaluate(-100.0f, Array.Empty<float>()) < -0.99f);
    }

    [Fact]
    public void Softplus_ProducesPositiveOutput()
    {
        var activation = ActivationType.Softplus;

        // Softplus(0) ≈ ln(2) ≈ 0.6931
        Assert.Equal(0.6931f, activation.Evaluate(0.0f, Array.Empty<float>()), precision: 3);

        // Softplus(x) ≈ x for large positive x
        float largeInput = 10.0f;
        Assert.True(MathF.Abs(activation.Evaluate(largeInput, Array.Empty<float>()) - largeInput) < 0.01f);

        // Softplus is always positive
        Assert.True(activation.Evaluate(-10.0f, Array.Empty<float>()) > 0.0f);
    }

    [Fact]
    public void Sin_ProducesPeriodicOutput()
    {
        var activation = ActivationType.Sin;

        // Sin(0) = 0
        Assert.Equal(0.0f, activation.Evaluate(0.0f, Array.Empty<float>()), precision: 6);

        // Sin(π/2) = 1
        Assert.Equal(1.0f, activation.Evaluate(MathF.PI / 2, Array.Empty<float>()), precision: 6);

        // Sin(π) ≈ 0
        Assert.Equal(0.0f, activation.Evaluate(MathF.PI, Array.Empty<float>()), precision: 5);

        // Sin(3π/2) = -1
        Assert.Equal(-1.0f, activation.Evaluate(3 * MathF.PI / 2, Array.Empty<float>()), precision: 6);

        // Bounded output [-1, 1]
        Assert.InRange(activation.Evaluate(123.456f, Array.Empty<float>()), -1.0f, 1.0f);
    }

    [Fact]
    public void Gaussian_ProducesBellCurve()
    {
        var activation = ActivationType.Gaussian;

        // Gaussian(0) = exp(0) = 1
        Assert.Equal(1.0f, activation.Evaluate(0.0f, Array.Empty<float>()), precision: 6);

        // Gaussian(1) = exp(-1) ≈ 0.3679
        Assert.Equal(0.3679f, activation.Evaluate(1.0f, Array.Empty<float>()), precision: 3);

        // Gaussian(-1) = exp(-1) ≈ 0.3679 (symmetric)
        Assert.Equal(0.3679f, activation.Evaluate(-1.0f, Array.Empty<float>()), precision: 3);

        // Gaussian approaches 0 for large |x|
        Assert.True(activation.Evaluate(5.0f, Array.Empty<float>()) < 0.001f);
        Assert.True(activation.Evaluate(-5.0f, Array.Empty<float>()) < 0.001f);

        // Always positive, bounded [0, 1]
        Assert.InRange(activation.Evaluate(2.5f, Array.Empty<float>()), 0.0f, 1.0f);
    }

    [Fact]
    public void GELU_ProducesSmoothedReLU()
    {
        var activation = ActivationType.GELU;

        // GELU(0) ≈ 0
        Assert.Equal(0.0f, activation.Evaluate(0.0f, Array.Empty<float>()), precision: 2);

        // GELU(x) ≈ x for large positive x
        float largePositive = 3.0f;
        Assert.True(MathF.Abs(activation.Evaluate(largePositive, Array.Empty<float>()) - largePositive) < 0.1f);

        // GELU(x) ≈ 0 for large negative x (unlike ReLU, not exactly 0)
        Assert.True(MathF.Abs(activation.Evaluate(-3.0f, Array.Empty<float>())) < 0.01f);

        // GELU(1) ≈ 0.841 (known approximate value)
        Assert.Equal(0.84f, activation.Evaluate(1.0f, Array.Empty<float>()), precision: 1);
    }

    [Theory]
    [InlineData(ActivationType.Linear)]
    [InlineData(ActivationType.Tanh)]
    [InlineData(ActivationType.Sigmoid)]
    [InlineData(ActivationType.ReLU)]
    [InlineData(ActivationType.Softsign)]
    [InlineData(ActivationType.Softplus)]
    [InlineData(ActivationType.Sin)]
    [InlineData(ActivationType.Gaussian)]
    [InlineData(ActivationType.GELU)]
    public void ParameterlessActivations_RequireZeroParameters(ActivationType activation)
    {
        Assert.Equal(0, activation.RequiredParamCount());
        Assert.Empty(activation.GetDefaultParameters());
    }

    [Theory]
    [InlineData(ActivationType.Linear)]
    [InlineData(ActivationType.Tanh)]
    public void OnlyLinearAndTanh_ValidForOutput(ActivationType activation)
    {
        Assert.True(activation.IsValidForOutput());
    }

    [Theory]
    [InlineData(ActivationType.Sigmoid)]
    [InlineData(ActivationType.ReLU)]
    [InlineData(ActivationType.LeakyReLU)]
    [InlineData(ActivationType.ELU)]
    [InlineData(ActivationType.Softsign)]
    [InlineData(ActivationType.Softplus)]
    [InlineData(ActivationType.Sin)]
    [InlineData(ActivationType.Gaussian)]
    [InlineData(ActivationType.GELU)]
    public void NonOutputActivations_NotValidForOutput(ActivationType activation)
    {
        Assert.False(activation.IsValidForOutput());
    }

    [Fact]
    public void AllActivations_ProduceFiniteOutputs()
    {
        // Test that no activation produces NaN or Infinity for reasonable inputs
        var testInputs = new[] { -10.0f, -1.0f, 0.0f, 1.0f, 10.0f };
        var allActivations = Enum.GetValues<ActivationType>();

        foreach (var activation in allActivations)
        {
            var defaultParams = activation.GetDefaultParameters();
            foreach (var input in testInputs)
            {
                var output = activation.Evaluate(input, defaultParams);
                Assert.True(float.IsFinite(output),
                    $"{activation} produced non-finite output {output} for input {input}");
            }
        }
    }

    [Fact]
    public void ActivationParameters_HaveCorrectSizes()
    {
        // Verify parameter arrays match required counts
        var allActivations = Enum.GetValues<ActivationType>();

        foreach (var activation in allActivations)
        {
            var requiredCount = activation.RequiredParamCount();
            var defaults = activation.GetDefaultParameters();

            Assert.Equal(requiredCount, defaults.Length);
        }
    }
}
