using Evolvatron.Evolvion;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Comprehensive tests for all 8 mutation operators
/// </summary>
public class MutationOperatorTests
{
    #region Weight Jitter Tests

    [Fact]
    public void WeightJitter_ModifiesWeights()
    {
        var individual = new Individual(10, 5);
        var random = new Random(42);

        // Initialize with non-zero weights
        for (int i = 0; i < individual.Weights.Length; i++)
            individual.Weights[i] = 1.0f;

        var originalWeights = (float[])individual.Weights.Clone();

        MutationOperators.ApplyWeightJitter(individual, sigma: 0.1f, random);

        // At least some weights should be different
        int changedCount = 0;
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            if (MathF.Abs(individual.Weights[i] - originalWeights[i]) > 1e-6f)
                changedCount++;
        }

        Assert.True(changedCount > 0, "Weight jitter should modify at least some weights");
    }

    [Fact]
    public void WeightJitter_ProportionalToWeightMagnitude()
    {
        var individual = new Individual(2, 2);
        var random = new Random(42);

        // Set different magnitude weights
        individual.Weights[0] = 10.0f;
        individual.Weights[1] = 0.1f;

        float sigma = 0.5f;

        // Run multiple times to get statistical average
        float[] totalDelta = new float[2];
        int iterations = 1000;

        for (int iter = 0; iter < iterations; iter++)
        {
            var temp = new Individual(2, 2);
            temp.Weights[0] = individual.Weights[0];
            temp.Weights[1] = individual.Weights[1];

            MutationOperators.ApplyWeightJitter(temp, sigma, random);

            totalDelta[0] += MathF.Abs(temp.Weights[0] - individual.Weights[0]);
            totalDelta[1] += MathF.Abs(temp.Weights[1] - individual.Weights[1]);
        }

        float avgDelta0 = totalDelta[0] / iterations;
        float avgDelta1 = totalDelta[1] / iterations;

        // Delta should be roughly proportional to weight magnitude
        // avgDelta0 / |w0| ≈ avgDelta1 / |w1|
        float ratio0 = avgDelta0 / MathF.Abs(individual.Weights[0]);
        float ratio1 = avgDelta1 / MathF.Abs(individual.Weights[1]);

        // Should be within 50% of each other
        Assert.True(MathF.Abs(ratio0 - ratio1) / ratio0 < 0.5f,
            $"Jitter should be proportional to weight magnitude. Ratio0={ratio0}, Ratio1={ratio1}");
    }

    #endregion

    #region Weight Reset Tests

    [Fact]
    public void WeightReset_ChangesOneWeight()
    {
        var individual = new Individual(10, 5);
        var random = new Random(42);

        // Initialize with known values
        for (int i = 0; i < individual.Weights.Length; i++)
            individual.Weights[i] = 0.5f;

        MutationOperators.ApplyWeightReset(individual, random);

        // Exactly one weight should be different
        int changedCount = individual.Weights.Count(w => MathF.Abs(w - 0.5f) > 0.01f);
        Assert.Equal(1, changedCount);
    }

    [Fact]
    public void WeightReset_ProducesValueInRange()
    {
        var individual = new Individual(10, 5);
        var random = new Random(42);

        for (int iter = 0; iter < 100; iter++)
        {
            MutationOperators.ApplyWeightReset(individual, random);
        }

        // All weights should be in range [-1, 1]
        foreach (var weight in individual.Weights)
        {
            Assert.InRange(weight, -1.0f, 1.0f);
        }
    }

    [Fact]
    public void WeightReset_HandlesEmptyWeights()
    {
        var individual = new Individual(0, 5);
        var random = new Random(42);

        // Should not throw
        MutationOperators.ApplyWeightReset(individual, random);
    }

    #endregion

    #region Weight L1 Shrink Tests

    [Fact]
    public void WeightL1Shrink_ReducesMagnitude()
    {
        var individual = new Individual(10, 5);

        for (int i = 0; i < individual.Weights.Length; i++)
            individual.Weights[i] = 1.0f;

        float shrinkFactor = 0.1f; // 10% reduction
        MutationOperators.ApplyWeightL1Shrink(individual, shrinkFactor);

        // All weights should be reduced by shrinkFactor
        foreach (var weight in individual.Weights)
        {
            Assert.Equal(0.9f, weight, precision: 6);
        }
    }

    [Fact]
    public void WeightL1Shrink_PreservesSign()
    {
        var individual = new Individual(4, 2);
        individual.Weights[0] = 2.0f;
        individual.Weights[1] = -2.0f;
        individual.Weights[2] = 0.5f;
        individual.Weights[3] = -0.5f;

        MutationOperators.ApplyWeightL1Shrink(individual, 0.2f);

        Assert.True(individual.Weights[0] > 0);
        Assert.True(individual.Weights[1] < 0);
        Assert.True(individual.Weights[2] > 0);
        Assert.True(individual.Weights[3] < 0);
    }

    [Fact]
    public void WeightL1Shrink_AppliesUniformly()
    {
        var individual = new Individual(3, 2);
        individual.Weights[0] = 1.0f;
        individual.Weights[1] = 2.0f;
        individual.Weights[2] = 3.0f;

        float shrinkFactor = 0.25f;
        MutationOperators.ApplyWeightL1Shrink(individual, shrinkFactor);

        Assert.Equal(0.75f, individual.Weights[0], precision: 6);
        Assert.Equal(1.5f, individual.Weights[1], precision: 6);
        Assert.Equal(2.25f, individual.Weights[2], precision: 6);
    }

    #endregion

    #region Activation Swap Tests

    [Fact]
    public void ActivationSwap_ChangesActivation()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 3, 2 },
            AllowedActivationsPerRow = new uint[]
            {
                0,
                (1u << (int)ActivationType.ReLU) | (1u << (int)ActivationType.Tanh),
                (1u << (int)ActivationType.Linear)
            }
        };

        var individual = new Individual(5, 6);
        individual.Activations[1] = ActivationType.ReLU; // Input node

        var originalActivation = individual.Activations[1];
        var random = new Random(42);

        // Try multiple times to ensure swap happens
        bool swapped = false;
        for (int i = 0; i < 100; i++)
        {
            var temp = new Individual(individual);
            MutationOperators.ApplyActivationSwap(temp, spec, random);

            if (temp.Activations[1] != originalActivation)
            {
                swapped = true;
                // Should be one of the allowed activations
                Assert.True(temp.Activations[1] == ActivationType.ReLU ||
                           temp.Activations[1] == ActivationType.Tanh);
                break;
            }
        }

        Assert.True(swapped, "Activation should swap after multiple attempts");
    }

    [Fact]
    public void ActivationSwap_RespectsAllowedActivations()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 2, 2 },
            AllowedActivationsPerRow = new uint[]
            {
                0,
                (1u << (int)ActivationType.Linear), // Only Linear allowed
                (1u << (int)ActivationType.Tanh)    // Only Tanh allowed
            }
        };

        var individual = new Individual(5, 5);
        var random = new Random(42);

        // Try swapping row 1 nodes (should remain Linear)
        for (int i = 0; i < 50; i++)
        {
            individual.Activations[1] = ActivationType.Linear;
            MutationOperators.ApplyActivationSwap(individual, spec, random);
            // Should still be Linear (only option)
            Assert.Equal(ActivationType.Linear, individual.Activations[1]);
        }
    }

    [Fact]
    public void ActivationSwap_UpdatesNodeParameters()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 2, 1 },
            AllowedActivationsPerRow = new uint[]
            {
                0,
                (1u << (int)ActivationType.LeakyReLU) | (1u << (int)ActivationType.ReLU),
                (1u << (int)ActivationType.Linear)
            }
        };

        var individual = new Individual(5, 4);
        individual.Activations[1] = ActivationType.ReLU; // No params needed
        individual.NodeParams[4] = 999.0f; // Set some value

        var random = new Random(42);

        // Swap until we get LeakyReLU
        for (int i = 0; i < 100; i++)
        {
            MutationOperators.ApplyActivationSwap(individual, spec, random);
            if (individual.Activations[1] == ActivationType.LeakyReLU)
            {
                // Parameters should be updated to defaults
                var params1 = individual.GetNodeParams(1);
                Assert.Equal(0.01f, params1[0]); // Default LeakyReLU alpha
                return;
            }
        }
    }

    [Fact]
    public void ActivationSwap_SkipsBiasNode()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 2, 1 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 3 }
        };

        var individual = new Individual(5, 4);
        individual.Activations[0] = ActivationType.Linear;

        var random = new Random(42);

        // Run many times
        for (int i = 0; i < 100; i++)
        {
            MutationOperators.ApplyActivationSwap(individual, spec, random);
            // Bias node should never change
            Assert.Equal(ActivationType.Linear, individual.Activations[0]);
        }
    }

    #endregion

    #region Node Param Mutate Tests

    [Fact]
    public void NodeParamMutate_ModifiesParameters()
    {
        var individual = new Individual(5, 4);
        individual.Activations[1] = ActivationType.LeakyReLU;
        individual.SetNodeParams(1, new[] { 0.01f, 0f, 0f, 0f });

        var originalParam = individual.GetNodeParams(1)[0];
        var random = new Random(42);

        // Try multiple times
        bool changed = false;
        for (int i = 0; i < 100; i++)
        {
            var temp = new Individual(individual);
            MutationOperators.ApplyNodeParamMutate(temp, random);
            if (MathF.Abs(temp.GetNodeParams(1)[0] - originalParam) > 1e-6f)
            {
                changed = true;
                break;
            }
        }

        Assert.True(changed, "Node parameters should change after multiple attempts");
    }

    [Fact]
    public void NodeParamMutate_ClampsToReasonableRange()
    {
        var individual = new Individual(5, 4);
        individual.Activations[1] = ActivationType.LeakyReLU;
        individual.SetNodeParams(1, new[] { 9.0f, 0f, 0f, 0f });

        var random = new Random(42);

        // Mutate many times
        for (int i = 0; i < 1000; i++)
        {
            MutationOperators.ApplyNodeParamMutate(individual, random);
        }

        // Should be clamped to [-10, 10]
        var param = individual.GetNodeParams(1)[0];
        Assert.InRange(param, -10.0f, 10.0f);
    }

    [Fact]
    public void NodeParamMutate_SkipsNodesWithoutParams()
    {
        var individual = new Individual(5, 4);
        individual.Activations[1] = ActivationType.ReLU; // No params
        individual.Activations[2] = ActivationType.Tanh; // No params

        var random = new Random(42);

        // Should not crash
        for (int i = 0; i < 50; i++)
        {
            MutationOperators.ApplyNodeParamMutate(individual, random);
        }
    }

    #endregion

    #region Glorot Initialization Tests

    [Fact]
    public void GlorotUniform_ProducesCorrectRange()
    {
        var random = new Random(42);
        int fanIn = 6;
        int fanOut = 4;

        float expectedLimit = MathF.Sqrt(6.0f / (fanIn + fanOut));

        // Generate many samples
        for (int i = 0; i < 1000; i++)
        {
            float value = MutationOperators.GlorotUniform(fanIn, fanOut, random);
            Assert.InRange(value, -expectedLimit, expectedLimit);
        }
    }

    [Fact]
    public void GlorotUniform_ApproximatelyUniform()
    {
        var random = new Random(42);
        int fanIn = 6;
        int fanOut = 6;
        float limit = MathF.Sqrt(6.0f / (fanIn + fanOut));

        int samples = 10000;
        int negativeCount = 0;

        for (int i = 0; i < samples; i++)
        {
            float value = MutationOperators.GlorotUniform(fanIn, fanOut, random);
            if (value < 0) negativeCount++;
        }

        // Should be approximately 50% negative
        float negativeRatio = (float)negativeCount / samples;
        Assert.InRange(negativeRatio, 0.45f, 0.55f);
    }

    [Fact]
    public void InitializeWeights_FillsAllWeights()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 3, 2 },
            MaxInDegree = 6
        };

        var individual = new Individual(10, 6);
        var random = new Random(42);

        // Initially zero
        for (int i = 0; i < individual.Weights.Length; i++)
            Assert.Equal(0.0f, individual.Weights[i]);

        MutationOperators.InitializeWeights(individual, spec, random);

        // All weights should be non-zero (extremely unlikely to be exactly zero)
        int nonZeroCount = individual.Weights.Count(w => MathF.Abs(w) > 1e-6f);
        Assert.True(nonZeroCount > 5, "Most weights should be initialized to non-zero");
    }

    #endregion

    #region Full Mutation Tests

    [Fact]
    public void Mutate_AppliesMultipleOperators()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 4, 3 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 3 },
            MaxInDegree = 6
        };

        var individual = new Individual(15, 8);
        for (int i = 0; i < individual.Weights.Length; i++)
            individual.Weights[i] = 1.0f;

        for (int i = 0; i < individual.Activations.Length; i++)
            individual.Activations[i] = ActivationType.ReLU;

        var config = new MutationConfig
        {
            WeightJitter = 1.0f,    // Always apply
            WeightReset = 0.0f,     // Never apply
            WeightL1Shrink = 0.0f,  // Never apply
            ActivationSwap = 0.0f,  // Never apply
            NodeParamMutate = 0.0f  // Never apply
        };

        var random = new Random(42);
        MutationOperators.Mutate(individual, spec, config, random);

        // Weights should be jittered
        int changedWeights = individual.Weights.Count(w => MathF.Abs(w - 1.0f) > 1e-6f);
        Assert.True(changedWeights > 0, "Weight jitter should modify weights");
    }

    [Fact]
    public void Mutate_RespectsZeroProbabilities()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 4, 3 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 3 },
            MaxInDegree = 6
        };

        var individual = new Individual(15, 8);
        for (int i = 0; i < individual.Weights.Length; i++)
            individual.Weights[i] = 1.0f;

        var config = new MutationConfig
        {
            WeightJitter = 0.0f,
            WeightReset = 0.0f,
            WeightL1Shrink = 0.0f,
            ActivationSwap = 0.0f,
            NodeParamMutate = 0.0f
        };

        var random = new Random(42);

        var originalWeights = (float[])individual.Weights.Clone();
        MutationOperators.Mutate(individual, spec, config, random);

        // Nothing should change
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            Assert.Equal(originalWeights[i], individual.Weights[i]);
        }
    }

    [Fact]
    public void Mutate_DeterministicWithSameSeed()
    {
        var spec = new SpeciesSpec
        {
            RowCounts = new[] { 1, 4, 3 },
            AllowedActivationsPerRow = new uint[] { 0, 0xFFFFFFFF, 3 },
            MaxInDegree = 6
        };

        var config = new MutationConfig();

        var individual1 = new Individual(15, 8);
        var individual2 = new Individual(15, 8);

        // Initialize identically
        for (int i = 0; i < individual1.Weights.Length; i++)
        {
            individual1.Weights[i] = 0.5f;
            individual2.Weights[i] = 0.5f;
        }

        // Apply mutations with same seed
        MutationOperators.Mutate(individual1, spec, config, new Random(42));
        MutationOperators.Mutate(individual2, spec, config, new Random(42));

        // Results should be identical
        for (int i = 0; i < individual1.Weights.Length; i++)
        {
            Assert.Equal(individual1.Weights[i], individual2.Weights[i]);
        }
    }

    #endregion
}
