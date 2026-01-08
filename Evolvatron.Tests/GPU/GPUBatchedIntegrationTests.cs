using Xunit;
using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace Evolvatron.Tests.GPU;

/// <summary>
/// Integration tests for GPU batched fitness evaluation.
/// Verifies the integration of:
/// - Neural network evaluation (GPUEvolvionKernels)
/// - Batched physics (GPUBatchedStepper)
/// - Environment logic (GPUBatchedEnvironmentKernels)
/// </summary>
public class GPUBatchedIntegrationTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly Random _random;

    public GPUBatchedIntegrationTests()
    {
        _context = Context.CreateDefault();
        _accelerator = _context.CreateCPUAccelerator(0);
        _random = new Random(42);
    }

    #region Helper Methods

    /// <summary>
    /// Creates a simple topology with given input, hidden, and output sizes.
    /// For rocket control: 8 inputs (observations), hidden layers, 2 outputs (thrust/gimbal).
    /// </summary>
    private SpeciesSpec CreateTopology(int inputSize, int[] hiddenSizes, int outputSize)
    {
        var rowCounts = new List<int> { inputSize };
        rowCounts.AddRange(hiddenSizes);
        rowCounts.Add(outputSize);

        // Build edges: fully connected between consecutive layers
        var edges = new List<(int Source, int Dest)>();
        int nodeOffset = 0;

        for (int layer = 0; layer < rowCounts.Count - 1; layer++)
        {
            int srcStart = nodeOffset;
            int srcCount = rowCounts[layer];
            int dstStart = nodeOffset + srcCount;
            int dstCount = rowCounts[layer + 1];

            // Connect each source to each destination
            for (int s = 0; s < srcCount; s++)
            {
                for (int d = 0; d < dstCount; d++)
                {
                    edges.Add((srcStart + s, dstStart + d));
                }
            }

            nodeOffset += srcCount;
        }

        // Allowed activations: all for hidden, Linear/Tanh for output
        uint allActivations = 0xFFFF; // All activation types
        uint outputActivations = (1u << (int)ActivationType.Linear) | (1u << (int)ActivationType.Tanh);

        var allowedActivations = new uint[rowCounts.Count];
        for (int i = 0; i < rowCounts.Count - 1; i++)
        {
            allowedActivations[i] = allActivations;
        }
        allowedActivations[^1] = outputActivations;

        var spec = new SpeciesSpec
        {
            RowCounts = rowCounts.ToArray(),
            AllowedActivationsPerRow = allowedActivations,
            Edges = edges,
            MaxInDegree = 100 // Large enough for full connectivity
        };

        spec.BuildRowPlans();
        return spec;
    }

    /// <summary>
    /// Creates an individual with random weights/biases.
    /// </summary>
    private Individual CreateRandomIndividual(SpeciesSpec spec, int seed)
    {
        var rng = new Random(seed);
        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);

        // Random weights in [-1, 1]
        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // Random biases in [-0.5, 0.5]
        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] = (float)(rng.NextDouble() - 0.5);
        }

        // Random activations (hidden layers only)
        int nodeIdx = 0;
        for (int row = 0; row < spec.RowCounts.Length; row++)
        {
            for (int j = 0; j < spec.RowCounts[row]; j++)
            {
                if (row == 0)
                {
                    // Input layer: Linear
                    individual.Activations[nodeIdx] = ActivationType.Linear;
                }
                else if (row == spec.RowCounts.Length - 1)
                {
                    // Output layer: Tanh (bounded output for actions)
                    individual.Activations[nodeIdx] = ActivationType.Tanh;
                }
                else
                {
                    // Hidden layer: random (prefer ReLU, Tanh, Sigmoid for stability)
                    var choices = new[] { ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid };
                    individual.Activations[nodeIdx] = choices[rng.Next(choices.Length)];
                }
                nodeIdx++;
            }
        }

        // Node params (for LeakyReLU/ELU)
        for (int i = 0; i < individual.NodeParams.Length; i++)
        {
            individual.NodeParams[i] = 0.01f; // Default alpha
        }

        return individual;
    }

    /// <summary>
    /// Creates individuals with specific weight patterns for determinism testing.
    /// </summary>
    private Individual CreateDeterministicIndividual(SpeciesSpec spec, float weightValue)
    {
        var individual = new Individual(spec.TotalEdges, spec.TotalNodes);

        for (int i = 0; i < individual.Weights.Length; i++)
        {
            individual.Weights[i] = weightValue;
        }

        for (int i = 0; i < individual.Biases.Length; i++)
        {
            individual.Biases[i] = 0.0f;
        }

        for (int i = 0; i < individual.Activations.Length; i++)
        {
            individual.Activations[i] = ActivationType.Tanh;
        }

        for (int i = 0; i < individual.NodeParams.Length; i++)
        {
            individual.NodeParams[i] = 0.01f;
        }

        return individual;
    }

    /// <summary>
    /// Evaluates individuals using GPUEvaluator with XOR environment (simple integration test).
    /// </summary>
    private float[] EvaluateWithXOR(SpeciesSpec spec, List<Individual> individuals, int episodesPerIndividual, int seed)
    {
        using var evaluator = new GPUEvaluator(maxIndividuals: individuals.Count);
        evaluator.Initialize(spec, individuals);
        return evaluator.EvaluateWithXOR(spec, individuals, episodesPerIndividual, seed);
    }

    /// <summary>
    /// Evaluates individuals using GPUEvaluator with landscape environment.
    /// </summary>
    private float[] EvaluateWithLandscape(SpeciesSpec spec, List<Individual> individuals, GPULandscapeConfig config, int episodesPerIndividual, int seed)
    {
        using var evaluator = new GPUEvaluator(maxIndividuals: individuals.Count);
        evaluator.Initialize(spec, individuals);
        return evaluator.EvaluateWithEnvironment(spec, individuals, config, episodesPerIndividual, seed);
    }

    #endregion

    #region Basic Sanity Tests

    [Fact]
    public void Evaluator_CanEvaluateSingleIndividual()
    {
        // Arrange: Create simple topology (2 inputs -> 4 hidden -> 1 output) for XOR
        var spec = CreateTopology(inputSize: 2, hiddenSizes: new[] { 4 }, outputSize: 1);
        var individual = CreateRandomIndividual(spec, seed: 123);
        var individuals = new List<Individual> { individual };

        // Act: Evaluate with XOR environment
        var fitnessValues = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 1, seed: 42);

        // Assert: Should return one fitness value
        Assert.Single(fitnessValues);
        Assert.False(float.IsNaN(fitnessValues[0]), "Fitness should not be NaN");
        Assert.False(float.IsInfinity(fitnessValues[0]), "Fitness should not be infinite");

        // XOR fitness is negative error, so should be in [-4, 0] range
        // (worst case: all wrong with max error ~4 total)
        Assert.InRange(fitnessValues[0], -4.0f, 0.1f);
    }

    [Fact]
    public void Evaluator_CanEvaluateMultipleIndividuals()
    {
        // Arrange: Create topology for XOR problem
        var spec = CreateTopology(inputSize: 2, hiddenSizes: new[] { 6 }, outputSize: 1);
        var individuals = new List<Individual>();

        for (int i = 0; i < 10; i++)
        {
            individuals.Add(CreateRandomIndividual(spec, seed: i * 100));
        }

        // Act: Evaluate batch
        var fitnessValues = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 1, seed: 42);

        // Assert: Should return correct number of fitness values
        Assert.Equal(10, fitnessValues.Length);

        foreach (var fitness in fitnessValues)
        {
            Assert.False(float.IsNaN(fitness), "Fitness should not be NaN");
            Assert.False(float.IsInfinity(fitness), "Fitness should not be infinite");
        }

        // At least some variation in fitness values (different random networks)
        var uniqueValues = fitnessValues.Distinct().Count();
        Assert.True(uniqueValues > 1, "Different networks should have different fitness values");
    }

    #endregion

    #region Determinism Tests

    [Fact]
    public void Evaluator_ProducesDeterministicResults()
    {
        // Arrange: Create topology and individuals
        var spec = CreateTopology(inputSize: 2, hiddenSizes: new[] { 4, 4 }, outputSize: 1);
        var individuals = new List<Individual>();

        for (int i = 0; i < 5; i++)
        {
            individuals.Add(CreateRandomIndividual(spec, seed: i * 50));
        }

        int evalSeed = 12345;

        // Act: Evaluate twice with same seed
        var fitnessValues1 = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 2, seed: evalSeed);
        var fitnessValues2 = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 2, seed: evalSeed);

        // Assert: Results should be identical
        Assert.Equal(fitnessValues1.Length, fitnessValues2.Length);

        for (int i = 0; i < fitnessValues1.Length; i++)
        {
            Assert.Equal(fitnessValues1[i], fitnessValues2[i], precision: 6);
        }
    }

    [Fact]
    public void Evaluator_DifferentSeedsProduceDifferentResults()
    {
        // Arrange
        var spec = CreateTopology(inputSize: 2, hiddenSizes: new[] { 5 }, outputSize: 2);

        // Use landscape environment where seed affects starting position
        var config = new GPULandscapeConfig(
            dimensions: 2,
            maxSteps: 10,
            stepSize: 0.1f,
            minBound: -5f,
            maxBound: 5f,
            landscapeType: 0); // Sphere

        var individuals = new List<Individual>();
        for (int i = 0; i < 3; i++)
        {
            individuals.Add(CreateRandomIndividual(spec, seed: i * 10));
        }

        // Act
        var fitness1 = EvaluateWithLandscape(spec, individuals, config, episodesPerIndividual: 2, seed: 100);
        var fitness2 = EvaluateWithLandscape(spec, individuals, config, episodesPerIndividual: 2, seed: 200);

        // Assert: Different seeds should (likely) produce different results
        bool anyDifferent = false;
        for (int i = 0; i < fitness1.Length; i++)
        {
            if (MathF.Abs(fitness1[i] - fitness2[i]) > 1e-5f)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Different seeds should produce different fitness values due to different starting positions");
    }

    #endregion

    #region Network Differentiation Tests

    [Fact]
    public void Evaluator_DifferentNetworksDifferentFitness()
    {
        // Arrange: Create topology and individuals with very different weights
        var spec = CreateTopology(inputSize: 2, hiddenSizes: new[] { 8 }, outputSize: 1);

        var individuals = new List<Individual>
        {
            CreateDeterministicIndividual(spec, weightValue: 0.1f),
            CreateDeterministicIndividual(spec, weightValue: 0.5f),
            CreateDeterministicIndividual(spec, weightValue: 1.0f),
            CreateDeterministicIndividual(spec, weightValue: -0.5f),
            CreateDeterministicIndividual(spec, weightValue: 2.0f)
        };

        // Act
        var fitnessValues = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 1, seed: 42);

        // Assert: Networks with different weights should have different fitness
        Assert.Equal(5, fitnessValues.Length);

        var uniqueFitness = fitnessValues.Select(f => MathF.Round(f, 4)).Distinct().Count();
        Assert.True(uniqueFitness >= 2, "Networks with very different weights should have different fitness values");
    }

    [Fact]
    public void Evaluator_IdenticalNetworksIdenticalFitness()
    {
        // Arrange: Create identical networks
        var spec = CreateTopology(inputSize: 2, hiddenSizes: new[] { 4 }, outputSize: 1);

        var template = CreateDeterministicIndividual(spec, weightValue: 0.3f);
        var individuals = new List<Individual>
        {
            new Individual(template),
            new Individual(template),
            new Individual(template)
        };

        // Act
        var fitnessValues = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 1, seed: 42);

        // Assert: Identical networks should have identical fitness
        Assert.Equal(3, fitnessValues.Length);
        Assert.Equal(fitnessValues[0], fitnessValues[1], precision: 6);
        Assert.Equal(fitnessValues[1], fitnessValues[2], precision: 6);
    }

    #endregion

    #region Scaling Tests

    [Fact]
    public void Evaluator_ScalesTo100Individuals()
    {
        // Arrange: Create larger batch
        var spec = CreateTopology(inputSize: 8, hiddenSizes: new[] { 16, 8 }, outputSize: 2);
        var individuals = new List<Individual>();

        for (int i = 0; i < 100; i++)
        {
            individuals.Add(CreateRandomIndividual(spec, seed: i));
        }

        // Act & Assert: Should complete without error
        var fitnessValues = EvaluateWithXOR(spec, individuals, episodesPerIndividual: 1, seed: 42);

        Assert.Equal(100, fitnessValues.Length);

        // Verify all values are valid
        foreach (var fitness in fitnessValues)
        {
            Assert.False(float.IsNaN(fitness), "Fitness should not be NaN");
            Assert.False(float.IsInfinity(fitness), "Fitness should not be infinite");
        }
    }

    [Fact]
    public void Evaluator_HandlesLargerTopology()
    {
        // Arrange: Create deeper network (more like rocket control)
        var spec = CreateTopology(
            inputSize: 8,      // Observations: direction, velocity, up vector, distance, angular vel
            hiddenSizes: new[] { 16, 16 },
            outputSize: 2);    // Actions: throttle, gimbal

        var individuals = new List<Individual>();
        for (int i = 0; i < 20; i++)
        {
            individuals.Add(CreateRandomIndividual(spec, seed: i * 7));
        }

        // Use landscape environment (simulates position-based control)
        var config = new GPULandscapeConfig(
            dimensions: 2,
            maxSteps: 50,
            stepSize: 0.2f,
            minBound: -10f,
            maxBound: 10f,
            landscapeType: 0); // Sphere

        // Act
        var fitnessValues = EvaluateWithLandscape(spec, individuals, config, episodesPerIndividual: 3, seed: 123);

        // Assert
        Assert.Equal(20, fitnessValues.Length);

        foreach (var fitness in fitnessValues)
        {
            Assert.False(float.IsNaN(fitness), "Fitness should not be NaN");
            Assert.False(float.IsInfinity(fitness), "Fitness should not be infinite");
        }
    }

    #endregion

    #region Integration with Batched Physics Components

    [Fact]
    public void BatchedPhysics_IntegratesWithEnvironmentKernels()
    {
        // This test verifies that the GPUBatchedStepper and GPUBatchedEnvironmentKernels
        // work together correctly (through GPUBatchedEnvironment)

        int worldCount = 10;
        var worldConfig = GPUBatchedWorldConfig.ForRocketChase(worldCount: worldCount);
        var envConfig = GPUBatchedEnvironmentConfig.ForTargetChase(worldCount: worldCount);

        using var env = new GPUBatchedEnvironment(_accelerator, worldConfig, envConfig);

        // Setup rocket template
        var templateBodies = new GPURigidBody[worldConfig.RigidBodiesPerWorld];
        templateBodies[0] = new GPURigidBody
        {
            X = 0f, Y = 5f, Angle = 0f,
            VelX = 0f, VelY = 0f, AngularVel = 0f,
            InvMass = 0.2f, InvInertia = 0.5f
        };
        templateBodies[1] = new GPURigidBody { X = -0.5f, Y = 4f, InvMass = 1f, InvInertia = 2f };
        templateBodies[2] = new GPURigidBody { X = 0.5f, Y = 4f, InvMass = 1f, InvInertia = 2f };

        var templateGeoms = new GPURigidBodyGeom[worldConfig.GeomsPerWorld];
        var templateJoints = new GPURevoluteJoint[worldConfig.JointsPerWorld];

        env.UploadRocketTemplate(templateBodies, templateGeoms, templateJoints);

        // Setup colliders
        var colliders = new GPUOBBCollider[worldConfig.SharedColliderCount];
        colliders[0] = new GPUOBBCollider { CX = 0f, CY = -5f, UX = 1f, UY = 0f, HalfExtentX = 20f, HalfExtentY = 0.5f };
        env.UploadSharedColliders(colliders);

        // Reset and run
        env.Reset(baseSeed: 42);

        var simConfig = new SimulationConfig { Dt = 1f / 120f, GravityY = -10f, XpbdIterations = 4 };

        // Run for a few steps
        for (int i = 0; i < 50; i++)
        {
            // Get observations
            var observations = env.GetObservations();
            Assert.Equal(envConfig.TotalObservations, observations.Length);

            // Apply random actions
            var actions = new float[envConfig.TotalActions];
            for (int j = 0; j < actions.Length; j++)
            {
                actions[j] = (float)(_random.NextDouble() * 2.0 - 1.0);
            }
            env.UploadActions(actions);

            // Step
            env.Step(simConfig);

            if (env.AllTerminal())
                break;
        }

        // Compute and verify fitness
        env.ComputeFitness();
        var fitness = env.GetFitness();

        Assert.Equal(worldCount, fitness.Length);
        foreach (var f in fitness)
        {
            Assert.False(float.IsNaN(f), "Fitness should not be NaN");
            Assert.False(float.IsInfinity(f), "Fitness should not be infinite");
        }
    }

    [Fact]
    public void BatchedEnvironment_ResetProducesDifferentTargets()
    {
        int worldCount = 5;
        var worldConfig = GPUBatchedWorldConfig.ForRocketChase(worldCount: worldCount);
        var envConfig = GPUBatchedEnvironmentConfig.ForTargetChase(worldCount: worldCount);

        using var env = new GPUBatchedEnvironment(_accelerator, worldConfig, envConfig);

        // Setup minimal template
        var templateBodies = new GPURigidBody[worldConfig.RigidBodiesPerWorld];
        templateBodies[0] = new GPURigidBody { X = 0f, Y = 0f, InvMass = 1f, InvInertia = 1f };
        env.UploadRocketTemplate(templateBodies, new GPURigidBodyGeom[worldConfig.GeomsPerWorld], new GPURevoluteJoint[worldConfig.JointsPerWorld]);
        env.UploadSharedColliders(new GPUOBBCollider[worldConfig.SharedColliderCount]);

        // Reset with seed 1
        env.Reset(baseSeed: 100);
        var obs1 = env.GetObservations().ToArray();

        // Reset with seed 2
        env.Reset(baseSeed: 200);
        var obs2 = env.GetObservations().ToArray();

        // Observations should differ due to different target positions
        bool anyDifferent = false;
        for (int i = 0; i < obs1.Length; i++)
        {
            if (MathF.Abs(obs1[i] - obs2[i]) > 1e-5f)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Different seeds should produce different target positions and thus different observations");
    }

    #endregion

    public void Dispose()
    {
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
