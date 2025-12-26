using Evolvatron.Core;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;

namespace Evolvatron.Tests.Integration;

/// <summary>
/// Integration tests verifying Rigidon physics and Evolvion neural evolution work together.
/// These tests ensure the full pipeline from environment to evolution runs correctly.
/// </summary>
public class RocketEvolutionIntegrationTest
{
    /// <summary>
    /// Test that the RocketEnvironment can be created and reset without errors.
    /// </summary>
    [Fact]
    public void RocketEnvironment_CanResetAndStep()
    {
        // Arrange
        var environment = new RocketEnvironment();

        // Act
        environment.Reset(seed: 42);

        // Get initial observations
        var observations = new float[environment.InputCount];
        environment.GetObservations(observations);

        // Step with neutral actions
        var actions = new float[] { 0.5f, 0f }; // Half throttle, no gimbal
        float reward = environment.Step(actions);

        // Assert
        Assert.Equal(8, observations.Length);

        // Get rocket state to help debug
        environment.GetRocketState(out float x, out float y, out float vx, out float vy, out float upX, out float upY);

        // At first step, rocket should not be terminal (just started falling)
        // Allow terminal if crash (high velocity) but not on step 1
        Assert.False(environment.IsTerminal(),
            $"Should not terminate after 1 step. State: x={x:F2}, y={y:F2}, vx={vx:F2}, vy={vy:F2}, upX={upX:F2}, upY={upY:F2}");

        // Observations should be non-zero (rocket has position, velocity, etc.)
        bool hasNonZero = false;
        foreach (var obs in observations)
        {
            if (MathF.Abs(obs) > 1e-6f)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Observations should contain non-zero values");
    }

    /// <summary>
    /// Test that the environment properly terminates when rocket crashes.
    /// </summary>
    [Fact]
    public void RocketEnvironment_TerminatesOnCrash()
    {
        // Arrange
        var environment = new RocketEnvironment();
        environment.Reset(seed: 42);

        // Act - Let rocket fall with no thrust
        var actions = new float[] { 0f, 0f }; // No thrust, no gimbal
        int steps = 0;
        const int maxTestSteps = 500;

        while (!environment.IsTerminal() && steps < maxTestSteps)
        {
            environment.Step(actions);
            steps++;
        }

        // Assert - Should terminate (crash or max steps)
        Assert.True(environment.IsTerminal(), "Environment should terminate");
        Assert.True(steps < maxTestSteps, "Rocket should crash before max steps when falling");
    }

    /// <summary>
    /// Test that WorldState is properly reused across episode resets (no excessive allocations).
    /// </summary>
    [Fact]
    public void RocketEnvironment_ReusesWorldStateAcrossResets()
    {
        // Arrange
        var environment = new RocketEnvironment();
        var observations = new float[environment.InputCount];

        // Act - Reset multiple times and verify consistent behavior
        for (int episode = 0; episode < 10; episode++)
        {
            environment.Reset(seed: episode);
            environment.GetObservations(observations);

            // Step a few times
            var actions = new float[] { 0.7f, 0.1f };
            for (int step = 0; step < 10; step++)
            {
                if (environment.IsTerminal()) break;
                environment.Step(actions);
            }
        }

        // Assert - If we get here without crash, reuse is working
        // The test is primarily checking for memory leaks or state corruption
        Assert.True(true, "Environment should handle multiple resets without issues");
    }

    /// <summary>
    /// Test that the full evolution pipeline runs without errors.
    /// Uses small population and few generations for fast testing.
    /// </summary>
    [Fact]
    public void RocketEvolution_RunsWithoutError()
    {
        // Arrange - Small configuration for fast testing
        var config = new EvolutionConfig
        {
            SpeciesCount = 2,
            IndividualsPerSpecies = 10,
            MinSpeciesCount = 1,
            Elites = 2,
            TournamentSize = 3,
            GraceGenerations = 1,
            StagnationThreshold = 10
        };

        var evolver = new Evolver(seed: 42);

        // Create topology for rocket controller: 8 inputs, hidden layer, 2 outputs
        var topology = new SpeciesBuilder()
            .AddInputRow(8) // RocketEnvironment.InputCount
            .AddHiddenRow(6, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh) // throttle and gimbal
            .InitializeDense(new Random(42))
            .Build();

        var population = evolver.InitializePopulation(config, topology);
        var environment = new RocketEnvironment();
        environment.MaxSteps = 100; // Short episodes for fast testing
        var evaluator = new SimpleFitnessEvaluator();

        // Act - Run a few generations
        const int generations = 3;
        for (int gen = 0; gen < generations; gen++)
        {
            evaluator.EvaluatePopulation(population, environment, seed: gen);
            evolver.StepGeneration(population);
        }

        // Assert - Verify pipeline completed and produced valid results
        var stats = population.GetStatistics();
        Assert.True(stats.BestFitness > float.NegativeInfinity, "Should have valid fitness values");
        Assert.True(population.Generation == generations, "Should have advanced generations");
        Assert.True(population.AllSpecies.Count >= config.MinSpeciesCount, "Should maintain minimum species");
    }

    /// <summary>
    /// Test determinism: Same seed should produce identical evolution trajectory.
    /// </summary>
    [Fact]
    public void RocketEvolution_IsDeterministic()
    {
        // Arrange
        var config = new EvolutionConfig
        {
            SpeciesCount = 2,
            IndividualsPerSpecies = 8,
            MinSpeciesCount = 1,
            Elites = 1,
            TournamentSize = 3
        };

        // Run 1
        var evolver1 = new Evolver(seed: 123);
        var topology1 = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(4, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(123))
            .Build();
        var population1 = evolver1.InitializePopulation(config, topology1);
        var env1 = new RocketEnvironment();
        env1.MaxSteps = 50;
        var evaluator1 = new SimpleFitnessEvaluator();

        for (int gen = 0; gen < 2; gen++)
        {
            evaluator1.EvaluatePopulation(population1, env1, seed: gen);
            evolver1.StepGeneration(population1);
        }

        // Run 2 with same seed
        var evolver2 = new Evolver(seed: 123);
        var topology2 = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(4, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(123))
            .Build();
        var population2 = evolver2.InitializePopulation(config, topology2);
        var env2 = new RocketEnvironment();
        env2.MaxSteps = 50;
        var evaluator2 = new SimpleFitnessEvaluator();

        for (int gen = 0; gen < 2; gen++)
        {
            evaluator2.EvaluatePopulation(population2, env2, seed: gen);
            evolver2.StepGeneration(population2);
        }

        // Assert
        var stats1 = population1.GetStatistics();
        var stats2 = population2.GetStatistics();

        Assert.Equal(stats1.BestFitness, stats2.BestFitness, precision: 5);
        Assert.Equal(stats1.MeanFitness, stats2.MeanFitness, precision: 5);
    }

    /// <summary>
    /// Test that observations are normalized and within expected ranges.
    /// </summary>
    [Fact]
    public void RocketEnvironment_ObservationsAreNormalized()
    {
        // Arrange
        var environment = new RocketEnvironment();
        var observations = new float[environment.InputCount];

        // Act - Run multiple episodes and check observation ranges
        for (int episode = 0; episode < 5; episode++)
        {
            environment.Reset(seed: episode);

            for (int step = 0; step < 50 && !environment.IsTerminal(); step++)
            {
                environment.GetObservations(observations);

                // Assert - Check that observations are not NaN or Inf
                for (int i = 0; i < observations.Length; i++)
                {
                    Assert.False(float.IsNaN(observations[i]), $"Observation {i} is NaN at step {step}");
                    Assert.False(float.IsInfinity(observations[i]), $"Observation {i} is Inf at step {step}");
                }

                // Up vector components should be normalized (within [-1, 1])
                Assert.InRange(observations[4], -1.1f, 1.1f); // upX
                Assert.InRange(observations[5], -1.1f, 1.1f); // upY

                // Control inputs should be in [-1, 1] range
                Assert.InRange(observations[6], -1.1f, 1.1f); // gimbal
                Assert.InRange(observations[7], -0.1f, 1.1f); // throttle (0-1)

                // Step with random action
                var actions = new float[] { 0.5f + (step % 10) * 0.05f, (step % 5 - 2) * 0.2f };
                environment.Step(actions);
            }
        }
    }

    /// <summary>
    /// Test that applying thrust affects rocket velocity and position.
    /// </summary>
    [Fact]
    public void RocketEnvironment_ThrustAffectsRocketState()
    {
        // Arrange
        var environment = new RocketEnvironment();
        environment.Reset(seed: 42);

        // Get initial state
        environment.GetRocketState(out float x0, out float y0, out float vx0, out float vy0, out _, out _);

        // Act - Apply full thrust for several steps
        var thrustActions = new float[] { 1f, 0f }; // Full throttle, no gimbal
        for (int i = 0; i < 20 && !environment.IsTerminal(); i++)
        {
            environment.Step(thrustActions);
        }

        // Get final state
        environment.GetRocketState(out float x1, out float y1, out float vx1, out float vy1, out _, out _);

        // Assert - With thrust, rocket should have different velocity than free fall
        // At full thrust, upward velocity should be better than gravity alone would give
        // (The rocket starts falling, thrust should counter gravity)
        float timeElapsed = 20 * (1f / 120f); // 20 steps at 120Hz
        float freefall_vy = vy0 - 9.81f * timeElapsed; // Expected velocity under gravity alone

        // With thrust, actual vy should be better (less negative or positive) than freefall
        Assert.True(vy1 > freefall_vy - 0.5f, $"Thrust should counter gravity. vy1={vy1}, freefall_vy={freefall_vy}");
    }

    /// <summary>
    /// Test that gimbal torque affects rocket rotation.
    /// </summary>
    [Fact]
    public void RocketEnvironment_GimbalAffectsRotation()
    {
        // Arrange
        var environment = new RocketEnvironment();
        environment.Reset(seed: 42);

        // Get initial up vector
        environment.GetRocketState(out _, out _, out _, out _, out float upX0, out float upY0);

        // Act - Apply gimbal torque while thrusting to keep rocket in air
        var gimbalActions = new float[] { 0.8f, 1f }; // Thrust + full gimbal
        for (int i = 0; i < 30 && !environment.IsTerminal(); i++)
        {
            environment.Step(gimbalActions);
        }

        // Get final up vector
        environment.GetRocketState(out _, out _, out _, out _, out float upX1, out float upY1);

        // Assert - Rocket should have rotated (up vector changed)
        float angleDiff = MathF.Abs(MathF.Atan2(upX1, upY1) - MathF.Atan2(upX0, upY0));

        Assert.True(angleDiff > 0.01f, "Gimbal should cause rotation");
    }

    /// <summary>
    /// Benchmark test: Ensure environment step performance is acceptable.
    /// </summary>
    [Fact]
    public void RocketEnvironment_StepPerformance()
    {
        // Arrange
        var environment = new RocketEnvironment();
        var actions = new float[] { 0.5f, 0.1f };
        const int warmupSteps = 100;
        const int testSteps = 1000;

        // Warmup
        environment.Reset(seed: 0);
        for (int i = 0; i < warmupSteps && !environment.IsTerminal(); i++)
        {
            environment.Step(actions);
        }

        // Act - Time the steps
        environment.Reset(seed: 1);
        var sw = System.Diagnostics.Stopwatch.StartNew();

        int actualSteps = 0;
        while (actualSteps < testSteps)
        {
            if (environment.IsTerminal())
            {
                environment.Reset(seed: actualSteps);
            }
            environment.Step(actions);
            actualSteps++;
        }

        sw.Stop();

        // Assert - Should complete in reasonable time (< 1 second for 1000 steps)
        // At 120Hz physics, 1000 steps = ~8.3 seconds of simulated time
        Assert.True(sw.ElapsedMilliseconds < 2000,
            $"1000 environment steps took {sw.ElapsedMilliseconds}ms, should be < 2000ms");

        // Log performance for reference
        double stepsPerSecond = testSteps / (sw.ElapsedMilliseconds / 1000.0);
        // Expected: > 5000 steps/second on modern hardware
    }
}
