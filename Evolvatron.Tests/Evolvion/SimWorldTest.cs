using System.Diagnostics;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.GPU.MegaKernel;
using Evolvatron.Evolvion.World;

namespace Evolvatron.Tests.Evolvion;

/// <summary>
/// Phase 1+2+3 tests: SimWorld JSON loading, evaluator configuration, parameterized fitness, reward zones.
/// </summary>
public class SimWorldTest
{
    /// <summary>
    /// JSON matching the editor's default world export.
    /// </summary>
    private const string DefaultWorldJson = """
    {
      "GroundY": -5,
      "LandingPad": {
        "PadX": 0,
        "PadY": -4.5,
        "PadHalfWidth": 2,
        "PadHalfHeight": 0.25,
        "LandingBonus": 100,
        "MaxLandingVelocity": 2.0,
        "MaxLandingAngle": 15
      },
      "Spawn": {
        "X": 0,
        "Y": 15,
        "XRange": 8,
        "HeightRange": 2,
        "AngleRange": 30,
        "VelXRange": 2,
        "VelYMax": 3
      },
      "Obstacles": [
        {
          "CX": -9.517,
          "CY": 9.467,
          "UX": 1.0,
          "UY": 0.0,
          "HalfExtentX": 2.583,
          "HalfExtentY": 2.267,
          "IsLethal": false
        }
      ],
      "Checkpoints": [
        { "X": 3.7, "Y": 10.6, "Radius": 1.447, "Order": 1, "RewardBonus": 20, "InfluenceRadius": 2 },
        { "X": -1.933, "Y": 9.583, "Radius": 1.792, "Order": 0, "RewardBonus": 20, "InfluenceRadius": 2 }
      ],
      "SpeedZones": [
        { "X": 6.383, "Y": 7.783, "HalfExtentX": 1.25, "HalfExtentY": 1.783, "MaxSpeed": 5, "RewardPerStep": 0.1 }
      ],
      "DangerZones": [
        { "X": -4.617, "Y": 3.517, "HalfExtentX": 0.75, "HalfExtentY": 1.083, "PenaltyPerStep": 10, "IsLethal": false, "InfluenceRadius": 3 }
      ],
      "Attractors": [
        { "X": 2.683, "Y": 1.15, "HalfExtentX": 1.883, "HalfExtentY": 1.95, "Magnitude": 10, "InfluenceRadius": 3, "ContactBonus": 50 }
      ],
      "SimulationConfig": {
        "Dt": 0.008333333,
        "GravityY": -9.81,
        "FrictionMu": 0.8,
        "Restitution": 0.0,
        "GlobalDamping": 0.02,
        "AngularDamping": 0.1,
        "SolverIterations": 6,
        "MaxThrust": 200,
        "MaxGimbalAngle": 15,
        "SensorCount": 4,
        "MaxSteps": 600
      },
      "RewardWeights": {
        "PositionWeight": 1.0,
        "VelocityWeight": 0.5,
        "AngleWeight": 0.3,
        "AngularVelocityWeight": 0.1,
        "ControlEffortWeight": 0.05
      }
    }
    """;

    [Fact]
    public void FromJson_DeserializesAllFields()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);

        Assert.Equal(-5f, world.GroundY);
        Assert.Equal(0f, world.LandingPad.PadX);
        Assert.Equal(-4.5f, world.LandingPad.PadY);
        Assert.Equal(2f, world.LandingPad.PadHalfWidth);
        Assert.Equal(100f, world.LandingPad.LandingBonus);
        Assert.Equal(2f, world.LandingPad.MaxLandingVelocity);

        Assert.Equal(0f, world.Spawn.X);
        Assert.Equal(15f, world.Spawn.Y);
        Assert.Equal(8f, world.Spawn.XRange);
        Assert.Equal(2f, world.Spawn.HeightRange);
        Assert.Equal(2f, world.Spawn.VelXRange);
        Assert.Equal(3f, world.Spawn.VelYMax);

        Assert.Single(world.Obstacles);
        Assert.Equal(2, world.Checkpoints.Length);
        Assert.Single(world.SpeedZones);
        Assert.Single(world.DangerZones);
        Assert.Single(world.Attractors);

        Assert.Equal(200f, world.SimulationConfig.MaxThrust);
        Assert.Equal(4, world.SimulationConfig.SensorCount);
        Assert.Equal(600, world.SimulationConfig.MaxSteps);

        Assert.Equal(1f, world.RewardWeights.PositionWeight);
        Assert.Equal(0.5f, world.RewardWeights.VelocityWeight);
        Assert.Equal(0.05f, world.RewardWeights.ControlEffortWeight);
    }

    [Fact]
    public void FromJson_ConvertsAnglesToRadians()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);

        float deg2Rad = MathF.PI / 180f;
        Assert.Equal(15f * deg2Rad, world.LandingPad.MaxLandingAngle, 1e-6f);
        Assert.Equal(30f * deg2Rad, world.Spawn.AngleRange, 1e-6f);
        Assert.Equal(15f * deg2Rad, world.SimulationConfig.MaxGimbalAngle, 1e-6f);
    }

    [Fact]
    public void FromJson_SortsCheckpointsByOrder()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);

        // JSON has order 1 first, order 0 second — loader should sort
        Assert.Equal(0, world.Checkpoints[0].Order);
        Assert.Equal(1, world.Checkpoints[1].Order);
        Assert.Equal(-1.933f, world.Checkpoints[0].X, 0.001f);
        Assert.Equal(3.7f, world.Checkpoints[1].X, 0.001f);
    }

    [Fact]
    public void FromJson_ValidatesSpawnAboveGround()
    {
        string badJson = DefaultWorldJson.Replace("\"Y\": 15", "\"Y\": -10");
        Assert.Throws<InvalidOperationException>(() => SimWorldLoader.FromJson(badJson));
    }

    [Fact]
    public void Configure_SetsEvaluatorProperties()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);
        var topology = DenseTopology.ForRocket(new[] { 16, 8 }, sensorCount: world.SimulationConfig.SensorCount);
        using var evaluator = new GPUDenseRocketLandingEvaluator(topology);

        evaluator.Configure(world);

        Assert.Equal(0f, evaluator.PadX);
        Assert.Equal(-4.5f, evaluator.PadY);
        Assert.Equal(2f, evaluator.PadHalfWidth);
        Assert.Equal(-5f, evaluator.GroundY);
        Assert.Equal(15f, evaluator.SpawnHeight);
        Assert.Equal(2f, evaluator.SpawnHeightRange);
        Assert.Equal(8f, evaluator.SpawnXRange);
        Assert.Equal(200f, evaluator.MaxThrust);
        Assert.Equal(600, evaluator.MaxSteps);
        Assert.Equal(4, evaluator.SensorCount);
        Assert.Single(evaluator.Obstacles);
        Assert.False(evaluator.ObstacleDeathEnabled);

        // Phase 2: reward weights
        Assert.Equal(20f, evaluator.RewardSurvivalWeight);
        Assert.Equal(20f * 1.0f, evaluator.RewardPositionWeight);
        Assert.Equal(5f * 0.5f, evaluator.RewardVelocityWeight);
        Assert.Equal(5f * 0.3f, evaluator.RewardAngleWeight);
        Assert.Equal(5f * 0.1f, evaluator.RewardAngVelWeight);
        Assert.Equal(0.05f, evaluator.WagglePenalty);
    }

    /// <summary>
    /// Integration test: load JSON -> configure evaluator -> run 10 CEM generations -> nonzero fitness.
    /// </summary>
    [Fact]
    public void Configure_ThenTrain_ProducesNonzeroFitness()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);
        var topology = DenseTopology.ForRocket(
            new[] { 16, 8 },
            sensorCount: world.SimulationConfig.SensorCount);
        using var evaluator = new GPUDenseRocketLandingEvaluator(topology);
        evaluator.Configure(world);

        int gpuCapacity = evaluator.OptimalPopulationSize;
        var config = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.25f,
            MinSigma = 0.08f,
            MaxSigma = 2.0f,
            CEMEliteFraction = 0.01f,
            CEMSigmaSmoothing = 0.3f,
            CEMMuSmoothing = 0.2f,
            StagnationThreshold = 9999,
        };

        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(42);
        var sw = Stopwatch.StartNew();

        float bestFitness = float.MinValue;
        for (int gen = 0; gen < 10; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, landings, _, _) = evaluator.EvaluateMultiSpawn(
                paramVectors, optimizer.TotalPopulation, numSpawns: 5, baseSeed: gen * 100);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            float maxFit = fitness.Max();
            if (maxFit > bestFitness) bestFitness = maxFit;

            Console.WriteLine($"  Gen {gen}: bestFit={maxFit:F2}, landings={landings}");
        }

        Console.WriteLine($"Best fitness after 10 gens: {bestFitness:F2} ({sw.Elapsed.TotalSeconds:F1}s)");
        Assert.True(bestFitness > 0f, "Expected nonzero fitness after 10 generations");
    }

    /// <summary>
    /// Phase 2: Reward weights flow to GPU and affect fitness values.
    /// Evaluate the same random population with two different weight profiles.
    /// Zero position weight should produce lower fitness (missing the 20-point close bonus).
    /// </summary>
    [Fact]
    public void RewardWeights_AffectFitnessValues()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);
        var topology = DenseTopology.ForRocket(new[] { 16, 8 }, sensorCount: world.SimulationConfig.SensorCount);

        // Generate a fixed random population
        using var evalDefault = new GPUDenseRocketLandingEvaluator(topology);
        evalDefault.Configure(world);

        var cemConfig = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.25f,
            MinSigma = 0.08f,
            MaxSigma = 2.0f,
            CEMEliteFraction = 0.01f,
            StagnationThreshold = 9999,
        };
        var optimizer = new IslandOptimizer(cemConfig, topology, evalDefault.OptimalPopulationSize);
        var rng = new Random(42);
        var paramVectors = optimizer.GeneratePopulation(rng);
        int pop = optimizer.TotalPopulation;

        // Evaluate with default weights (position=20)
        var (fitnessDefault, _, _, _) = evalDefault.EvaluateMultiSpawn(paramVectors, pop, numSpawns: 3, baseSeed: 0);
        float meanDefault = fitnessDefault.Average();

        // Evaluate with zero position weight
        using var evalNoPos = new GPUDenseRocketLandingEvaluator(topology);
        evalNoPos.Configure(world);
        evalNoPos.RewardPositionWeight = 0f;

        var (fitnessNoPos, _, _, _) = evalNoPos.EvaluateMultiSpawn(paramVectors, pop, numSpawns: 3, baseSeed: 0);
        float meanNoPos = fitnessNoPos.Average();

        Console.WriteLine($"  Mean fitness — default: {meanDefault:F2}, no-position: {meanNoPos:F2}");
        Assert.True(meanDefault > meanNoPos,
            $"Removing position weight should lower fitness ({meanDefault:F2} vs {meanNoPos:F2})");
    }

    [Fact]
    public void Configure_SetsZoneProperties()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);
        var topology = DenseTopology.ForRocket(new[] { 16, 8 }, sensorCount: world.SimulationConfig.SensorCount);
        using var evaluator = new GPUDenseRocketLandingEvaluator(topology);

        evaluator.Configure(world);

        Assert.Equal(2, evaluator.Checkpoints.Count);
        Assert.Single(evaluator.DangerZones);
        Assert.Single(evaluator.SpeedZones);
        Assert.Single(evaluator.Attractors);

        Assert.Equal(-1.933f, evaluator.Checkpoints[0].X, 0.001f);
        Assert.Equal(0, evaluator.Checkpoints[0].Order);
        Assert.Equal(20f, evaluator.Checkpoints[0].RewardBonus);

        Assert.Equal(10f, evaluator.DangerZones[0].PenaltyPerStep);
        Assert.Equal(0, evaluator.DangerZones[0].IsLethal);
        Assert.Equal(3f, evaluator.DangerZones[0].InfluenceRadius);

        Assert.Equal(5f, evaluator.SpeedZones[0].MaxSpeed);
        Assert.Equal(0.1f, evaluator.SpeedZones[0].RewardPerStep);

        Assert.Equal(10f, evaluator.Attractors[0].Magnitude);
        Assert.Equal(50f, evaluator.Attractors[0].ContactBonus);
    }

    /// <summary>
    /// Phase 3: Zones affect fitness. Compare same population evaluated with zones vs without.
    /// Danger zone penalty + attractor proximity shaping should produce measurably different fitness.
    /// </summary>
    [Fact]
    public void RewardZones_AffectFitnessValues()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);
        var topology = DenseTopology.ForRocket(new[] { 16, 8 }, sensorCount: world.SimulationConfig.SensorCount);

        // Generate a fixed random population
        using var evalWithZones = new GPUDenseRocketLandingEvaluator(topology);
        evalWithZones.Configure(world);

        var cemConfig = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.25f,
            MinSigma = 0.08f,
            MaxSigma = 2.0f,
            CEMEliteFraction = 0.01f,
            StagnationThreshold = 9999,
        };
        var optimizer = new IslandOptimizer(cemConfig, topology, evalWithZones.OptimalPopulationSize);
        var rng = new Random(42);
        var paramVectors = optimizer.GeneratePopulation(rng);
        int pop = optimizer.TotalPopulation;

        // Evaluate WITH zones (danger zone penalty + attractor + checkpoints)
        var (fitnessZones, _, _, _) = evalWithZones.EvaluateMultiSpawn(paramVectors, pop, numSpawns: 3, baseSeed: 0);
        float meanZones = fitnessZones.Average();

        // Evaluate WITHOUT zones
        using var evalNoZones = new GPUDenseRocketLandingEvaluator(topology);
        evalNoZones.Configure(world);
        evalNoZones.Checkpoints.Clear();
        evalNoZones.DangerZones.Clear();
        evalNoZones.SpeedZones.Clear();
        evalNoZones.Attractors.Clear();

        var (fitnessNoZones, _, _, _) = evalNoZones.EvaluateMultiSpawn(paramVectors, pop, numSpawns: 3, baseSeed: 0);
        float meanNoZones = fitnessNoZones.Average();

        Console.WriteLine($"  Mean fitness — with zones: {meanZones:F2}, no zones: {meanNoZones:F2}");
        Assert.NotEqual(meanZones, meanNoZones);
    }

    /// <summary>
    /// Phase 3: Training with zones completes without errors and produces nonzero fitness.
    /// </summary>
    [Fact]
    public void Configure_WithZones_ThenTrain_ProducesNonzeroFitness()
    {
        var world = SimWorldLoader.FromJson(DefaultWorldJson);
        var topology = DenseTopology.ForRocket(
            new[] { 16, 8 },
            sensorCount: world.SimulationConfig.SensorCount);
        using var evaluator = new GPUDenseRocketLandingEvaluator(topology);
        evaluator.Configure(world);

        int gpuCapacity = evaluator.OptimalPopulationSize;
        var config = new IslandConfig
        {
            IslandCount = 1,
            Strategy = UpdateStrategyType.CEM,
            InitialSigma = 0.25f,
            MinSigma = 0.08f,
            MaxSigma = 2.0f,
            CEMEliteFraction = 0.01f,
            CEMSigmaSmoothing = 0.3f,
            CEMMuSmoothing = 0.2f,
            StagnationThreshold = 9999,
        };

        var optimizer = new IslandOptimizer(config, topology, gpuCapacity);
        var rng = new Random(42);
        var sw = Stopwatch.StartNew();

        float bestFitness = float.MinValue;
        for (int gen = 0; gen < 10; gen++)
        {
            var paramVectors = optimizer.GeneratePopulation(rng);
            var (fitness, landings, _, _) = evaluator.EvaluateMultiSpawn(
                paramVectors, optimizer.TotalPopulation, numSpawns: 5, baseSeed: gen * 100);
            optimizer.Update(fitness, paramVectors);
            optimizer.ManageIslands(rng);

            float maxFit = fitness.Max();
            if (maxFit > bestFitness) bestFitness = maxFit;

            Console.WriteLine($"  Gen {gen}: bestFit={maxFit:F2}, landings={landings}");
        }

        Console.WriteLine($"Best fitness (with zones) after 10 gens: {bestFitness:F2} ({sw.Elapsed.TotalSeconds:F1}s)");
        Assert.True(bestFitness > 0f, "Expected nonzero fitness after 10 generations with zones");
    }
}
