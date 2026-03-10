using System.Text.Json;

namespace Evolvatron.Evolvion.World;

/// <summary>
/// Loads SimWorld from editor-exported JSON.
/// Converts degree angles to radians, sorts checkpoints by order, validates.
/// </summary>
public static class SimWorldLoader
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public static SimWorld FromJson(string json)
    {
        var world = JsonSerializer.Deserialize<SimWorld>(json, Options)
            ?? throw new InvalidOperationException("Failed to deserialize SimWorld JSON");

        ConvertAnglesToRadians(world);
        SortCheckpoints(world);
        Validate(world);

        return world;
    }

    private static void ConvertAnglesToRadians(SimWorld world)
    {
        const float deg2Rad = MathF.PI / 180f;

        world.LandingPad.MaxLandingAngle *= deg2Rad;
        world.Spawn.AngleRange *= deg2Rad;
        world.SimulationConfig.MaxGimbalAngle *= deg2Rad;
    }

    private static void SortCheckpoints(SimWorld world)
    {
        if (world.Checkpoints.Length > 1)
            Array.Sort(world.Checkpoints, (a, b) => a.Order.CompareTo(b.Order));
    }

    private static void Validate(SimWorld world)
    {
        if (world.LandingPad == null)
            throw new InvalidOperationException("SimWorld.LandingPad is required");
        if (world.Spawn == null)
            throw new InvalidOperationException("SimWorld.Spawn is required");
        if (world.SimulationConfig == null)
            throw new InvalidOperationException("SimWorld.SimulationConfig is required");
        if (world.RewardWeights == null)
            throw new InvalidOperationException("SimWorld.RewardWeights is required");
        if (world.Spawn.Y <= world.GroundY)
            throw new InvalidOperationException(
                $"Spawn Y ({world.Spawn.Y}) must be above ground ({world.GroundY})");
        if (world.SimulationConfig.MaxSteps <= 0)
            throw new InvalidOperationException("MaxSteps must be positive");
    }
}
