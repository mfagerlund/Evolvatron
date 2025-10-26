using Colonel.Tests.HagridTests.FollowTheCorridor;
using Godot;
using static Colonel.Tests.HagridTests.FollowTheCorridor.SimpleCarWorld;

namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// Evolvion adapter for FollowTheCorridor environment.
/// Uses real SVG track from Colonel with SimpleCar physics.
/// </summary>
public class FollowTheCorridorEnvironment : IEnvironment
{
    private readonly SimpleCarWorld _world;
    private SimpleCarWorld.SimpleCar _car;

    public int InputCount => 9; // 9 distance sensors
    public int OutputCount => 2; // steering, throttle
    public int MaxSteps => _world.MaxSteps;

    public FollowTheCorridorEnvironment(int maxSteps = 320)
    {
        _world = SimpleCarWorld.LoadFromFile(maxSteps);
        _car = new SimpleCarWorld.SimpleCar(_world);
    }

    public void Reset(int seed)
    {
        // Seed not used - SimpleCar track is deterministic
        _car.Reset();
    }

    public void GetObservations(Span<float> observations)
    {
        var state = _car.GetState(_world.WallGrid);
        for (int i = 0; i < state.Length && i < observations.Length; i++)
        {
            observations[i] = state[i];
        }
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        float steering = Math.Clamp(actions[0], -1f, 1f);
        float throttle = Math.Clamp(actions[1], -1f, 1f);

        return _world.Update(_car, new[] { steering, throttle });
    }

    public bool IsTerminal()
    {
        return _car.IsDead;
    }

    // Additional properties for visualization
    public Vector2 GetCarPosition() => _car.Position;
    public float GetCarHeading() => _car.HeadingAngle;
    public float GetCarSpeed() => _car.Speed;
    public Vector2 GetCarDirection() => _car.Direction;
    public SimpleCarWorld.Sensor[] GetSensors() => _car.Sensors;
    public SimpleCarWorld World => _world;
    public SimpleCarWorld.SimpleCar Car => _car;
}
