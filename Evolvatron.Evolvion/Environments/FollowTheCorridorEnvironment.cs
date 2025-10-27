using Colonel.Tests.HagridTests.FollowTheCorridor;
using Godot;
using static Colonel.Tests.HagridTests.FollowTheCorridor.SimpleCarWorld;

namespace Evolvatron.Evolvion.Environments;

public enum DeathCause
{
    None,
    WallCollision,
    TooSlowTo4thMarker,
    TooSlowAfter4thMarker,
    Finished,
    Timeout
}

/// <summary>
/// Evolvion adapter for FollowTheCorridor environment.
/// Uses real SVG track from Colonel with SimpleCar physics.
/// </summary>
public class FollowTheCorridorEnvironment : IEnvironment
{
    private readonly SimpleCarWorld _world;
    private SimpleCarWorld.SimpleCar _car;
    private int _currentStep;
    private float _lastReward;
    private bool _wasDeadLastStep;

    public int InputCount => 9; // 9 distance sensors
    public int OutputCount => 2; // steering, throttle
    public int MaxSteps => _world.MaxSteps;
    public DeathCause CauseOfDeath { get; private set; }

    public FollowTheCorridorEnvironment(int maxSteps = 320)
    {
        _world = SimpleCarWorld.LoadFromFile(maxSteps);
        _car = new SimpleCarWorld.SimpleCar(_world);
        _currentStep = 0;
    }

    public FollowTheCorridorEnvironment(SimpleCarWorld sharedWorld)
    {
        _world = sharedWorld;
        _car = new SimpleCarWorld.SimpleCar(_world);
        _currentStep = 0;
    }

    public void Reset(int seed)
    {
        // Seed not used - SimpleCar track is deterministic
        _car.Reset();
        _currentStep = 0;
        CauseOfDeath = DeathCause.None;
        _lastReward = 0f;
        _wasDeadLastStep = false;
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

        _wasDeadLastStep = _car.IsDead;
        _currentStep++;
        _lastReward = _world.Update(_car, new[] { steering, throttle });

        // Detect cause of death by checking what changed
        if (_car.IsDead && !_wasDeadLastStep && CauseOfDeath == DeathCause.None)
        {
            // Check reward pattern to determine cause
            if (_lastReward > 0)
            {
                CauseOfDeath = DeathCause.Finished;
            }
            else if (_lastReward <= -0.49f) // -0.5f timeout penalties
            {
                // Check which timeout it was
                if (_car.CurrentProgressMarkerId < 4)
                {
                    CauseOfDeath = DeathCause.TooSlowTo4thMarker;
                }
                else
                {
                    CauseOfDeath = DeathCause.TooSlowAfter4thMarker;
                }
            }
            else // Negative reward from collision
            {
                CauseOfDeath = DeathCause.WallCollision;
            }
        }
        else if (_currentStep >= MaxSteps && CauseOfDeath == DeathCause.None)
        {
            CauseOfDeath = DeathCause.Timeout;
        }

        return _lastReward;
    }

    public bool IsTerminal()
    {
        return _car.IsDead || _currentStep >= MaxSteps;
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
