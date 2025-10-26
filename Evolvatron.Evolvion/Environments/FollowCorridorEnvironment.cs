namespace Evolvatron.Evolvion.Environments;

/// <summary>
/// "Follow the Corridor" environment - navigate a 2D car through a winding track.
///
/// This is a STUB that will need integration with Colonel.Tests SimpleCarWorld.
/// The actual environment requires:
/// - SimpleCarWorld.cs (track geometry, walls, progress markers)
/// - SimpleCar.cs (car physics, sensors)
/// - Grid.cs (spatial hash for collision detection)
/// - SVG track data from C:\Dev\Colonel\Data\Besofa\Race Track.svg
///
/// State: 9 distance sensors (raycasts to walls)
/// Action: [steering, throttle] both in [-1, 1]
///
/// Episode terminates when:
/// - Car hits a wall (failure)
/// - Car reaches finish line (success)
/// - MaxSteps reached (timeout)
/// </summary>
public class FollowCorridorEnvironment : IEnvironment
{
    // This is a placeholder - actual implementation requires Colonel.Tests dependencies
    // Which include Godot types and complex rendering code.

    public int InputCount => 9; // 9 distance sensors
    public int OutputCount => 2; // steering + throttle
    public int MaxSteps => 320;

    public void Reset(int seed = 0)
    {
        throw new NotImplementedException(
            "FollowCorridorEnvironment requires Colonel.Tests.SimpleCarWorld integration. " +
            "See C:\\Dev\\Colonel\\Colonel.Tests\\HagridTests\\FollowTheCorridor\\ for source.");
    }

    public void GetObservations(Span<float> observations)
    {
        throw new NotImplementedException();
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        throw new NotImplementedException();
    }

    public bool IsTerminal()
    {
        throw new NotImplementedException();
    }
}

/// <summary>
/// Instructions for integrating FollowCorridor environment:
///
/// 1. Copy required files from Colonel.Tests:
///    - SimpleCarWorld.cs (contains SimpleCar, Sensor classes)
///    - Extract Grid class or create simplified collision detection
///    - Copy SVG track file: C:\Dev\Colonel\Data\Besofa\Race Track.svg
///
/// 2. Remove dependencies:
///    - Remove Godot.Vector2 usage → use System.Numerics.Vector2
///    - Remove SVG rendering code (or make it optional)
///    - Simplify Grid to just wall segments + raycasting
///
/// 3. Implement IEnvironment:
///    private SimpleCarWorld _world;
///    private SimpleCar _car;
///
///    Reset(seed):
///      _world = SimpleCarWorld.LoadFromFile() or create programmatically
///      _car = new SimpleCar(_world.Start.MidPoint, _world.Start.Angle)
///
///    GetObservations(obs):
///      var state = _car.GetState(_world.WallGrid);
///      state.CopyTo(obs);
///
///    Step(actions):
///      _car.Update(actions[0], actions[1]); // steering, throttle
///      return _world.Update(_car, actions);
///
///    IsTerminal():
///      return _car.IsDead || _car.CurrentProgressMarkerId >= _world.ProgressMarkers.Count;
///
/// 4. Simplified version (no SVG loading):
///    - Create procedural track (e.g., sine wave corridor)
///    - Use simple line segment collision detection
///    - Place progress markers along centerline
///
/// Example procedural track:
///
/// for (float x = 0; x < 200; x += 5)
/// {
///     float y = 50 * sin(x / 20);
///     float width = 20;
///     walls.Add((x, y - width), (x + 5, y - width)); // left wall
///     walls.Add((x, y + width), (x + 5, y + width)); // right wall
///     progressMarkers.Add((x, y));
/// }
/// </summary>
