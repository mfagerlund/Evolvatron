# Converting Hagrid Tests to Evolvion Environments

**Goal:** Convert Colonel.Tests Hagrid environments to work with Evolvion evolutionary ML.

---

## Case Study: FollowTheCorridor

**Source:** `C:\Dev\Colonel\Colonel.Tests\HagridTests\FollowTheCorridor\`

### Original Architecture (Hagrid)

```
FollowTheCorridorTests.cs
├── SimpleCarWorldAgentManager : AgentManager
│   ├── Uses Hagrid's policy gradient + trajectory optimization
│   ├── Neural network trained with backprop
│   └── Exploration via trajectory sampling
├── SimpleCarWorld (environment)
│   ├── LoadFromFile() - parses SVG track
│   ├── WallGrid - spatial hash for collision detection
│   ├── ProgressMarkers - 256 checkpoints along track
│   └── Update(car, action) → reward
└── SimpleCar (agent)
    ├── 9 distance sensors (raycasts)
    ├── 2 actions: steering [-1,1], throttle [-1,1]
    └── Physics: position, velocity, heading, steering angle
```

**Key Stats:**
- State: 9 floats (sensor distances)
- Action: 2 floats (continuous control)
- Episode: ~320 steps max
- Success: Reach all 256 progress markers
- Hagrid performance: Solves in ~4-12k agent runs (varies by config)

---

## Target Architecture (Evolvion)

```
FollowCorridorEnvironment : IEnvironment
├── Reset(seed) → initialize car + track
├── GetObservations(span) → 9 sensor readings
├── Step(actions) → apply steering/throttle, return reward
└── IsTerminal() → hit wall OR finished OR timeout

Used by:
└── SimpleFitnessEvaluator.EvaluatePopulation()
    └── CPUEvaluator.Evaluate() - forward pass neural network
```

**Differences from Hagrid:**
- **No backprop**: Weights evolved, not trained
- **No trajectories**: Each episode is independent
- **Population-based**: 400 individuals tested in parallel
- **Simpler**: No policy network, no value function, no replay buffer

---

## Integration Steps

### Option 1: Full Port (Complex)

**Pros:** Get exact FollowCorridor environment
**Cons:** Heavy dependencies (Godot, SVG parsing, rendering)

**Steps:**
1. Copy `SimpleCarWorld.cs` → Evolvatron.Evolvion/Environments/
2. Copy `Grid.cs` (or extract collision detection logic)
3. Remove Godot dependencies:
   - Replace `Godot.Vector2` → `System.Numerics.Vector2`
   - Remove rendering (RenderToSvg, Svg class usage)
4. Copy SVG track: `C:\Dev\Colonel\Data\Besofa\Race Track.svg`
5. Implement `IEnvironment` wrapper:

```csharp
public class FollowCorridorEnvironment : IEnvironment
{
    private SimpleCarWorld _world;
    private SimpleCar _car;

    public int InputCount => 9; // sensors
    public int OutputCount => 2; // steering, throttle
    public int MaxSteps => 320;

    public void Reset(int seed = 0)
    {
        _world = SimpleCarWorld.LoadFromFile(MaxSteps);
        _car = new SimpleCar(_world.Start.MidPoint, _world.Start.Angle);
    }

    public void GetObservations(Span<float> observations)
    {
        var state = _car.GetState(_world.WallGrid);
        state.CopyTo(observations);
    }

    public float Step(ReadOnlySpan<float> actions)
    {
        float steering = actions[0];
        float throttle = actions[1];
        return _world.Update(_car, new[] { steering, throttle });
    }

    public bool IsTerminal()
    {
        return _car.IsDead ||
               _car.CurrentProgressMarkerId >= _world.ProgressMarkers.Count;
    }
}
```

**Estimated effort:** 2-4 hours (dependency removal, testing)

---

### Option 2: Simplified Corridor (Recommended)

**Pros:** Minimal dependencies, easier to understand
**Cons:** Different geometry than original

**Steps:**
1. Create procedural track (sine wave or S-curve)
2. Implement simple raycast collision (line-vs-line intersection)
3. Place progress markers along centerline
4. Simplified car physics

```csharp
public class SimpleCorridorEnvironment : IEnvironment
{
    private List<(Vector2 left, Vector2 right)> _walls;
    private List<Vector2> _checkpoints;
    private Vector2 _position;
    private float _heading;
    private float _speed;
    private int _checkpointIndex;
    private int _step;

    public int InputCount => 9;
    public int OutputCount => 2;
    public int MaxSteps => 320;

    public void Reset(int seed = 0)
    {
        GenerateTrack(seed); // Procedural sine wave corridor
        _position = new Vector2(0, 0);
        _heading = 0;
        _speed = 0;
        _checkpointIndex = 0;
        _step = 0;
    }

    private void GenerateTrack(int seed)
    {
        var random = new Random(seed);
        _walls = new List<(Vector2, Vector2)>();
        _checkpoints = new List<Vector2>();

        for (float x = 0; x < 200; x += 5)
        {
            float y = 30 * MathF.Sin(x / 20f);
            float width = 15;

            _walls.Add((
                new Vector2(x, y - width),
                new Vector2(x + 5, y - width)
            ));
            _walls.Add((
                new Vector2(x, y + width),
                new Vector2(x + 5, y + width)
            ));

            _checkpoints.Add(new Vector2(x, y));
        }
    }

    public void GetObservations(Span<float> observations)
    {
        // 9 raycasts at different angles
        float[] angles = { -60, -45, -30, -15, 0, 15, 30, 45, 60 };
        for (int i = 0; i < 9; i++)
        {
            float rayAngle = _heading + angles[i] * MathF.PI / 180f;
            float distance = Raycast(_position, rayAngle, maxRange: 50f);
            observations[i] = 1f - (distance / 50f); // Normalize
        }
    }

    private float Raycast(Vector2 origin, float angle, float maxRange)
    {
        Vector2 direction = new Vector2(MathF.Cos(angle), MathF.Sin(angle));
        Vector2 rayEnd = origin + direction * maxRange;

        float minDist = maxRange;
        foreach (var (wallStart, wallEnd) in _walls)
        {
            if (LineIntersection(origin, rayEnd, wallStart, wallEnd, out float t))
            {
                float dist = t * maxRange;
                if (dist < minDist) minDist = dist;
            }
        }
        return minDist;
    }

    private bool LineIntersection(Vector2 p1, Vector2 p2, Vector2 p3, Vector2 p4, out float t)
    {
        // Standard line-line intersection algorithm
        Vector2 s1 = p2 - p1;
        Vector2 s2 = p4 - p3;
        float denom = Cross2D(s1, s2);
        if (MathF.Abs(denom) < 1e-6f)
        {
            t = 0;
            return false;
        }

        float s = Cross2D(p3 - p1, s1) / denom;
        t = Cross2D(p3 - p1, s2) / denom;
        return t >= 0 && t <= 1 && s >= 0 && s <= 1;
    }

    private float Cross2D(Vector2 a, Vector2 b) => a.X * b.Y - a.Y * b.X;

    public float Step(ReadOnlySpan<float> actions)
    {
        _step++;

        // Apply actions
        float steering = Math.Clamp(actions[0], -1f, 1f);
        float throttle = Math.Clamp(actions[1], -1f, 1f);

        // Update car state (simplified physics)
        _heading += steering * 0.1f * _speed / 10f;
        _speed += throttle * 2f;
        _speed = Math.Clamp(_speed, 0f, 10f);

        Vector2 velocity = new Vector2(MathF.Cos(_heading), MathF.Sin(_heading)) * _speed;
        _position += velocity * 0.1f;

        // Check collision
        if (Raycast(_position, _heading, maxRange: 2f) < 2f)
        {
            return -1f; // Hit wall
        }

        // Check checkpoints
        float reward = 0f;
        while (_checkpointIndex < _checkpoints.Count)
        {
            if (Vector2.Distance(_position, _checkpoints[_checkpointIndex]) < 5f)
            {
                reward += 1f / _checkpoints.Count;
                _checkpointIndex++;
            }
            else break;
        }

        return reward;
    }

    public bool IsTerminal()
    {
        return _step >= MaxSteps ||
               Raycast(_position, _heading, 2f) < 2f ||
               _checkpointIndex >= _checkpoints.Count;
    }
}
```

**Estimated effort:** 1-2 hours

---

## Performance Expectations

**XOR:** 7 generations (simple logic problem)
**CartPole:** 1 generation (continuous control, sparse reward)
**FollowCorridor:** ???

**Challenges:**
1. **State space:** 9D continuous (vs 4D for CartPole)
2. **Action space:** 2D continuous with coupling (steering affects heading affects position)
3. **Credit assignment:** Sparse reward (only at checkpoints)
4. **Long horizon:** 320 steps (vs 1000 for CartPole, but more complex dynamics)

**Prediction:**
- With tuned hyperparameters (400 pop, Jitter=0.95, JitterStd=0.3)
- Likely 10-50 generations for basic navigation
- May need reward shaping (e.g., distance to next checkpoint)

---

## Recommended Approach

**For demonstration purposes:**

1. **Start with SimpleCorridorEnvironment** (procedural track)
2. Run evolution with tuned hyperparameters
3. If it solves easily, consider full port of FollowCorridor
4. Compare Evolvion performance vs Hagrid benchmarks

**For production use:**

1. **Full port** if you need exact Hagrid comparison
2. **Abstract environment loader** to support multiple tracks
3. **Add visualization** (export trajectories to SVG like Hagrid does)

---

## Code Location

**Hagrid (original):**
- `C:\Dev\Colonel\Colonel.Tests\HagridTests\FollowTheCorridor\`
- Uses policy gradients + trajectory optimization
- 9 sensors → neural network → 2 actions
- Success: ~4-23k agent runs depending on config

**Evolvion (target):**
- `C:\Dev\Evolvatron\Evolvatron.Evolvion\Environments\`
- Uses evolutionary strategies
- Same I/O signature (9 → NN → 2)
- Success: TBD (need to implement and test)

---

## Next Steps

1. ☐ Implement SimpleCorridorEnvironment (simplified version)
2. ☐ Create FollowCorridorEvolutionTest
3. ☐ Run evolution and measure generations to solve
4. ☐ Compare vs Hagrid performance
5. ☐ (Optional) Full port if simplified version validates approach

---

## Questions

**Q: Why not just use Hagrid?**
A: Hagrid requires backprop (PyTorch/TorchSharp), complex trajectory optimization. Evolvion is pure evolutionary - simpler, more parallelizable, potentially GPU-friendly (via ILGPU kernels).

**Q: Will evolution be competitive with policy gradients?**
A: Unknown! XOR and CartPole suggest yes for simpler problems. FollowCorridor is the first real test.

**Q: Should we use shaped rewards?**
A: Try sparse first (checkpoints only). If it struggles, add:
- Distance to next checkpoint (negative gradient)
- Speed bonus (encourage forward progress)
- Smoothness penalty (discourage oscillation)
