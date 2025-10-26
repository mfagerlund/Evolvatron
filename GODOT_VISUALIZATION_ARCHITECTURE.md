# Godot Visualization Architecture for Evolvion

**Created:** 2025-01-26
**Status:** Design Document

---

## Overview

This document outlines the architecture for integrating Godot visualization with Evolvion's evolutionary training system for FollowTheCorridor.

## Goals

1. **Multi-Agent Visualization**: Show ALL individuals in population (400 cars) evolving in real-time
2. **Real SVG Track**: Use actual `C:\Dev\Colonel\Data\Besofa\Race Track.svg` from Colonel
3. **Generation Playback**: Step through generations, watch best individual per generation
4. **Interactive Controls**: Play/pause, speed control, generation stepping
5. **Performance Stats**: Display fitness, generation #, speed, sensors

## Architecture Layers

### 1. Evolvion Core (C#)
- **FollowTheCorridorEnvironment**: IEnvironment adapter (✅ COMPLETE)
- **Population**: 400 individuals (4 species × 100)
- **Evolver**: Generation stepping with tournament selection + elitism
- **CPUEvaluator**: Neural network forward pass

### 2. Godot Integration Layer (C#)
- **EvolvionBridge**: Godot Node that manages evolution loop
- **CarAgent**: Godot Node2D for each individual car
- **TrackRenderer**: Godot Node2D that renders SVG track geometry
- **UIController**: Godot Control for stats and controls

### 3. Godot Scene (GDScript optional)
- **Main.tscn**: Root scene with camera, UI, track
- **CarAgent.tscn**: Instanced scene for each car (400 instances)
- **Sensor.tscn**: Visual representation of distance sensors

---

## Component Design

### EvolvionBridge.cs (Godot Node)

```csharp
using Godot;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;

public partial class EvolvionBridge : Node
{
    private Evolver _evolver;
    private Population _population;
    private FollowTheCorridorEnvironment _baseEnvironment;
    private SimpleFitnessEvaluator _evaluator;
    private CPUEvaluator _cpuEvaluator;

    private List<CarAgent> _carAgents;
    private int _generation = 0;
    private bool _isPlaying = false;
    private int _playbackSpeed = 1;

    public override void _Ready()
    {
        // Initialize evolution
        var topology = CreateTopology();
        var config = new EvolutionConfig { /* ... */ };

        _evolver = new Evolver(seed: 42);
        _population = _evolver.InitializePopulation(config, topology);
        _baseEnvironment = new FollowTheCorridorEnvironment(maxSteps: 320);
        _evaluator = new SimpleFitnessEvaluator();
        _cpuEvaluator = new CPUEvaluator(topology);

        // Create 400 car agents
        SpawnCarAgents();

        // Evaluate initial population
        _evaluator.EvaluatePopulation(_population, _baseEnvironment, seed: 0);
    }

    public override void _Process(double delta)
    {
        if (_isPlaying)
        {
            // Step simulation for all cars
            StepAllCars();
        }
    }

    public void NextGeneration()
    {
        _evolver.StepGeneration(_population);
        _generation++;
        _evaluator.EvaluatePopulation(_population, _baseEnvironment, seed: _generation);
        ResetAllCars();
    }

    private void SpawnCarAgents()
    {
        _carAgents = new List<CarAgent>();
        int index = 0;

        foreach (var species in _population.Species)
        {
            foreach (var individual in species.Individuals)
            {
                var carScene = GD.Load<PackedScene>("res://Scenes/CarAgent.tscn");
                var carAgent = carScene.Instantiate<CarAgent>();

                carAgent.Initialize(individual, species.Topology, _cpuEvaluator, index++);
                carAgent.Position = _baseEnvironment.World.Start.MidPoint;

                AddChild(carAgent);
                _carAgents.Add(carAgent);
            }
        }
    }
}
```

### CarAgent.cs (Godot Node2D)

```csharp
using Godot;
using Evolvatron.Evolvion;

public partial class CarAgent : Node2D
{
    private Individual _individual;
    private SpeciesSpec _topology;
    private CPUEvaluator _evaluator;
    private FollowTheCorridorEnvironment _environment;

    private Sprite2D _carSprite;
    private Line2D[] _sensorLines;
    private int _agentIndex;

    public override void _Ready()
    {
        _carSprite = GetNode<Sprite2D>("CarSprite");
        CreateSensorVisuals();
    }

    public void Initialize(Individual individual, SpeciesSpec topology, CPUEvaluator evaluator, int index)
    {
        _individual = individual;
        _topology = topology;
        _evaluator = evaluator;
        _agentIndex = index;

        // Each car gets its own environment instance
        _environment = new FollowTheCorridorEnvironment(maxSteps: 320);
        _environment.Reset(seed: 0);

        // Set color based on fitness
        UpdateColor();
    }

    public void Step()
    {
        if (_environment.IsTerminal()) return;

        // Get observations
        var observations = new float[_environment.InputCount];
        _environment.GetObservations(observations);

        // Run neural network
        var actions = _evaluator.Evaluate(_individual, observations);

        // Step environment
        _environment.Step(actions);

        // Update visual position
        Position = _environment.GetCarPosition();
        Rotation = _environment.GetCarHeading();

        // Update sensors
        UpdateSensorVisuals();
    }

    private void UpdateColor()
    {
        // Color based on fitness: green=best, red=worst
        float normalizedFitness = (_individual.Fitness + 1f) / 2f; // Assume [-1, 1] range
        _carSprite.Modulate = new Color(1 - normalizedFitness, normalizedFitness, 0, 0.5f);
    }
}
```

### TrackRenderer.cs (Godot Node2D)

```csharp
using Godot;
using Colonel.Tests.HagridTests.FollowTheCorridor;

public partial class TrackRenderer : Node2D
{
    private SimpleCarWorld _world;

    public override void _Ready()
    {
        _world = SimpleCarWorld.LoadFromFile();
        QueueRedraw();
    }

    public override void _Draw()
    {
        // Draw track walls
        foreach (var lineSegment in _world.WallGrid.LineSegments)
        {
            DrawLine(lineSegment.Start, lineSegment.End, Colors.Black, 2f);
        }

        // Draw progress markers
        foreach (var marker in _world.ProgressMarkers)
        {
            DrawCircle(marker.Position, _world.ProgressMarkerRadius,
                      new Color(0, 1, 0, 0.3f));
        }

        // Draw start/finish
        DrawLine(_world.Start.Start, _world.Start.End, Colors.Blue, 3f);
        DrawLine(_world.Finish.Start, _world.Finish.End, Colors.Green, 3f);
    }
}
```

---

## Godot Project Structure

```
FollowTheCorridorDemo/
├── project.godot
├── Scenes/
│   ├── Main.tscn                  # Root scene
│   ├── CarAgent.tscn              # Individual car (instanced 400x)
│   ├── Track.tscn                 # Track renderer
│   └── UI/
│       ├── StatsPanel.tscn        # Fitness, generation stats
│       └── Controls.tscn          # Play/pause, speed, next gen
├── Scripts/
│   ├── EvolvionBridge.cs          # Main evolution controller
│   ├── CarAgent.cs                # Per-agent logic
│   ├── TrackRenderer.cs           # SVG track rendering
│   └── UIController.cs            # UI event handling
├── Assets/
│   ├── car_sprite.png             # Simple car visual
│   └── track_data/                # SVG track (symlink to Colonel/Data)
└── .csproj                        # C# project reference to Evolvatron.Evolvion
```

---

## Performance Considerations

### 400 Cars Challenge
- **Instancing**: Use Godot's MultiMesh for cars (single draw call)
- **LOD**: Cull off-screen cars, reduce sensor visual detail
- **Batch Updates**: Update positions in chunks (100 cars/frame)
- **Async Evolution**: Run generation stepping in background thread

### Memory
- **Shared Environment**: Each car needs own environment instance (9 floats state)
- **Topology Sharing**: All individuals in species share same SpeciesSpec
- **Sparse Updates**: Only update visual for cars that moved significantly

---

## Controls & UI

### Keyboard Controls
- **SPACE**: Play/Pause simulation
- **N**: Next Generation
- **R**: Reset current generation
- **LEFT/RIGHT**: Adjust playback speed (1x-10x)
- **F**: Focus camera on best individual
- **1-4**: Focus on specific species

### UI Elements
- **Generation Counter**: Current generation #
- **Best Fitness**: Highest fitness in population (% of track completed)
- **Species Stats**: Per-species median fitness, diversity
- **Playback Speed**: Current simulation speed multiplier
- **Progress Bar**: Completion % for best individual

---

## Implementation Plan

### Phase 1: Basic Integration (1-2 days)
1. Create Godot project with C# support
2. Reference Evolvatron.Evolvion project
3. Implement EvolvionBridge basic loop
4. Spawn 400 CarAgent instances (static positions first)

### Phase 2: Track & Physics (1 day)
5. Implement TrackRenderer with SVG loading
6. Connect CarAgent.Step() to environment
7. Verify car movement follows physics

### Phase 3: Visualization (1 day)
8. Add sensor line rendering
9. Color code cars by fitness
10. Implement camera controls

### Phase 4: UI & Controls (1 day)
11. Build stats panel
12. Add playback controls
13. Implement generation stepping

### Phase 5: Optimization (1 day)
14. Profile 400-car performance
15. Implement culling & LOD
16. Add async evolution option

---

## Next Steps

1. ✅ **DONE**: Create FollowTheCorridorEnvironment adapter
2. ✅ **DONE**: Add Colonel.Framework + Colonel.Tests references
3. ✅ **DONE**: Write integration test
4. **TODO**: Create Godot project structure
5. **TODO**: Implement EvolvionBridge
6. **TODO**: Build TrackRenderer
7. **TODO**: Deploy and test with real SVG track

---

## Notes

- Godot 4.x recommended (better C# support, MultiMesh improvements)
- Consider GDScript for UI if performance is not critical
- SVG track geometry can be baked into Godot scene for faster loading
- Record best trajectories per generation for replay mode

---

**Status**: Ready for implementation once test confirms environment works correctly.
