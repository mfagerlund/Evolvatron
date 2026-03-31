using Godot;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.World;

namespace Evolvatron.Godot;

/// <summary>
/// Live CEM training + visualization in Godot.
/// Training runs on a background thread; render thread shows top individuals
/// from the latest generation, cycling through episodes automatically.
/// Same architecture as the Raylib LiveTrainer.
/// </summary>
public partial class RocketReplayScene : Node2D
{
    const int BodiesPerWorld = 3;
    const int GeomsPerBody0 = 7;
    const int GeomsPerLeg = 6;
    const int GeomsPerWorld = GeomsPerBody0 + GeomsPerLeg * 2;
    const float BodyHalfLength = 0.75f;
    const float InitialZoom = 30f;
    const int MaxReplayCount = 10;
    const int MaxGenerations = 300;
    const int NumSpawns = 10;

    // --- Shared training state (accessed under _lock) ---
    readonly object _lock = new();
    float[]? _latestTopNParams;
    int _currentGeneration;
    float _bestFitness;
    float _landingRate;
    bool _trainingDone;
    int _solvedGeneration = -1;
    readonly System.Collections.Generic.List<float> _rateHistory = new();
    readonly System.Collections.Generic.List<float> _fitHistory = new();
    int _trainPop;

    // --- Viz state (render thread only) ---
    SimWorld _simWorld = null!;
    DenseTopology _topology = null!;
    GPUDenseRocketLandingEvaluator? _vizEval;
    CancellationTokenSource _trainCts = new();
    Thread? _trainingThread;

    float[]? _currentReplayParams;
    bool _replayActive;
    int _replayStep;
    int _postTerminalFrames;
    int _activeReplayCount;

    GPURigidBody[]? _bodies;
    GPURigidBodyGeom[]? _geoms;
    byte[]? _terminal;
    byte[]? _landed;
    float[]? _throttle;
    float[]? _gimbal;

    // --- Godot nodes ---
    Camera2D _camera = null!;
    Node2D _worldLayer = null!;
    Node2D _rocketsLayer = null!;
    Label _hudLabel = null!;
    Label _controlsLabel = null!;

    Node2D[][] _rocketGeomNodes = [];
    Line2D[] _trailNodes = [];
    Line2D[] _flameNodes = [];

    int _simSpeed = 2;
    bool _gpuReady;

    public override void _Ready()
    {
        _camera = new Camera2D();
        _camera.Zoom = new Vector2(InitialZoom, InitialZoom);
        AddChild(_camera);

        _worldLayer = new Node2D();
        AddChild(_worldLayer);

        _rocketsLayer = new Node2D();
        AddChild(_rocketsLayer);

        var hud = new CanvasLayer();
        AddChild(hud);
        _hudLabel = new Label();
        _hudLabel.Position = new Vector2(10, 10);
        _hudLabel.AddThemeColorOverride("font_color", Colors.White);
        _hudLabel.AddThemeFontSizeOverride("font_size", 16);
        hud.AddChild(_hudLabel);

        _controlsLabel = new Label();
        _controlsLabel.AddThemeColorOverride("font_color", new Color(0.5f, 0.5f, 0.5f));
        _controlsLabel.AddThemeFontSizeOverride("font_size", 14);
        _controlsLabel.Text = "[R] Restart training   [+/-] Speed   [Space] Pause   [Scroll] Zoom";
        hud.AddChild(_controlsLabel);

        string basePath = ProjectSettings.GlobalizePath("res://");
        string worldPath = Path.GetFullPath(Path.Combine(basePath, "..", "scratch", "test-world.json"));

        if (!File.Exists(worldPath))
        {
            _hudLabel.Text = $"World not found:\n{worldPath}";
            return;
        }

        try
        {
            LoadWorldAndStartTraining(worldPath);
        }
        catch (Exception ex)
        {
            _hudLabel.Text = $"Init failed:\n{ex.Message}";
            GD.PrintErr($"Init failed: {ex}");
        }
    }

    void LoadWorldAndStartTraining(string jsonPath)
    {
        string json = File.ReadAllText(jsonPath);
        _simWorld = SimWorldLoader.FromJson(json);
        int sensorCount = _simWorld.SimulationConfig?.SensorCount ?? 0;
        _topology = DenseTopology.ForRocket(new[] { 16, 12 }, sensorCount: sensorCount);

        GD.Print($"LiveTrainer: topology={_topology}, sensors={sensorCount}");

        // Build static world visuals
        BuildWorldVisuals();

        // Create viz evaluator
        _vizEval?.Dispose();
        _vizEval = new GPUDenseRocketLandingEvaluator(_topology);
        _vizEval.Configure(_simWorld);
        _gpuReady = true;

        // Reset shared state
        lock (_lock)
        {
            _latestTopNParams = null;
            _currentGeneration = 0;
            _bestFitness = 0f;
            _landingRate = 0f;
            _trainingDone = false;
            _solvedGeneration = -1;
            _rateHistory.Clear();
            _fitHistory.Clear();
        }

        // Start training thread
        _trainCts = new CancellationTokenSource();
        var ct = _trainCts.Token;
        _trainingThread = new Thread(() => RunTraining(ct)) { IsBackground = true };
        _trainingThread.Start();

        // Position camera to show world
        float cx = _simWorld.LandingPad?.PadX ?? 0f;
        float cy = (_simWorld.GroundY + (_simWorld.Spawn?.Y ?? 15f)) / 2f;
        _camera.Position = ToGodot(cx, cy);
    }

    void RunTraining(CancellationToken ct)
    {
        try
        {
            using var trainEval = new GPUDenseRocketLandingEvaluator(_topology);
            trainEval.Configure(_simWorld);

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

            int gpuCapacity = trainEval.OptimalPopulationSize;
            var optimizer = new IslandOptimizer(config, _topology, gpuCapacity);
            var rng = new Random(42);
            int actualSpawns = trainEval.SpawnCount > 0 ? trainEval.SpawnCount : NumSpawns;
            int paramCount = _topology.TotalParams;

            lock (_lock) _trainPop = optimizer.TotalPopulation;
            GD.Print($"  Training: pop={optimizer.TotalPopulation}, spawns={actualSpawns}");

            float[]? eliteParams = null;
            float eliteFitness = float.NegativeInfinity;

            for (int gen = 0; gen < MaxGenerations; gen++)
            {
                if (ct.IsCancellationRequested) break;

                var paramVectors = optimizer.GeneratePopulation(rng);
                int totalPop = optimizer.TotalPopulation;

                if (eliteParams != null)
                    Array.Copy(eliteParams, 0, paramVectors, (totalPop - 1) * paramCount, paramCount);

                var (fitness, landings, maxLandingCount, maxLandingIdx) =
                    trainEval.EvaluateMultiSpawn(
                        paramVectors, totalPop, actualSpawns, baseSeed: trainEval.SpawnSeed);

                for (int i = 0; i < totalPop; i++)
                {
                    if (fitness[i] > eliteFitness)
                    {
                        eliteFitness = fitness[i];
                        eliteParams ??= new float[paramCount];
                        Array.Copy(paramVectors, i * paramCount, eliteParams, 0, paramCount);
                    }
                }

                optimizer.Update(fitness, paramVectors);
                optimizer.ManageIslands(rng);

                float maxFit = fitness.Max();
                float rate = (float)landings / (totalPop * actualSpawns) * 100f;

                // Extract top N
                int topN = Math.Min(MaxReplayCount, totalPop);
                var indices = Enumerable.Range(0, totalPop).ToArray();
                Array.Sort(indices, (a, b) => fitness[b].CompareTo(fitness[a]));

                var topParams = new float[topN * paramCount];
                var topFitness = new float[topN];
                for (int k = 0; k < topN; k++)
                {
                    Array.Copy(paramVectors, indices[k] * paramCount, topParams, k * paramCount, paramCount);
                    topFitness[k] = fitness[indices[k]];
                }

                if (eliteParams != null && eliteFitness > topFitness[0])
                {
                    Array.Copy(eliteParams, 0, topParams, 0, paramCount);
                    topFitness[0] = eliteFitness;
                    maxFit = eliteFitness;
                }

                lock (_lock)
                {
                    _latestTopNParams = topParams;
                    _currentGeneration = gen + 1;
                    _bestFitness = MathF.Max(_bestFitness, maxFit);
                    _landingRate = rate;
                    _rateHistory.Add(rate);
                    _fitHistory.Add(MathF.Max(_bestFitness, maxFit));

                    if (maxLandingCount >= actualSpawns && _solvedGeneration < 0)
                    {
                        _solvedGeneration = gen + 1;
                        GD.Print($"  *** SOLVED at Gen {_solvedGeneration}! ***");
                    }
                }

                if (gen % 10 == 0)
                    GD.Print($"  Gen {gen,3}: fit={MathF.Max(eliteFitness, maxFit),8:F1}  land={landings}/{totalPop * actualSpawns} ({rate:F1}%)");
            }

            lock (_lock) _trainingDone = true;
            GD.Print("  Training complete.");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"Training thread crashed: {ex}");
        }
    }

    void StartNewReplay()
    {
        if (!_gpuReady || _currentReplayParams == null) return;

        int paramCount = _topology.TotalParams;
        int numIndividuals = _currentReplayParams.Length / paramCount;
        int numSpawns = NumSpawns;
        _activeReplayCount = numIndividuals * numSpawns;

        _vizEval!.PrepareMultiIndividualSpreadReplay(
            _currentReplayParams, numIndividuals, numSpawns, baseSeed: _replayStep);

        SetupRocketVisuals(_activeReplayCount);
        _replayActive = true;
        _replayStep = 0;
    }

    int _replaySeed;

    public override void _Process(double delta)
    {
        if (!_gpuReady) return;

        var vp = GetViewportRect().Size;
        _controlsLabel.Position = new Vector2(10, vp.Y - 30);

        // Check for new generation from training thread
        float[]? pendingParams = null;
        int dispGen; float dispFit, dispRate; bool dispDone; int dispSolved;
        lock (_lock)
        {
            dispGen = _currentGeneration;
            dispFit = _bestFitness;
            dispRate = _landingRate;
            dispDone = _trainingDone;
            dispSolved = _solvedGeneration;

            if (_latestTopNParams != null && !ReferenceEquals(_latestTopNParams, _currentReplayParams))
                pendingParams = _latestTopNParams;
        }

        // Switch to new generation when current replay finishes
        if (pendingParams != null && !_replayActive && _postTerminalFrames <= 0)
        {
            _currentReplayParams = pendingParams;
            _replaySeed++;
            StartNewReplay();
        }
        else if (!_replayActive && _currentReplayParams != null && _postTerminalFrames <= 0)
        {
            _replaySeed++;
            StartNewReplay();
        }

        // Step physics
        if (_replayActive && _bodies != null)
        {
            bool allTerminal = CheckAllTerminal();

                if (!allTerminal)
            {
                for (int s = 0; s < _simSpeed; s++)
                {
                    _vizEval!.StepReplay();
                    _replayStep++;
                }
                _vizEval!.ReadMultiReplayState(out _bodies, out _geoms,
                    out _terminal, out _landed, out _throttle, out _gimbal);

                UpdateTrails();
                allTerminal = CheckAllTerminal();
            }

            if (allTerminal && _replayActive)
            {
                _replayActive = false;
                _postTerminalFrames = 60; // ~1s pause before next
            }
        }

        if (_postTerminalFrames > 0) _postTerminalFrames--;

        UpdateVisuals();

        // HUD
        int landedCount = 0;
        if (_landed != null)
            for (int i = 0; i < _activeReplayCount; i++)
                if (_landed[i] != 0) landedCount++;

        string status = dispDone ? "DONE" : $"Gen {dispGen}";
        if (dispSolved > 0) status += $" (solved@{dispSolved})";
        _hudLabel.Text = $"{status}  Fit: {dispFit:F0}  Land: {dispRate:F0}%  Viz: {landedCount}/{_activeReplayCount}  Speed: {_simSpeed}x";
        _hudLabel.AddThemeColorOverride("font_color",
            dispSolved > 0 ? Colors.LimeGreen : (dispGen > 0 ? Colors.White : Colors.Gray));
    }

    bool CheckAllTerminal()
    {
        if (_terminal == null || _replayStep == 0) return false;
        for (int i = 0; i < _activeReplayCount; i++)
            if (_terminal[i] == 0) return false;
        return true;
    }

    void UpdateTrails()
    {
        if (_bodies == null) return;
        for (int r = 0; r < _activeReplayCount && r < _trailNodes.Length; r++)
        {
            if (_terminal![r] != 0 && _trailNodes[r].GetPointCount() > 0) continue;
            var com = GetCOM(r);
            _trailNodes[r].AddPoint(ToGodot(com.X, com.Y));
            if (_trailNodes[r].GetPointCount() > 600)
                _trailNodes[r].RemovePoint(0);
        }
    }

    // --- Input ---

    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventKey key && key.Pressed)
        {
            switch (key.Keycode)
            {
                case Key.R:
                    // TODO: restart training
                    break;
                case Key.Space:
                    _simSpeed = _simSpeed > 0 ? 0 : 2;
                    break;
                case Key.Equal or Key.KpAdd:
                    _simSpeed = Math.Min(_simSpeed + 1, 8);
                    break;
                case Key.Minus or Key.KpSubtract:
                    _simSpeed = Math.Max(_simSpeed - 1, 0);
                    break;
                case Key.Escape:
                    Cleanup();
                    GetTree().Quit();
                    break;
            }
        }

        if (@event is InputEventMouseButton mb && mb.Pressed)
        {
            var zoom = _camera.Zoom;
            if (mb.ButtonIndex == MouseButton.WheelUp)
                _camera.Zoom = zoom * 1.1f;
            else if (mb.ButtonIndex == MouseButton.WheelDown)
                _camera.Zoom = zoom / 1.1f;
        }
    }

    // --- World visuals (static, built once) ---

    void BuildWorldVisuals()
    {
        foreach (var child in _worldLayer.GetChildren()) child.QueueFree();

        float groundY = _simWorld.GroundY;
        var ground = CreateRect(60f, 1f, new Color(0.2f, 0.2f, 0.25f));
        ground.Position = ToGodot(0f, groundY - 0.5f);
        _worldLayer.AddChild(ground);

        var lp = _simWorld.LandingPad!;
        var pad = CreateRect(lp.PadHalfWidth * 2f, lp.PadHalfHeight * 2f, new Color(0.85f, 0.15f, 0.1f));
        pad.Position = ToGodot(lp.PadX, lp.PadY);
        _worldLayer.AddChild(pad);

        if (_simWorld.Obstacles != null)
            foreach (var obs in _simWorld.Obstacles)
            {
                var poly = CreateOBBPolygon(obs.HalfExtentX, obs.HalfExtentY, new Color(0.35f, 0.35f, 0.4f));
                poly.Position = ToGodot(obs.CX, obs.CY);
                poly.Rotation = -MathF.Atan2(obs.UY, obs.UX);
                _worldLayer.AddChild(poly);
            }

        if (_simWorld.Checkpoints != null)
            foreach (var cp in _simWorld.Checkpoints)
            {
                var c = CreateCircleOutline(cp.Radius, new Color(0.2f, 0.85f, 0.3f, 0.5f));
                c.Position = ToGodot(cp.X, cp.Y);
                _worldLayer.AddChild(c);
            }

        if (_simWorld.DangerZones != null)
            foreach (var dz in _simWorld.DangerZones)
            {
                var p = CreateRect(dz.HalfExtentX * 2f, dz.HalfExtentY * 2f, new Color(0.9f, 0.15f, 0.15f, 0.25f));
                p.Position = ToGodot(dz.X, dz.Y);
                _worldLayer.AddChild(p);
            }

        if (_simWorld.Attractors != null)
            foreach (var att in _simWorld.Attractors)
            {
                var p = CreateRect(att.HalfExtentX * 2f, att.HalfExtentY * 2f, new Color(0.2f, 0.4f, 0.9f, 0.25f));
                p.Position = ToGodot(att.X, att.Y);
                _worldLayer.AddChild(p);
            }
    }

    // --- Rocket visuals (rebuilt each replay) ---

    void SetupRocketVisuals(int count)
    {
        foreach (var child in _rocketsLayer.GetChildren()) child.QueueFree();

        _rocketGeomNodes = new Node2D[count][];
        _trailNodes = new Line2D[count];
        _flameNodes = new Line2D[count];

        for (int r = 0; r < count; r++)
        {
            var trail = new Line2D { Width = 0.04f, DefaultColor = new Color(0.3f, 0.7f, 1f, 0.2f), Antialiased = true };
            _rocketsLayer.AddChild(trail);
            _trailNodes[r] = trail;

            var flame = new Line2D { Width = 0.12f, DefaultColor = new Color(1f, 0.5f, 0.1f, 0.9f), Visible = false };
            _rocketsLayer.AddChild(flame);
            _flameNodes[r] = flame;

            _rocketGeomNodes[r] = new Node2D[GeomsPerWorld];
            for (int g = 0; g < GeomsPerWorld; g++)
            {
                bool isCapsule = g < GeomsPerBody0;
                float radius = isCapsule ? 0.2f : 0.1f;
                Color color = isCapsule ? new Color(0.9f, 0.9f, 0.95f) : new Color(1f, 0.6f, 0.2f);
                var circle = CreateCircleFilled(radius, color);
                _rocketsLayer.AddChild(circle);
                _rocketGeomNodes[r][g] = circle;
            }
        }

        // Initial read
        _vizEval!.ReadMultiReplayState(out _bodies, out _geoms,
            out _terminal, out _landed, out _throttle, out _gimbal);
    }

    void UpdateVisuals()
    {
        if (_bodies == null || _geoms == null) return;

        // Camera: fit all rockets
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        for (int r = 0; r < _activeReplayCount; r++)
        {
            var com = GetCOM(r);
            minX = MathF.Min(minX, com.X); maxX = MathF.Max(maxX, com.X);
            minY = MathF.Min(minY, com.Y); maxY = MathF.Max(maxY, com.Y);
        }
        minY = MathF.Min(minY, _simWorld.GroundY);

        if (minX < float.MaxValue)
        {
            _camera.Position = ToGodot((minX + maxX) / 2f, (minY + maxY) / 2f);
            var vp = GetViewportRect().Size;
            float spanX = maxX - minX + 10f;
            float spanY = maxY - minY + 8f;
            float targetZoom = MathF.Min(vp.X / spanX, vp.Y / spanY);
            targetZoom = Math.Clamp(targetZoom, 5f, 60f);
            float cur = _camera.Zoom.X;
            float smoothed = cur + (targetZoom - cur) * 0.05f;
            _camera.Zoom = new Vector2(smoothed, smoothed);
        }

        for (int r = 0; r < _activeReplayCount && r < _rocketGeomNodes.Length; r++)
        {
            bool terminal = _terminal![r] != 0;
            bool landed = _landed![r] != 0;
            float alpha = terminal ? 0.3f : 1f;

            for (int b = 0; b < BodiesPerWorld; b++)
            {
                var body = _bodies[r * BodiesPerWorld + b];
                float cos = MathF.Cos(body.Angle);
                float sin = MathF.Sin(body.Angle);

                int localStart = b == 0 ? 0 : (b == 1 ? GeomsPerBody0 : GeomsPerBody0 + GeomsPerLeg);
                int localCount = b == 0 ? GeomsPerBody0 : GeomsPerLeg;

                for (int g = 0; g < localCount; g++)
                {
                    var geom = _geoms[body.GeomStartIndex + g];
                    float wx = body.X + geom.LocalX * cos - geom.LocalY * sin;
                    float wy = body.Y + geom.LocalX * sin + geom.LocalY * cos;

                    var node = _rocketGeomNodes[r][localStart + g];
                    node.Position = ToGodot(wx, wy);
                    node.Modulate = landed ? Colors.LimeGreen
                        : terminal ? new Color(0.5f, 0.2f, 0.2f, alpha)
                        : Colors.White;
                }
            }

            // Flame
            float thr = _throttle![r];
            var flame = _flameNodes[r];
            if (thr > 0.05f && !terminal)
            {
                var capsule = _bodies[r * BodiesPerWorld];
                float c2 = MathF.Cos(capsule.Angle), s2 = MathF.Sin(capsule.Angle);
                float bx = capsule.X - c2 * BodyHalfLength, by = capsule.Y - s2 * BodyHalfLength;
                float tx = bx - c2 * thr * 2.5f, ty = by - s2 * thr * 2.5f;
                flame.ClearPoints();
                flame.AddPoint(ToGodot(bx, by));
                flame.AddPoint(ToGodot(tx, ty));
                flame.Width = 0.08f + thr * 0.12f;
                flame.Visible = true;
            }
            else flame.Visible = false;

            _trailNodes[r].DefaultColor = landed
                ? new Color(0.2f, 0.9f, 0.2f, 0.15f)
                : new Color(0.3f, 0.7f, 1f, 0.12f * alpha);
        }
    }

    (float X, float Y) GetCOM(int r)
    {
        float cx = 0, cy = 0, tm = 0;
        for (int b = 0; b < BodiesPerWorld; b++)
        {
            var body = _bodies![r * BodiesPerWorld + b];
            if (body.InvMass > 0)
            {
                float m = 1f / body.InvMass;
                cx += body.X * m; cy += body.Y * m; tm += m;
            }
        }
        return tm > 0 ? (cx / tm, cy / tm) : (0, 0);
    }

    // --- Helpers ---

    static Vector2 ToGodot(float wx, float wy) => new(wx, -wy);

    static Polygon2D CreateRect(float w, float h, Color color)
    {
        float hw = w / 2f, hh = h / 2f;
        return new Polygon2D
        {
            Polygon = [new(-hw, -hh), new(hw, -hh), new(hw, hh), new(-hw, hh)],
            Color = color
        };
    }

    static Polygon2D CreateOBBPolygon(float hx, float hy, Color color)
        => new() { Polygon = [new(-hx, -hy), new(hx, -hy), new(hx, hy), new(-hx, hy)], Color = color };

    static Polygon2D CreateCircleFilled(float radius, Color color, int segs = 16)
    {
        var pts = new Vector2[segs];
        for (int i = 0; i < segs; i++)
        {
            float a = i * MathF.Tau / segs;
            pts[i] = new(MathF.Cos(a) * radius, MathF.Sin(a) * radius);
        }
        return new Polygon2D { Polygon = pts, Color = color };
    }

    static Line2D CreateCircleOutline(float radius, Color color, int segs = 32)
    {
        var line = new Line2D { DefaultColor = color, Width = 0.05f, Antialiased = true };
        for (int i = 0; i <= segs; i++)
        {
            float a = i * MathF.Tau / segs;
            line.AddPoint(new(MathF.Cos(a) * radius, MathF.Sin(a) * radius));
        }
        return line;
    }

    void Cleanup()
    {
        _trainCts.Cancel();
        _trainingThread?.Join(3000);
        _vizEval?.Dispose();
    }

    public override void _ExitTree() => Cleanup();
}
