using Godot;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Evolvatron.Core;
using Evolvatron.Core.Physics;
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
    const int GeomsPerBody0 = 5;
    const int GeomsPerLeg = 7;
    const int GeomsPerWorld = GeomsPerBody0 + GeomsPerLeg * 2;
    const float BodyHalfLength = 0.75f;
    const float InitialZoom = 30f;
    const int MaxReplayCount = 10;
    const int MaxGenerations = 300;
    const int DefaultRollouts = 10;
    const float ReplaySettleSpeedThreshold = 0.12f;
    const float ReplaySettleAngVelThreshold = 0.2f;
    const int ReplaySettleStepsRequired = 90;

    // --- Shared training state (accessed under _lock) ---
    readonly object _lock = new();
    float[]? _latestTopNParams;
    float[]? _latestTopNFitness;
    int _currentGeneration;
    float _bestFitness;
    float _landingRate;
    int _landingCount;
    int _landingEvalCount;
    bool _trainingDone;
    int _solvedGeneration = -1;
    readonly System.Collections.Generic.List<float> _rateHistory = new();
    readonly System.Collections.Generic.List<float> _fitHistory = new();
    int _trainPop;

    // --- Viz state (render thread only) ---
    SimWorld _simWorld = null!;
    DenseTopology _topology = null!;
    CancellationTokenSource _trainCts = new();
    Thread? _trainingThread;

    float[]? _currentReplayParams;
    float[]? _currentReplayFitness;
    float[] _replayAlpha = [];
    bool _replayActive;
    int _replayStep;
    int _postTerminalFrames;
    int _activeReplayCount;
    int _replayPolicyCount;
    int _replaySpawnCount;

    GPURigidBody[]? _bodies;
    GPURigidBodyGeom[]? _geoms;
    byte[]? _terminal;
    byte[]? _landed;
    float[]? _throttle;
    float[]? _gimbal;
    CpuReplayWorld[] _replays = [];

    // --- Godot nodes ---
    Camera2D _camera = null!;
    Node2D _worldLayer = null!;
    Node2D _rocketsLayer = null!;
    Label _hudLabel = null!;
    Label _controlsLabel = null!;
    Label _statusLabel = null!;
    Button _goButton = null!;
    HSlider _positionWeightSlider = null!;
    HSlider _velocityWeightSlider = null!;
    HSlider _angleWeightSlider = null!;
    HSlider _controlWeightSlider = null!;
    HSlider _spawnWidthSlider = null!;
    HSlider _rolloutsSlider = null!;
    HSlider _obstacleXSlider = null!;
    CheckBox _obstacleEnabledCheck = null!;

    Node2D[][] _rocketGeomNodes = [];
    Line2D[] _trailNodes = [];
    Line2D[] _flameNodes = [];
    Color[] _policyColors = [];

    int _simSpeed = 2;
    bool _gpuReady;

    sealed class CpuReplayWorld
    {
        public WorldState World = null!;
        public CPUStepper Stepper = new();
        public int[] Rocket = [];
        public float[] Weights = [];
        public float[] Biases = [];
        public float CurrentThrottle;
        public float CurrentGimbal;
        public bool IsTerminal;
        public bool HasLanded;
        public int ContactSteps;
        public int RestSteps;
        public int Steps;
    }

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
        _controlsLabel.Text = "[Go/R] Train   [+/-] Speed   [Space] Pause   [Esc] Quit";
        hud.AddChild(_controlsLabel);

        BuildControlPanel(hud);

        try
        {
            RestartTrainingFromControls();
        }
        catch (Exception ex)
        {
            _hudLabel.Text = $"Init failed:\n{ex.Message}";
            GD.PrintErr($"Init failed: {ex}");
        }
    }

    void BuildControlPanel(CanvasLayer hud)
    {
        var panel = new PanelContainer();
        panel.Position = new Vector2(10, 48);
        panel.CustomMinimumSize = new Vector2(320, 0);
        hud.AddChild(panel);

        var box = new VBoxContainer();
        panel.AddChild(box);

        var title = new Label();
        title.Text = "Training Setup";
        title.AddThemeFontSizeOverride("font_size", 16);
        box.AddChild(title);

        _positionWeightSlider = AddSlider(box, "Position", 0f, 3f, 1f);
        _velocityWeightSlider = AddSlider(box, "Velocity", 0f, 3f, 0.8f);
        _angleWeightSlider = AddSlider(box, "Angle", 0f, 3f, 0.6f);
        _controlWeightSlider = AddSlider(box, "Control", 0f, 1f, 0.05f);
        _spawnWidthSlider = AddSlider(box, "Spawn Width", 2f, 16f, 8f);
        _rolloutsSlider = AddSlider(box, "Rollouts", 1f, 30f, DefaultRollouts, 1f, "0");
        _obstacleXSlider = AddSlider(box, "Obstacle X", -8f, 8f, -2.5f);

        _obstacleEnabledCheck = new CheckBox { Text = "Obstacle", ButtonPressed = true };
        _obstacleEnabledCheck.Toggled += _ => UpdateSetupStatus();
        box.AddChild(_obstacleEnabledCheck);

        _goButton = new Button { Text = "Go" };
        _goButton.Pressed += RestartTrainingFromControls;
        box.AddChild(_goButton);

        _statusLabel = new Label();
        _statusLabel.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        _statusLabel.AddThemeColorOverride("font_color", new Color(0.7f, 0.7f, 0.75f));
        box.AddChild(_statusLabel);

        UpdateSetupStatus();
    }

    HSlider AddSlider(VBoxContainer parent, string label, float min, float max, float value, float step = 0.05f, string format = "0.00")
    {
        var row = new HBoxContainer();
        parent.AddChild(row);

        var name = new Label { Text = label, CustomMinimumSize = new Vector2(95, 0) };
        row.AddChild(name);

        var valueLabel = new Label { CustomMinimumSize = new Vector2(42, 0) };
        row.AddChild(valueLabel);

        var slider = new HSlider
        {
            MinValue = min,
            MaxValue = max,
            Step = step,
            Value = value,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill
        };
        slider.ValueChanged += v =>
        {
            valueLabel.Text = v.ToString(format);
            UpdateSetupStatus();
        };
        row.AddChild(slider);
        valueLabel.Text = value.ToString(format);
        return slider;
    }

    void UpdateSetupStatus()
    {
        if (_statusLabel == null) return;
        int rollouts = _rolloutsSlider == null ? DefaultRollouts : GetRolloutCount();
        _statusLabel.Text =
            $"Fixed rocket, CEM trainer, {_spawnWidthSlider?.Value ?? 8:0.0}m spawn band, {rollouts} rollouts. " +
            (_obstacleEnabledCheck?.ButtonPressed == true ? "Obstacle sensors on." : "No obstacle sensors.");
    }

    int GetRolloutCount() => Math.Clamp((int)MathF.Round((float)_rolloutsSlider.Value), 1, 30);

    public void SetGoButtonEnabled(bool enabled)
    {
        if (_goButton != null)
            _goButton.Disabled = !enabled;
    }

    void RestartTrainingFromControls()
    {
        CleanupTrainingOnly();
        _simWorld = BuildWorldFromControls();
        int sensorCount = _simWorld.SimulationConfig?.SensorCount ?? 0;
        _topology = DenseTopology.ForRocket(new[] { 16, 12 }, sensorCount: sensorCount);

        GD.Print($"LiveTrainer: topology={_topology}, sensors={sensorCount}");

        // Build static world visuals
        BuildWorldVisuals();

        _gpuReady = true;

        // Reset shared state
        lock (_lock)
        {
            _latestTopNParams = null;
            _latestTopNFitness = null;
            _currentGeneration = 0;
            _bestFitness = 0f;
            _landingRate = 0f;
            _landingCount = 0;
            _landingEvalCount = 0;
            _trainingDone = false;
            _solvedGeneration = -1;
            _rateHistory.Clear();
            _fitHistory.Clear();
        }
        _currentReplayParams = null;
        _currentReplayFitness = null;
        _replayAlpha = [];
        _replayActive = false;
        _postTerminalFrames = 0;
        _activeReplayCount = 0;
        _replayPolicyCount = 0;
        _replaySpawnCount = 0;
        _bodies = null;
        _geoms = null;
        foreach (var child in _rocketsLayer.GetChildren()) child.QueueFree();

        // Start training thread
        _trainCts = new CancellationTokenSource();
        var ct = _trainCts.Token;
        _trainingThread = new Thread(() => RunTraining(ct)) { IsBackground = true };
        _trainingThread.Start();
        _goButton.Disabled = true;

        // Position camera to show world
        float cx = _simWorld.LandingPad?.PadX ?? 0f;
        float cy = (_simWorld.GroundY + (_simWorld.Spawn?.Y ?? 15f)) / 2f;
        _camera.Position = ToGodot(cx, cy);
    }

    SimWorld BuildWorldFromControls()
    {
        float spawnWidth = (float)_spawnWidthSlider.Value;
        int rollouts = GetRolloutCount();
        float obstacleX = (float)_obstacleXSlider.Value;
        bool useObstacle = _obstacleEnabledCheck.ButtonPressed;

        return new SimWorld
        {
            GroundY = -5f,
            LandingPad = new SimLandingPad
            {
                PadX = 0f,
                PadY = -4.5f,
                PadHalfWidth = 2f,
                PadHalfHeight = 0.25f,
                LandingBonus = 160f,
                MaxLandingVelocity = 2f,
                MaxLandingAngle = 15f * MathF.PI / 180f,
                AttractionMagnitude = 0f,
                AttractionRadius = 0f
            },
            Spawn = new SimSpawn
            {
                X = 0f,
                Y = 14f,
                XRange = spawnWidth,
                HeightRange = 2f,
                AngleRange = 20f * MathF.PI / 180f,
                VelXRange = 1.5f,
                VelYMax = 2f,
                SpawnCount = rollouts,
                SpawnSeed = 0
            },
            Obstacles = useObstacle
                ? [new SimObstacle
                {
                    CX = obstacleX,
                    CY = 4.0f,
                    UX = 1f,
                    UY = 0f,
                    HalfExtentX = 1.6f,
                    HalfExtentY = 0.35f,
                    IsLethal = true,
                    PenaltyPerStep = 0f,
                    InfluenceRadius = 0f
                }]
                : [],
            Checkpoints = [],
            SpeedZones = [],
            DangerZones = [],
            Attractors = [],
            SimulationConfig = new SimSimulationConfig
            {
                Dt = 1f / 120f,
                GravityY = -9.81f,
                FrictionMu = 0.8f,
                Restitution = 0f,
                GlobalDamping = 0.02f,
                AngularDamping = 0.1f,
                SolverIterations = 6,
                MaxThrust = 200f,
                MaxGimbalAngle = 15f * MathF.PI / 180f,
                SensorCount = useObstacle ? 4 : 0,
                MaxSteps = 600,
                HasteBonus = 1f
            },
            RewardWeights = new SimRewardWeights
            {
                PositionWeight = (float)_positionWeightSlider.Value,
                VelocityWeight = (float)_velocityWeightSlider.Value,
                AngleWeight = (float)_angleWeightSlider.Value,
                AngularVelocityWeight = 0.1f,
                ControlEffortWeight = (float)_controlWeightSlider.Value
            }
        };
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
            int actualSpawns = trainEval.SpawnCount > 0 ? trainEval.SpawnCount : DefaultRollouts;
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
                    _latestTopNFitness = topFitness;
                    _currentGeneration = gen + 1;
                    _bestFitness = MathF.Max(_bestFitness, maxFit);
                    _landingRate = rate;
                    _landingCount = landings;
                    _landingEvalCount = totalPop * actualSpawns;
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
            CallDeferred(nameof(SetGoButtonEnabled), true);
            GD.Print("  Training complete.");
        }
        catch (Exception ex)
        {
            CallDeferred(nameof(SetGoButtonEnabled), true);
            GD.PrintErr($"Training thread crashed: {ex}");
        }
    }

    void StartNewReplay()
    {
        if (!_gpuReady || _currentReplayParams == null) return;

        int paramCount = _topology.TotalParams;
        int numIndividuals = _currentReplayParams.Length / paramCount;
        int numSpawns = _simWorld.Spawn.SpawnCount > 0 ? _simWorld.Spawn.SpawnCount : DefaultRollouts;
        _replayPolicyCount = numIndividuals;
        _replaySpawnCount = numSpawns;
        _activeReplayCount = numIndividuals * numSpawns;

        SetupRocketVisuals(_activeReplayCount);
        BuildReplayAlpha(numIndividuals, numSpawns);
        SetupCpuReplays(_currentReplayParams, numIndividuals, numSpawns, _replaySeed);
        _replayActive = true;
        _replayStep = 0;
    }

    void BuildReplayAlpha(int numIndividuals, int numSpawns)
    {
        _replayAlpha = new float[numIndividuals * numSpawns];

        if (_currentReplayFitness == null || _currentReplayFitness.Length < numIndividuals)
        {
            Array.Fill(_replayAlpha, 1f);
            return;
        }

        float best = _currentReplayFitness[0];
        float worst = _currentReplayFitness[0];
        for (int i = 1; i < numIndividuals; i++)
        {
            best = MathF.Max(best, _currentReplayFitness[i]);
            worst = MathF.Min(worst, _currentReplayFitness[i]);
        }

        for (int i = 0; i < numIndividuals; i++)
        {
            float rankT = numIndividuals <= 1 ? 1f : 1f - (float)i / (numIndividuals - 1);
            float scoreT = MathF.Abs(best - worst) < 1e-5f
                ? rankT
                : Math.Clamp((_currentReplayFitness[i] - worst) / (best - worst), 0f, 1f);
            float t = MathF.Max(rankT, scoreT);

            float alpha = 0.06f + 0.94f * MathF.Pow(t, 1.8f);
            int topCut = Math.Max(1, (int)MathF.Ceiling(numIndividuals * 0.05f));
            int bottomCut = Math.Max(1, (int)MathF.Ceiling(numIndividuals * 0.05f));
            if (i < topCut) alpha = 1f;
            if (i >= numIndividuals - bottomCut) alpha = 0.06f;

            for (int s = 0; s < numSpawns; s++)
                _replayAlpha[i * numSpawns + s] = alpha;
        }
    }

    void SetupCpuReplays(float[] flatParams, int numIndividuals, int numSpawns, int baseSeed)
    {
        int total = numIndividuals * numSpawns;
        int paramCount = _topology.TotalParams;
        _replays = new CpuReplayWorld[total];
        _bodies = new GPURigidBody[total * BodiesPerWorld];
        _geoms = new GPURigidBodyGeom[total * GeomsPerWorld];
        _terminal = new byte[total];
        _landed = new byte[total];
        _throttle = new float[total];
        _gimbal = new float[total];

        for (int i = 0; i < numIndividuals; i++)
        {
            var oneParams = new float[paramCount];
            Array.Copy(flatParams, i * paramCount, oneParams, 0, paramCount);
            var (weights, biases) = CpuDenseNN.SplitParams(oneParams, _topology.LayerSizes);

            for (int s = 0; s < numSpawns; s++)
            {
                int r = i * numSpawns + s;
                var replay = CreateReplayWorld(baseSeed + s, weights, biases);
                _replays[r] = replay;
                CopyReplayToRenderArrays(replay, r);
            }
        }
    }

    int _replaySeed;

    public override void _Process(double delta)
    {
        if (!_gpuReady) return;

        var vp = GetViewportRect().Size;
        _controlsLabel.Position = new Vector2(10, vp.Y - 30);

        // Check for new generation from training thread
        float[]? pendingParams = null;
        float[]? pendingFitness = null;
        int dispGen; float dispFit, dispRate; bool dispDone; int dispSolved; int dispLanded; int dispEvals;
        lock (_lock)
        {
            dispGen = _currentGeneration;
            dispFit = _bestFitness;
            dispRate = _landingRate;
            dispLanded = _landingCount;
            dispEvals = _landingEvalCount;
            dispDone = _trainingDone;
            dispSolved = _solvedGeneration;

            if (_latestTopNParams != null && !ReferenceEquals(_latestTopNParams, _currentReplayParams))
            {
                pendingParams = _latestTopNParams;
                pendingFitness = _latestTopNFitness;
            }
        }

        // Switch to new generation when current replay finishes
        if (pendingParams != null && !_replayActive && _postTerminalFrames <= 0)
        {
            _currentReplayParams = pendingParams;
            _currentReplayFitness = pendingFitness;
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
                    StepCpuReplays();
                    _replayStep++;
                }

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
        int terminalCount = 0;
        if (_landed != null)
            for (int i = 0; i < _activeReplayCount; i++)
            {
                if (_terminal != null && _terminal[i] != 0) terminalCount++;
            }

        string status = dispDone ? "DONE" : $"Gen {dispGen}";
        if (dispSolved > 0) status += $" (solved@{dispSolved})";
        string trainLand = dispEvals > 0
            ? $"{dispLanded}/{dispEvals} ({dispRate:F2}%)"
            : "0/0";
        string replayScope = _replayPolicyCount > 0
            ? $"top {_replayPolicyCount} x {_replaySpawnCount}"
            : "top policies";
        _hudLabel.Text = $"{status}  Fit: {dispFit:F0}  Train all: {trainLand}  Replay {replayScope}: stopped {terminalCount}/{_activeReplayCount}  Speed: {_simSpeed}x";
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

    void StepCpuReplays()
    {
        var cfg = CreateReplayConfig();
        Span<float> obs = stackalloc float[_topology.InputSize];
        Span<float> actions = stackalloc float[_topology.OutputSize];

        for (int r = 0; r < _replays.Length; r++)
        {
            var replay = _replays[r];
            if (replay.IsTerminal)
            {
                CopyReplayToRenderArrays(replay, r);
                continue;
            }

            FillObservations(replay, obs);
            CpuDenseNN.ForwardPass(replay.Weights, replay.Biases, _topology.LayerSizes, obs, actions);

            float throttle = Math.Clamp(actions[0], 0f, 1f);
            float gimbal = Math.Clamp(actions[1], -1f, 1f);
            replay.CurrentThrottle = throttle;
            replay.CurrentGimbal = gimbal;

            ApplyReplayControls(replay, throttle, gimbal, cfg.Dt);
            replay.Stepper.Step(replay.World, cfg);
            replay.Steps++;
            UpdateReplayTerminal(replay);
            CopyReplayToRenderArrays(replay, r);
        }
    }

    SimulationConfig CreateReplayConfig() => new()
    {
        Dt = 1f / 120f,
        GravityX = 0f,
        GravityY = -9.81f,
        FrictionMu = 0.8f,
        Restitution = 0f,
        GlobalDamping = 0.02f,
        AngularDamping = 0.1f,
        XpbdIterations = _simWorld.SimulationConfig.SolverIterations
    };

    CpuReplayWorld CreateReplayWorld(int seed, float[] weights, float[] biases)
    {
        var rng = new Random(seed);
        var replay = new CpuReplayWorld
        {
            World = new WorldState(),
            Weights = weights,
            Biases = biases
        };

        replay.World.Obbs.Add(OBBCollider.AxisAligned(0f, _simWorld.GroundY, 30f, 0.5f));
        replay.World.Obbs.Add(OBBCollider.AxisAligned(
            _simWorld.LandingPad.PadX,
            _simWorld.LandingPad.PadY,
            _simWorld.LandingPad.PadHalfWidth,
            _simWorld.LandingPad.PadHalfHeight));
        foreach (var obs in _simWorld.Obstacles)
            replay.World.Obbs.Add(new OBBCollider(obs.CX, obs.CY, obs.UX, obs.UY, obs.HalfExtentX, obs.HalfExtentY));

        float sx = _simWorld.Spawn.X + (float)(rng.NextDouble() * _simWorld.Spawn.XRange - _simWorld.Spawn.XRange * 0.5);
        float sy = _simWorld.Spawn.Y - _simWorld.Spawn.HeightRange * 0.5f + (float)rng.NextDouble() * _simWorld.Spawn.HeightRange;
        float st = (float)(rng.NextDouble() * _simWorld.Spawn.AngleRange * 2 - _simWorld.Spawn.AngleRange);
        float svx = (float)(rng.NextDouble() * _simWorld.Spawn.VelXRange * 2 - _simWorld.Spawn.VelXRange);
        float svy = (float)(rng.NextDouble() * -_simWorld.Spawn.VelYMax);

        replay.Rocket = CreateReplayRocket(replay.World, sx, sy, st, svx, svy);
        return replay;
    }

    int[] CreateReplayRocket(WorldState world, float spawnX, float spawnY, float spawnTilt, float velX, float velY)
    {
        const float bodyHeight = 1.5f, bodyRadius = 0.2f, bodyMass = 8f;
        const float legLength = 1.0f, legRadius = 0.1f, legMass = 1.5f;
        float bodyHalfLength = bodyHeight * 0.5f;
        float legHalfLength = legLength * 0.5f;
        float bodyInertia = bodyMass * (bodyRadius * bodyRadius * 0.25f + bodyHeight * bodyHeight / 12f);
        float legInertia = legMass * (legRadius * legRadius * 0.25f + legLength * legLength / 12f);

        const float bodyAngle = MathF.PI / 2f;
        float leftLegAngle = 225f * MathF.PI / 180f;
        float rightLegAngle = 315f * MathF.PI / 180f;
        float cosT = MathF.Cos(spawnTilt);
        float sinT = MathF.Sin(spawnTilt);

        int bodyGeomStart = world.RigidBodyGeoms.Count;
        for (int i = 0; i < GeomsPerBody0; i++)
        {
            float t = (float)i / (GeomsPerBody0 - 1);
            world.RigidBodyGeoms.Add(new RigidBodyGeom(-bodyHalfLength + t * bodyHeight, 0f, bodyRadius));
        }
        int body = world.RigidBodies.Count;
        world.RigidBodies.Add(new RigidBody(spawnX, spawnY + bodyHalfLength * cosT, bodyAngle + spawnTilt,
            bodyMass, bodyInertia, bodyGeomStart, GeomsPerBody0) { VelX = velX, VelY = velY });

        int leftGeomStart = world.RigidBodyGeoms.Count;
        for (int i = 0; i < GeomsPerLeg; i++)
        {
            float t = (float)i / (GeomsPerLeg - 1);
            world.RigidBodyGeoms.Add(new RigidBodyGeom(-legHalfLength + t * legLength, 0f, legRadius));
        }
        float leftOffX = MathF.Cos(leftLegAngle) * legHalfLength;
        float leftOffY = MathF.Sin(leftLegAngle) * legHalfLength;
        int leftLeg = world.RigidBodies.Count;
        world.RigidBodies.Add(new RigidBody(
            spawnX + leftOffX * cosT - leftOffY * sinT,
            spawnY + leftOffX * sinT + leftOffY * cosT,
            leftLegAngle + spawnTilt, legMass, legInertia, leftGeomStart, GeomsPerLeg) { VelX = velX, VelY = velY });

        int rightGeomStart = world.RigidBodyGeoms.Count;
        for (int i = 0; i < GeomsPerLeg; i++)
        {
            float t = (float)i / (GeomsPerLeg - 1);
            world.RigidBodyGeoms.Add(new RigidBodyGeom(-legHalfLength + t * legLength, 0f, legRadius));
        }
        float rightOffX = MathF.Cos(rightLegAngle) * legHalfLength;
        float rightOffY = MathF.Sin(rightLegAngle) * legHalfLength;
        int rightLeg = world.RigidBodies.Count;
        world.RigidBodies.Add(new RigidBody(
            spawnX + rightOffX * cosT - rightOffY * sinT,
            spawnY + rightOffX * sinT + rightOffY * cosT,
            rightLegAngle + spawnTilt, legMass, legInertia, rightGeomStart, GeomsPerLeg) { VelX = velX, VelY = velY });

        world.RevoluteJoints.Add(new RevoluteJoint(body, leftLeg, -bodyHalfLength, 0f, -legHalfLength, 0f)
        {
            ReferenceAngle = leftLegAngle - bodyAngle,
            EnableMotor = true,
            MotorSpeed = 0f,
            MaxMotorTorque = 1000f
        });
        world.RevoluteJoints.Add(new RevoluteJoint(body, rightLeg, -bodyHalfLength, 0f, -legHalfLength, 0f)
        {
            ReferenceAngle = rightLegAngle - bodyAngle,
            EnableMotor = true,
            MotorSpeed = 0f,
            MaxMotorTorque = 1000f
        });

        return [body, leftLeg, rightLeg];
    }

    void FillObservations(CpuReplayWorld replay, Span<float> obs)
    {
        GetReplayCom(replay, out float comX, out float comY, out float velX, out float velY);
        var body = replay.World.RigidBodies[replay.Rocket[0]];
        float upX = MathF.Cos(body.Angle);
        float upY = MathF.Sin(body.Angle);

        obs[0] = (comX - _simWorld.LandingPad.PadX) / 20f;
        obs[1] = (comY - _simWorld.LandingPad.PadY) / 20f;
        obs[2] = velX / 10f;
        obs[3] = velY / 10f;
        obs[4] = upX;
        obs[5] = upY;
        obs[6] = replay.CurrentGimbal;
        obs[7] = replay.CurrentThrottle;

        if (_simWorld.SimulationConfig.SensorCount >= 4)
        {
            float maxRange = 30f;
            float d0 = maxRange, d1 = maxRange, d2 = maxRange, d3 = maxRange;
            foreach (var obb in replay.World.Obbs)
            {
                d0 = MathF.Min(d0, RayVsOBB(comX, comY, upX, upY, maxRange, obb));
                d1 = MathF.Min(d1, RayVsOBB(comX, comY, -upX, -upY, maxRange, obb));
                d2 = MathF.Min(d2, RayVsOBB(comX, comY, -upY, upX, maxRange, obb));
                d3 = MathF.Min(d3, RayVsOBB(comX, comY, upY, -upX, maxRange, obb));
            }
            obs[8] = d0 / maxRange;
            obs[9] = d1 / maxRange;
            obs[10] = d2 / maxRange;
            obs[11] = d3 / maxRange;
        }
    }

    void ApplyReplayControls(CpuReplayWorld replay, float throttle, float gimbal, float dt)
    {
        var body = replay.World.RigidBodies[replay.Rocket[0]];
        float thrust = throttle * _simWorld.SimulationConfig.MaxThrust;
        float cos = MathF.Cos(body.Angle);
        float sin = MathF.Sin(body.Angle);
        body.VelX += cos * thrust * body.InvMass * dt;
        body.VelY += sin * thrust * body.InvMass * dt;
        body.AngularVel += gimbal * 50f * body.InvInertia * dt;
        replay.World.RigidBodies[replay.Rocket[0]] = body;
    }

    void UpdateReplayTerminal(CpuReplayWorld replay)
    {
        GetReplayCom(replay, out float comX, out float comY, out float velX, out float velY);
        var body = replay.World.RigidBodies[replay.Rocket[0]];
        float speed = MathF.Sqrt(velX * velX + velY * velY);
        float angleErr = MathF.Abs(NormalizeAngle(body.Angle - MathF.PI / 2f));
        bool nearPad = MathF.Abs(comX - _simWorld.LandingPad.PadX) < _simWorld.LandingPad.PadHalfWidth
            && MathF.Abs(comY - _simWorld.LandingPad.PadY) < 2f;

        DetectReplayContacts(replay, out bool hasContact, out bool hitObstacle);

        if (hitObstacle && _simWorld.Obstacles.Any(o => o.IsLethal))
        {
            replay.IsTerminal = true;
            replay.HasLanded = false;
            return;
        }

        if (hasContact && speed > _simWorld.LandingPad.MaxLandingVelocity)
        {
            replay.IsTerminal = true;
            replay.HasLanded = false;
            return;
        }

        if (!hasContact && angleErr > MathF.PI * 0.5f)
        {
            replay.IsTerminal = true;
            replay.HasLanded = false;
            return;
        }

        float dist = MathF.Sqrt(
            (comX - _simWorld.LandingPad.PadX) * (comX - _simWorld.LandingPad.PadX) +
            (comY - _simWorld.LandingPad.PadY) * (comY - _simWorld.LandingPad.PadY));
        if (dist > 50f || comY < _simWorld.GroundY - 10f || comY > _simWorld.Spawn.Y + 30f)
        {
            replay.IsTerminal = true;
            replay.HasLanded = false;
            return;
        }

        if (hasContact)
        {
            replay.ContactSteps++;
            if (speed < ReplaySettleSpeedThreshold && MathF.Abs(body.AngularVel) < ReplaySettleAngVelThreshold)
                replay.RestSteps++;
            else
                replay.RestSteps = 0;

            replay.HasLanded = nearPad && angleErr < 45f * MathF.PI / 180f && replay.RestSteps >= ReplaySettleStepsRequired;
        }
        else
        {
            replay.ContactSteps = 0;
            replay.RestSteps = 0;
            replay.HasLanded = false;
        }

        if (replay.Steps >= _simWorld.SimulationConfig.MaxSteps)
        {
            replay.IsTerminal = true;
            replay.HasLanded = false;
        }
    }

    static void DetectReplayContacts(CpuReplayWorld replay, out bool hasContact, out bool hitObstacle)
    {
        hasContact = false;
        hitObstacle = false;

        foreach (int bodyIndex in replay.Rocket)
        {
            var body = replay.World.RigidBodies[bodyIndex];
            for (int g = 0; g < body.GeomCount; g++)
            {
                var geom = replay.World.RigidBodyGeoms[body.GeomStartIndex + g];
                CircleCollision.TransformGeomToWorld(body, geom, out float gx, out float gy);

                for (int obbIndex = 0; obbIndex < replay.World.Obbs.Count; obbIndex++)
                {
                    if (!CircleCollision.CircleVsStaticOBB(gx, gy, geom.Radius, replay.World.Obbs[obbIndex], out _))
                        continue;

                    hasContact = true;
                    if (obbIndex >= 2)
                        hitObstacle = true;
                }
            }
        }
    }

    void CopyReplayToRenderArrays(CpuReplayWorld replay, int r)
    {
        int bodyBase = r * BodiesPerWorld;
        for (int i = 0; i < BodiesPerWorld; i++)
        {
            var rb = replay.World.RigidBodies[replay.Rocket[i]];
            _bodies![bodyBase + i] = new GPURigidBody
            {
                X = rb.X,
                Y = rb.Y,
                Angle = rb.Angle,
                VelX = rb.VelX,
                VelY = rb.VelY,
                AngularVel = rb.AngularVel,
                InvMass = rb.InvMass,
                InvInertia = rb.InvInertia,
                GeomStartIndex = r * GeomsPerWorld + rb.GeomStartIndex,
                GeomCount = rb.GeomCount
            };

            for (int g = 0; g < rb.GeomCount; g++)
            {
                var geom = replay.World.RigidBodyGeoms[rb.GeomStartIndex + g];
                _geoms![r * GeomsPerWorld + rb.GeomStartIndex + g] = new GPURigidBodyGeom
                {
                    LocalX = geom.LocalX,
                    LocalY = geom.LocalY,
                    Radius = geom.Radius,
                    BodyIndex = i
                };
            }
        }
        _terminal![r] = replay.IsTerminal ? (byte)1 : (byte)0;
        _landed![r] = replay.HasLanded ? (byte)1 : (byte)0;
        _throttle![r] = replay.CurrentThrottle;
        _gimbal![r] = replay.CurrentGimbal;
    }

    static void GetReplayCom(CpuReplayWorld replay, out float comX, out float comY, out float velX, out float velY)
    {
        comX = comY = velX = velY = 0f;
        float totalMass = 0f;
        foreach (int idx in replay.Rocket)
        {
            var rb = replay.World.RigidBodies[idx];
            if (rb.InvMass <= 0f) continue;
            float mass = 1f / rb.InvMass;
            comX += rb.X * mass;
            comY += rb.Y * mass;
            velX += rb.VelX * mass;
            velY += rb.VelY * mass;
            totalMass += mass;
        }
        if (totalMass > 0f)
        {
            float inv = 1f / totalMass;
            comX *= inv;
            comY *= inv;
            velX *= inv;
            velY *= inv;
        }
    }

    static float RayVsOBB(float ox, float oy, float dx, float dy, float maxRange, OBBCollider obb)
    {
        float relX = ox - obb.CX, relY = oy - obb.CY;
        float px = -obb.UY, py = obb.UX;
        float localOX = relX * obb.UX + relY * obb.UY;
        float localOY = relX * px + relY * py;
        float localDX = dx * obb.UX + dy * obb.UY;
        float localDY = dx * px + dy * py;

        float tMin = 0f, tMax = maxRange;
        if (!ClipRaySlab(localOX, localDX, obb.HalfExtentX, ref tMin, ref tMax)) return maxRange;
        if (!ClipRaySlab(localOY, localDY, obb.HalfExtentY, ref tMin, ref tMax)) return maxRange;
        return tMin > 0f ? tMin : (tMax > 0f ? tMax : maxRange);
    }

    static bool ClipRaySlab(float origin, float dir, float halfExtent, ref float tMin, ref float tMax)
    {
        if (MathF.Abs(dir) < 1e-8f)
            return origin >= -halfExtent && origin <= halfExtent;

        float t1 = (-halfExtent - origin) / dir;
        float t2 = (halfExtent - origin) / dir;
        if (t1 > t2) (t1, t2) = (t2, t1);
        tMin = MathF.Max(tMin, t1);
        tMax = MathF.Min(tMax, t2);
        return tMin <= tMax;
    }

    static float NormalizeAngle(float a)
    {
        while (a > MathF.PI) a -= MathF.Tau;
        while (a < -MathF.PI) a += MathF.Tau;
        return a;
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
                    RestartTrainingFromControls();
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
        var pad = CreateRect(lp.PadHalfWidth * 2f, lp.PadHalfHeight * 2f, new Color(0.15f, 0.8f, 0.25f));
        pad.Position = ToGodot(lp.PadX, lp.PadY);
        _worldLayer.AddChild(pad);

        if (_simWorld.Obstacles != null)
            foreach (var obs in _simWorld.Obstacles)
            {
                var poly = CreateOBBPolygon(obs.HalfExtentX, obs.HalfExtentY,
                    obs.IsLethal ? new Color(0.75f, 0.08f, 0.08f) : new Color(0.35f, 0.35f, 0.4f));
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
        _policyColors = new Color[Math.Max(1, _replayPolicyCount)];

        for (int r = 0; r < count; r++)
        {
            int policy = _replaySpawnCount > 0 ? r / _replaySpawnCount : r;
            Color familyColor = GetPolicyColor(policy);
            var trail = new Line2D { Width = 0.04f, DefaultColor = WithAlpha(familyColor, 0.2f), Antialiased = true };
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
            float performanceAlpha = r < _replayAlpha.Length ? _replayAlpha[r] : 1f;
            float alpha = terminal && !landed
                ? MathF.Max(0.025f, performanceAlpha * 0.28f)
                : performanceAlpha;
            int policy = _replaySpawnCount > 0 ? r / _replaySpawnCount : r;
            Color familyColor = GetPolicyColor(policy);

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
                    node.Modulate = landed ? WithAlpha(familyColor.Lerp(Colors.White, 0.35f), alpha)
                        : terminal ? new Color(0.5f, 0.2f, 0.2f, alpha)
                        : WithAlpha(familyColor, alpha);
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
                flame.DefaultColor = new Color(1f, 0.5f, 0.1f, MathF.Min(0.9f, alpha));
                flame.Visible = true;
            }
            else flame.Visible = false;

            _trailNodes[r].DefaultColor = landed
                ? WithAlpha(familyColor.Lerp(Colors.White, 0.35f), 0.10f * performanceAlpha)
                : WithAlpha(familyColor, 0.10f * alpha);
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

    Color GetPolicyColor(int policy)
    {
        if (_policyColors.Length == 0)
            return Colors.White;

        policy = Math.Clamp(policy, 0, _policyColors.Length - 1);
        if (_policyColors[policy].A > 0f)
            return _policyColors[policy];

        float hue = (policy * 0.61803398875f) % 1f;
        var color = Color.FromHsv(hue, 0.72f, 0.95f);
        _policyColors[policy] = color;
        return color;
    }

    // --- Helpers ---

    static Vector2 ToGodot(float wx, float wy) => new(wx, -wy);

    static Color WithAlpha(Color c, float alpha) => new(c.R, c.G, c.B, alpha);

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
        CleanupTrainingOnly();
    }

    void CleanupTrainingOnly()
    {
        _trainCts.Cancel();
        _trainingThread?.Join(3000);
    }

    public override void _ExitTree() => Cleanup();
}
