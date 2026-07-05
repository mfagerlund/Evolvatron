using Godot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Evolvatron.Core;
using Evolvatron.Core.Rockets;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.World;

namespace Evolvatron.Godot;

/// <summary>
/// Freeform rocket constructor + controller trainer (see docs/godot_pipeline_plan.md, P1–P3).
///
/// EDIT mode: edit a <see cref="RocketSpec"/> — toolbar adds bodies/sensors and deletes the
/// selection; click a body to select, drag to reposition.
///
/// TRAIN (G): runs in-process CEM (<see cref="GPUDenseRocketPoseEvaluator"/> + <see
/// cref="IslandOptimizer"/>) on a background thread to learn a goal-relative POSE-REACHING controller
/// (fly to and briefly hold a target position + orientation), AND shows a live swarm of the
/// current-best controller chasing random target poses (CPU replay via <see cref="CpuDenseNN"/>),
/// re-seeded and refreshed every generation — visual feedback while training. The metric is "hit %"
/// (held the pose within pos/angle/speed tolerance). Press G again to stop and keep the controller.
/// Editing the rocket invalidates the controller and resets the score. Training is gated to the STOCK
/// rocket (arbitrary rockets need GPU spec support, P2). Physical note: a single bottom thruster can
/// only hover near-upright, so target angles are sampled within ±60° of upright.
///
/// FLY (T): place a TARGET (position + orientation); the rocket flies there using its TRAINED
/// controller, which now genuinely honors the requested orientation (it is a network input). Move the
/// target — the rocket re-homes. No controller ⇒ real random-weight network ⇒ honest chaos.
/// </summary>
public partial class RocketEditorScene : Node2D
{
    const float GroundY = 0f;
    const float FlySpawnY = 4f;
    const float EditSpawnY = 4f;
    const float Zoom = 32f;
    const float PickTolerance = 0.18f;
    const float TargetRingR = 0.6f;
    const float TargetHandleR = 1.6f;
    const float Gravity = 9.81f;
    const int MaxTrainGenerations = 300;
    const int TrainVizCount = 8;          // simultaneous champion attempts shown while training
    const int TrainVizMaxSteps = 560;
    const string SavePath = "user://rocket_spec.json";
    static readonly int[] PoseHidden = { 24, 16 };   // [10 -> 24 -> 16 -> 2] pose-reaching controller

    // Pose-viz geometry (mirrors GPUDenseRocketPoseEvaluator: free-space spawn at origin, random target pose).
    const float PoseTargetXRange = 6f, PoseTargetYRange = 5f;
    const float PoseAngleCenter = MathF.PI / 2f, PoseAngleRange = 60f * MathF.PI / 180f;
    const float PoseHitRadius = 1.2f, PoseHitAngleTol = 15f * MathF.PI / 180f, PoseHitSpeedTol = 1.5f;
    const int PoseHoldSteps = 30;
    const float PoseVizFloorY = -12f;   // distant backdrop floor (training is free-space, no collider)
    const float PoseSpawnTilt = 0.3f, PoseSpawnSpeed = 2f, PoseSpawnAngVel = 0.4f;

    RocketSpec _spec = null!;
    WorldState _world = null!;
    CPUStepper _stepper = new();
    int[] _rocket = Array.Empty<int>();
    int _steps;
    bool _flying;
    bool _trainViz;            // showing the live training swarm (during training, and lingering after)
    int _simSpeed = 2;
    float _curThrottle, _curGimbal;

    float _targetX = 3f, _targetY = 5f, _targetAngle = MathF.PI / 2f;

    enum Drag { None, Body, TargetPos, TargetAngle }
    Drag _drag = Drag.None;
    int _selected = -1;
    float _dragOffX, _dragOffY;
    string _status = "";

    sealed class Controller
    {
        public float[] Params = Array.Empty<float>();
        public DenseTopology Topology = null!;
        public float Score;
        public int Signature;
    }
    Controller? _controller;
    DenseTopology _trainTopo = null!;
    int _stockSignature;

    // Cached CPU NN weights for the current FLY run.
    float[] _flyWeights = Array.Empty<float>();
    float[] _flyBiases = Array.Empty<float>();
    int[] _flyLayerSizes = Array.Empty<int>();
    bool _flyTrained;
    bool _flyDirty = true;

    // ---- training-viz swarm ---------------------------------------------------
    sealed class Rollout
    {
        public WorldState World = null!;
        public CPUStepper Stepper = new();
        public int[] Rocket = Array.Empty<int>();
        public float Throttle, Gimbal;
        public float TargetX, TargetY, TargetAngle;
        public int Step;
        public int HoldCount;
        public bool Done;
        public bool Hit;
        public int Pause;
        public readonly List<Vector2> Trail = new();
    }
    readonly List<Vector2> _flyTrail = new();
    Rollout[] _tv = Array.Empty<Rollout>();
    float[] _tvW = Array.Empty<float>();
    float[] _tvB = Array.Empty<float>();
    int[] _tvLS = Array.Empty<int>();
    object? _tvChampRef;
    bool _tvIsRandom;
    int _tvSeedBase = 1000;
    int _tvRespawnCounter;

    // ---- training thread (shared under _lock) ---------------------------------
    readonly object _lock = new();
    Thread? _trainThread;
    CancellationTokenSource? _trainCts;
    bool _training;
    int _trainGen, _lastSeenGen;
    float _trainBestFit;
    float _trainRate;
    bool _trainDone;
    string? _trainError;
    float[]? _trainChampion;
    int _dispGen; float _dispFit, _dispRate;

    // Long-lived GPU evaluator. ILGPU's CUDA Context/Accelerator must NOT be created and torn down
    // per training run inside the live Godot process — destroying a CUDA context while Godot's own
    // renderer holds the GPU segfaults (signal 11). Create it once, reuse across runs, dispose only
    // at scene exit. See _ExitTree.
    GPUDenseRocketPoseEvaluator? _eval;

    string? _shotPath;
    int _shotFrame;

    Camera2D _camera = null!;
    Label _hud = null!;
    Label _info = null!;
    Label _ctrl = null!;
    Label _controls = null!;

    SimulationConfig Config => new()
    {
        Dt = 1f / 120f,
        GravityX = 0f,
        GravityY = -Gravity,
        FrictionMu = 0.8f,
        Restitution = 0f,
        GlobalDamping = 0.02f,
        AngularDamping = 0.1f,
        XpbdIterations = 6
    };

    int _frame;

    // Env-gated crash breadcrumbs: set EVOLV_TRACE=1 to print the phase being entered each frame, so
    // the last line before a segfault pinpoints where it died (draw vs step-viz vs completion).
    static readonly bool _trace = System.Environment.GetEnvironmentVariable("EVOLV_TRACE") == "1";
    static void Trace(string m) { if (_trace) GD.Print($"[trace] {m}"); }

    static readonly Color Accent = new(0.40f, 0.78f, 0.98f);
    static readonly Color AccentDim = new(0.20f, 0.42f, 0.58f);
    static readonly Color TextCol = new(0.86f, 0.92f, 1f);
    static readonly Color PanelBg = new(0.05f, 0.07f, 0.13f, 0.82f);

    public override void _Ready()
    {
        RenderingServer.SetDefaultClearColor(new Color(0.015f, 0.02f, 0.06f));

        var skyLayer = new CanvasLayer { Layer = -100 };
        AddChild(skyLayer);
        skyLayer.AddChild(new SkyBackground());

        _camera = new Camera2D { Zoom = new Vector2(Zoom, Zoom) };
        AddChild(_camera);
        _camera.MakeCurrent();
        _camera.Position = ToGodot(0f, EditSpawnY);

        _trainTopo = DenseTopology.ForRocketPose(PoseHidden);
        BuildUi();

        _spec = RocketSpecLibrary.StockRocket();
        _stockSignature = Signature(_spec);
        RebuildWorld();
        _status = "Loaded stock rocket. Edit it, Train the controller (G), then Fly (T).";

        bool startFlying = false, startTraining = false;
        foreach (var a in OS.GetCmdlineUserArgs())
        {
            if (a.StartsWith("--shot=")) _shotPath = a.Substring("--shot=".Length);
            if (a == "--fly") startFlying = true;
            if (a == "--train") startTraining = true;
        }
        if (startTraining) ToggleTraining();
        if (startFlying) ToggleFlying();
    }

    void BuildUi()
    {
        var ui = new CanvasLayer();
        AddChild(ui);

        var title = new Label
        {
            Position = new Vector2(0, 8),
            Text = "EVOLVATRON  ·  ROCKET  EDITOR",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        title.AddThemeColorOverride("font_color", Accent);
        title.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0, 0.85f));
        title.AddThemeConstantOverride("outline_size", 5);
        title.AddThemeFontSizeOverride("font_size", 22);
        title.AnchorRight = 1f;
        ui.AddChild(title);

        var panel = new PanelContainer { Position = new Vector2(12, 44) };
        panel.AddThemeStyleboxOverride("panel", MakePanelBox());
        ui.AddChild(panel);
        var margin = new MarginContainer();
        foreach (var side in new[] { "margin_left", "margin_right", "margin_top", "margin_bottom" })
            margin.AddThemeConstantOverride(side, 8);
        panel.AddChild(margin);
        var toolbar = new VBoxContainer();
        toolbar.AddThemeConstantOverride("separation", 5);
        margin.AddChild(toolbar);
        toolbar.AddChild(MakeButton("Add Body", AddBody));
        toolbar.AddChild(MakeButton("Add Sensor", AddSensor));
        toolbar.AddChild(MakeButton("Delete Selected", DeleteSelected));
        toolbar.AddChild(MakeButton("◈  Train Controller  [G]", ToggleTraining));
        toolbar.AddChild(MakeButton("▲  Fly to Target  [T]", ToggleFlying));
        toolbar.AddChild(MakeButton("Reset  [R]", ResetAction));
        toolbar.AddChild(MakeButton("Save  [S]", SaveSpec));
        toolbar.AddChild(MakeButton("Load  [L]", LoadSpec));

        _info = MakeLabel(new Vector2(220, 48), new Color(0.72f, 0.82f, 0.92f), 14);
        ui.AddChild(_info);
        _ctrl = MakeLabel(new Vector2(220, 88), new Color(0.98f, 0.82f, 0.45f), 16);
        ui.AddChild(_ctrl);

        _hud = MakeLabel(new Vector2(12, -50), Colors.White, 15);
        _hud.AnchorTop = 1f; _hud.AnchorBottom = 1f;
        ui.AddChild(_hud);
        _controls = MakeLabel(new Vector2(12, -26), new Color(0.62f, 0.68f, 0.78f), 13);
        _controls.AnchorTop = 1f; _controls.AnchorBottom = 1f;
        ui.AddChild(_controls);
        UpdateControlsHint();
    }

    static Label MakeLabel(Vector2 pos, Color color, int size)
    {
        var l = new Label { Position = pos };
        l.AddThemeColorOverride("font_color", color);
        l.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0, 0.85f));
        l.AddThemeConstantOverride("outline_size", 4);
        l.AddThemeFontSizeOverride("font_size", size);
        return l;
    }

    static StyleBoxFlat MakePanelBox()
    {
        var sb = new StyleBoxFlat { BgColor = PanelBg, BorderColor = AccentDim };
        sb.SetBorderWidthAll(1);
        sb.SetCornerRadiusAll(8);
        return sb;
    }

    static StyleBoxFlat MakeButtonBox(Color bg, Color border)
    {
        var sb = new StyleBoxFlat { BgColor = bg, BorderColor = border };
        sb.SetBorderWidthAll(1);
        sb.SetCornerRadiusAll(6);
        sb.ContentMarginLeft = 12; sb.ContentMarginRight = 12;
        sb.ContentMarginTop = 6; sb.ContentMarginBottom = 6;
        return sb;
    }

    void UpdateControlsHint()
    {
        if (_training || _trainViz)
            _controls.Text = "TRAIN: watching the swarm learn   [G] stop & keep   [R] back to edit   " +
                             "[+/-] speed   [wheel] zoom   [Esc] quit";
        else if (_flying)
            _controls.Text = "FLY: [drag target] move goal   [drag handle] rotate goal   [T] edit   " +
                             "[G] train   [R] respawn   [+/-] speed   [wheel] zoom   [Esc] quit";
        else
            _controls.Text = "EDIT: [click] select   [drag] move body   [G] train   [T] fly   " +
                             "[R] reset   [S/L] save/load   [wheel] zoom   [Esc] quit";
    }

    Button MakeButton(string text, Action onPressed)
    {
        var b = new Button { Text = text, CustomMinimumSize = new Vector2(182, 0) };
        b.AddThemeStyleboxOverride("normal", MakeButtonBox(new Color(0.10f, 0.14f, 0.22f, 0.95f), AccentDim));
        b.AddThemeStyleboxOverride("hover", MakeButtonBox(new Color(0.16f, 0.25f, 0.38f, 0.98f), Accent));
        b.AddThemeStyleboxOverride("pressed", MakeButtonBox(new Color(0.22f, 0.34f, 0.50f, 1f), Accent));
        b.AddThemeStyleboxOverride("focus", new StyleBoxEmpty());
        b.AddThemeColorOverride("font_color", TextCol);
        b.AddThemeColorOverride("font_hover_color", Colors.White);
        b.AddThemeFontSizeOverride("font_size", 15);
        b.Pressed += () => { onPressed(); b.ReleaseFocus(); };
        return b;
    }

    void RebuildWorld()
    {
        try { _spec.Validate(); }
        catch (Exception ex) { _status = $"Invalid spec: {ex.Message}"; }

        _world = new WorldState();
        if (_flying)
            _world.Obbs.Add(OBBCollider.AxisAligned(0f, GroundY - 0.5f, 30f, 0.5f));
        float spawnY = _flying ? FlySpawnY : EditSpawnY;
        _rocket = RocketSpecFactory.ToCpuWorld(_spec, _world, 0f, spawnY);
        _stepper = new CPUStepper();
        _steps = 0;
        _curThrottle = 0f; _curGimbal = 0f;
        _flyDirty = true;
        _flyTrail.Clear();
        if (_selected >= _spec.BodyCount) _selected = -1;
    }

    void ResetAction()
    {
        if (_training) { RespawnTrainViz(); _status = "Re-seeded the training swarm."; }
        else if (_trainViz) { _trainViz = false; RebuildWorld(); UpdateControlsHint(); _status = "Back to edit."; }
        else { RebuildWorld(); _status = _flying ? "Respawned." : "Reset."; }
    }

    void ToggleFlying()
    {
        if (_training) { _status = "Stop training (G) first, then Fly your placed target."; return; }
        _flying = !_flying;
        _trainViz = false;
        _drag = Drag.None;
        RebuildWorld();
        UpdateControlsHint();
        _status = _flying
            ? (_flyTrainedForSpec ? "FLY: drag the target; the trained controller chases it."
                                  : "FLY: NO controller — random net = chaos. Train first (G).")
            : "Editing.";
    }

    bool _flyTrainedForSpec => _controller != null && _controller.Signature == Signature(_spec);
    bool CanTrain => Signature(_spec) == _stockSignature;

    public override void _Process(double delta)
    {
        _frame++;
        if (_training) { Trace("poll"); PollTraining(); }

        if (_trainViz)
        {
            Trace("stepviz");
            StepTrainViz();
            _camera.Position = ToGodot(0f, 0.5f);
        }
        else if (_flying && _rocket.Length > 0)
        {
            if (_flyDirty) RecacheFlyWeights();
            var cfg = Config;
            for (int s = 0; s < _simSpeed; s++) FlyStep(cfg);
            var (cx, cy) = Com();
            _camera.Position = ToGodot(cx, cy);
        }
        else
        {
            var (cx, cy) = Com();
            _camera.Position = ToGodot(cx, cy);
        }

        UpdateHud();
        QueueRedraw();
        HandleShot();
    }

    void UpdateHud()
    {
        if (_trainViz)
        {
            string head = _training
                ? $"TRAINING  gen {_dispGen}/{MaxTrainGenerations}  best {_dispFit:F0}  hit {_dispRate:F0}%"
                : $"RESULT  score {(_controller?.Score ?? 0):F0}";
            string what = _tvIsRandom ? "random net (no champion yet)" : "current-best controller";
            _hud.Text = $"{head}   {_simSpeed}x   watching {TrainVizCount} attempts of the {what}   {_status}";
        }
        else if (_flying)
        {
            var (cx, cy) = Com();
            float dist = Dist(_targetX, _targetY, cx, cy);
            string chaos = _flyTrained ? "" : "UNTRAINED — CHAOS   ";
            _hud.Text = $"{chaos}FLY   step {_steps}   {_simSpeed}x   dist {dist:0.0}m   " +
                        $"target ({_targetX:0.0},{_targetY:0.0}) @ {Mathf.RadToDeg(_targetAngle):0}°   {_status}";
        }
        else
        {
            _hud.Text = $"EDIT   {_status}";
        }

        string sel = _selected >= 0 && _selected < _spec.BodyCount
            ? $"selected body #{_selected}: geoms {_spec.Bodies[_selected].Geoms.Count}  " +
              $"mass {_spec.Bodies[_selected].Mass:0.##}  pos ({_spec.Bodies[_selected].X:0.##},{_spec.Bodies[_selected].Y:0.##})"
            : "no selection";
        _info.Text = $"bodies {_spec.BodyCount}  geoms {_spec.TotalGeoms}  joints {_spec.JointCount}  " +
                     $"sensors {_spec.SensorCount}  thrusters {_spec.Thrusters.Count}\n{sel}";

        _ctrl.Text = ControllerStatusText();
    }

    string ControllerStatusText()
    {
        if (_training)
            return $"Controller: TRAINING  gen {_dispGen}/{MaxTrainGenerations}  best {_dispFit:F0}  hit {_dispRate:F0}%   (G to stop & keep)";
        if (_flyTrainedForSpec)
            return $"Controller: TRAINED  score {_controller!.Score:F0}   (Fly to use it)";
        if (CanTrain)
            return "Controller: NONE (untrained — Fly = chaos).  Press G to train.";
        return "Controller: NONE.  Constructed rocket — training needs P2 (GPU spec support).";
    }

    // ---- fly: run the actual controller network -------------------------------

    void RecacheFlyWeights()
    {
        _flyDirty = false;
        if (_flyTrainedForSpec)
        {
            (_flyWeights, _flyBiases) = CpuDenseNN.SplitParams(_controller!.Params, _controller.Topology.LayerSizes);
            _flyLayerSizes = _controller.Topology.LayerSizes;
            _flyTrained = true;
        }
        else
        {
            (_flyWeights, _flyBiases) = CpuDenseNN.SplitParams(RandomParams(_trainTopo, 12345), _trainTopo.LayerSizes);
            _flyLayerSizes = _trainTopo.LayerSizes;
            _flyTrained = false;
        }
    }

    void FlyStep(SimulationConfig cfg)
    {
        Span<float> obs = stackalloc float[_flyLayerSizes[0]];
        Span<float> act = stackalloc float[_flyLayerSizes[^1]];
        BuildObs(_world, _rocket, _targetX, _targetY, _targetAngle, _curThrottle, _curGimbal, obs);
        CpuDenseNN.ForwardPass(_flyWeights, _flyBiases, _flyLayerSizes, obs, act);
        float thr = Math.Clamp(act[0], 0f, 1f);
        float gim = act.Length > 1 ? Math.Clamp(act[1], -1f, 1f) : 0f;
        _curThrottle = thr; _curGimbal = gim;
        ApplyThrustersTo(_world, _rocket, thr, gim, cfg.Dt);
        _stepper.Step(_world, cfg);
        _steps++;

        var (tcx, tcy) = Com();
        _flyTrail.Add(new Vector2(tcx, tcy));
        if (_flyTrail.Count > 44) _flyTrail.RemoveAt(0);
    }

    // ---- training-viz swarm (CPU replay of the live champion) ------------------

    void StartTrainViz()
    {
        _flying = false;
        _trainViz = true;
        _drag = Drag.None;
        _tvChampRef = null; _tvIsRandom = false;
        EnsureTrainVizWeights();
        _tvRespawnCounter = 0;
        _tv = new Rollout[TrainVizCount];
        for (int i = 0; i < TrainVizCount; i++)
        {
            _tv[i] = new Rollout();
            SpawnRollout(_tv[i], _tvSeedBase + i);
        }
        UpdateControlsHint();
    }

    void RespawnTrainViz()
    {
        if (_tv.Length == 0) { StartTrainViz(); return; }
        for (int i = 0; i < _tv.Length; i++)
            SpawnRollout(_tv[i], _tvSeedBase + (++_tvRespawnCounter) * 31 + i);
    }

    void EnsureTrainVizWeights()
    {
        if (_controller != null && _controller.Signature == _stockSignature)
        {
            if (!ReferenceEquals(_controller.Params, _tvChampRef))
            {
                (_tvW, _tvB) = CpuDenseNN.SplitParams(_controller.Params, _controller.Topology.LayerSizes);
                _tvLS = _controller.Topology.LayerSizes;
                _tvChampRef = _controller.Params;
                _tvIsRandom = false;
            }
        }
        else if (!_tvIsRandom || _tvLS.Length == 0)
        {
            (_tvW, _tvB) = CpuDenseNN.SplitParams(RandomParams(_trainTopo, 777), _trainTopo.LayerSizes);
            _tvLS = _trainTopo.LayerSizes;
            _tvIsRandom = true;
            _tvChampRef = null;
        }
    }

    void StepTrainViz()
    {
        EnsureTrainVizWeights();
        if (_tvLS.Length == 0 || _tv.Length == 0) return;
        var cfg = Config;
        foreach (var ro in _tv)
        {
            if (ro.Done)
            {
                if (--ro.Pause <= 0) SpawnRollout(ro, _tvSeedBase + (++_tvRespawnCounter) * 31 + 7);
                continue;
            }
            for (int s = 0; s < _simSpeed && !ro.Done; s++) StepRolloutOnce(ro, cfg);
        }
    }

    void StepRolloutOnce(Rollout ro, SimulationConfig cfg)
    {
        Span<float> obs = stackalloc float[_tvLS[0]];
        Span<float> act = stackalloc float[_tvLS[^1]];
        BuildObs(ro.World, ro.Rocket, ro.TargetX, ro.TargetY, ro.TargetAngle, ro.Throttle, ro.Gimbal, obs);
        CpuDenseNN.ForwardPass(_tvW, _tvB, _tvLS, obs, act);
        float thr = Math.Clamp(act[0], 0f, 1f);
        float gim = act.Length > 1 ? Math.Clamp(act[1], -1f, 1f) : 0f;
        ro.Throttle = thr; ro.Gimbal = gim;
        ApplyThrustersTo(ro.World, ro.Rocket, thr, gim, cfg.Dt);
        ro.Stepper.Step(ro.World, cfg);
        ro.Step++;

        var (cx, cy) = ComOf(ro.World, ro.Rocket);
        ro.Trail.Add(new Vector2(cx, cy));
        if (ro.Trail.Count > 30) ro.Trail.RemoveAt(0);

        // Hit detection: held the target pose (pos + angle + slow) for PoseHoldSteps frames.
        var (vx, vy) = ComVelOf(ro.World, ro.Rocket);
        float dx = cx - ro.TargetX, dy = cy - ro.TargetY;
        float speed = MathF.Sqrt(vx * vx + vy * vy);
        var b0 = ro.World.RigidBodies[ro.Rocket[0]];
        float ad = b0.Angle - ro.TargetAngle;
        float angErr = MathF.Abs(MathF.Atan2(MathF.Sin(ad), MathF.Cos(ad)));
        bool inTol = dx * dx + dy * dy < PoseHitRadius * PoseHitRadius
                  && angErr < PoseHitAngleTol && speed < PoseHitSpeedTol;
        if (inTol)
        {
            if (++ro.HoldCount >= PoseHoldSteps) { ro.Hit = true; ro.Done = true; ro.Pause = 48; }
        }
        else ro.HoldCount = 0;

        float distFromTarget = MathF.Sqrt(dx * dx + dy * dy);
        // Cull on max steps, escape, OR non-finite state — `NaN > 40f` is false, so a blown-up
        // rollout would otherwise persist and feed NaN into draws/camera every frame.
        if (ro.Step >= TrainVizMaxSteps || !(distFromTarget <= 40f))
        {
            ro.Done = true;
            ro.Pause = 36;
        }
    }

    void SpawnRollout(Rollout ro, int seed)
    {
        var rng = new Random(seed);
        ro.World = new WorldState();   // free space — no colliders (matches the pose evaluator)

        // Initial state: spawn at origin with a small random perturbation (mirrors the evaluator).
        float st = (float)(rng.NextDouble() * 2 - 1) * PoseSpawnTilt;
        double vang = rng.NextDouble() * 2.0 * Math.PI;
        double vspd = rng.NextDouble() * PoseSpawnSpeed;
        float svx = (float)(Math.Cos(vang) * vspd);
        float svy = (float)(Math.Sin(vang) * vspd);

        ro.Rocket = RocketSpecFactory.ToCpuWorld(_spec, ro.World, 0f, 0f, st, svx, svy);
        ro.Stepper = new CPUStepper();

        // Random target pose, same distribution as GPUDenseRocketPoseEvaluator.SampleTargetPose.
        GPUDenseRocketPoseEvaluator.SampleTargetPose(seed, PoseTargetXRange, PoseTargetYRange,
            PoseAngleCenter, PoseAngleRange, out ro.TargetX, out ro.TargetY, out ro.TargetAngle);

        ro.Throttle = 0f; ro.Gimbal = 0f; ro.Step = 0; ro.HoldCount = 0;
        ro.Done = false; ro.Hit = false; ro.Pause = 0;
        ro.Trail.Clear();
    }

    // ---- training (in-process CEM on a background thread) ----------------------

    void ToggleTraining()
    {
        if (_training) { StopTraining(); return; }
        if (!CanTrain)
        {
            _status = "Training a constructed rocket needs P2 (GPU spec support). Only the stock rocket trains for now.";
            return;
        }
        StartTraining();
    }

    void StartTraining()
    {
        // Ensure any prior training thread has fully exited before reusing the shared GPU evaluator —
        // two RunTraining threads must never touch the one ILGPU context concurrently.
        if (_trainThread != null && _trainThread.IsAlive)
        {
            _trainCts?.Cancel();
            _trainThread.Join();
        }

        // Create the GPU evaluator once, on the main thread, and keep it alive. The constructor
        // builds the ILGPU CUDA context — do it here (not on the background thread, not per run).
        if (_eval == null)
        {
            try { _eval = CreateEvaluator(); }
            catch (Exception ex)
            {
                _status = $"GPU init failed: {ex.Message}";
                GD.PrintErr($"[editor] GPU evaluator init failed: {ex}");
                return;
            }
        }

        lock (_lock)
        {
            _trainGen = 0; _lastSeenGen = 0; _trainBestFit = float.NegativeInfinity;
            _trainRate = 0f; _trainDone = false; _trainError = null; _trainChampion = null;
        }
        _training = true;
        _trainCts = new CancellationTokenSource();
        var ct = _trainCts.Token;
        _trainThread = new Thread(() => RunTraining(ct)) { IsBackground = true };
        _trainThread.Start();
        StartTrainViz();
        _status = "Training — watch the swarm; press G to stop & keep.";
    }

    GPUDenseRocketPoseEvaluator CreateEvaluator() => new(_trainTopo)
    {
        MaxSteps = 400,
        TargetXRange = PoseTargetXRange, TargetYRange = PoseTargetYRange,
        TargetAngleCenter = PoseAngleCenter, TargetAngleRange = PoseAngleRange,
        PoseHitRadius = PoseHitRadius, PoseHitAngle = PoseHitAngleTol, PoseHitSpeed = PoseHitSpeedTol,
        PoseHoldSteps = PoseHoldSteps,
        SpawnAngleRange = PoseSpawnTilt, InitialSpeedMax = PoseSpawnSpeed, InitialAngVelMax = PoseSpawnAngVel
    };

    void StopTraining()
    {
        _trainCts?.Cancel();
        _training = false;
        CommitChampion("Stopped — controller kept");
        UpdateControlsHint();
    }

    void CommitChampion(string what)
    {
        float[]? champ; float score;
        lock (_lock) { champ = _trainChampion; score = _trainBestFit; }
        if (champ != null)
        {
            _controller = new Controller
            {
                Params = (float[])champ.Clone(),
                Topology = _trainTopo,
                Score = score,
                Signature = _stockSignature
            };
            _flyDirty = true;
            _status = $"{what} · score {score:F0}.";
        }
        else _status = "No champion produced yet.";
    }

    void PollTraining()
    {
        int gen; float fit, rate; bool done; string? err; float[]? champ;
        lock (_lock)
        {
            gen = _trainGen; fit = _trainBestFit; rate = _trainRate;
            done = _trainDone; err = _trainError; champ = _trainChampion;
        }
        _dispGen = gen; _dispFit = fit; _dispRate = rate;

        if (gen != _lastSeenGen && champ != null)
        {
            _controller = new Controller { Params = champ, Topology = _trainTopo, Score = fit, Signature = _stockSignature };
            _flyDirty = true;
            _lastSeenGen = gen;
        }
        if (err != null) { _training = false; _status = $"Training error: {err}"; UpdateControlsHint(); }
        else if (done) { Trace("train-complete-transition"); _training = false; _status = $"Training finished · score {fit:F0}."; UpdateControlsHint(); }
    }

    void RunTraining(CancellationToken ct)
    {
        try
        {
            // Reuse the long-lived evaluator created on the main thread in StartTraining. NEVER
            // dispose it here — tearing down the CUDA context inside live Godot segfaults.
            var eval = _eval!;

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
                StagnationThreshold = 9999
            };

            var optimizer = new IslandOptimizer(config, _trainTopo, eval.OptimalPopulationSize);
            var rng = new Random(42);
            int spawns = eval.SpawnCount > 0 ? eval.SpawnCount : 10;
            int pc = _trainTopo.TotalParams;
            float[]? elite = null; float eliteFit = float.NegativeInfinity;
            GD.Print($"[editor] training: pop={optimizer.TotalPopulation}, spawns={spawns}, params={pc}");

            for (int gen = 0; gen < MaxTrainGenerations && !ct.IsCancellationRequested; gen++)
            {
                var pv = optimizer.GeneratePopulation(rng);
                int pop = optimizer.TotalPopulation;
                if (elite != null) Array.Copy(elite, 0, pv, (pop - 1) * pc, pc);

                var (fitness, hits, _, _) = eval.EvaluateMultiSpawn(pv, pop, spawns, baseSeed: eval.SpawnSeed);

                for (int i = 0; i < pop; i++)
                    if (fitness[i] > eliteFit)
                    {
                        eliteFit = fitness[i];
                        elite ??= new float[pc];
                        Array.Copy(pv, i * pc, elite, 0, pc);
                    }

                optimizer.Update(fitness, pv);
                optimizer.ManageIslands(rng);
                float rate = (float)hits / (pop * spawns) * 100f;

                lock (_lock)
                {
                    _trainGen = gen + 1;
                    _trainBestFit = MathF.Max(_trainBestFit, eliteFit);
                    _trainRate = rate;
                    _trainChampion = elite != null ? (float[])elite.Clone() : null;
                }
                if (gen % 10 == 0)
                    GD.Print($"[editor] gen {gen,3}: fit={eliteFit,8:F1}  hit {hits}/{pop * spawns} ({rate:F1}%)");
            }
            lock (_lock) _trainDone = true;
            GD.Print("[editor] training complete.");
        }
        catch (Exception ex)
        {
            lock (_lock) _trainError = ex.Message;
            GD.PrintErr($"[editor] training crashed: {ex}");
        }
    }

    // ---- drawing --------------------------------------------------------------

    public override void _Draw()
    {
        Trace("draw");
        float groundY = _trainViz ? PoseVizFloorY : GroundY;
        DrawWorldBackdrop(groundY, editGrid: !_trainViz && !_flying);

        if (_trainViz) { DrawTrainViz(); Trace("draw:viz-done"); return; }

        if (_flying)
        {
            DrawTarget();
            DrawTrail(_flyTrail, new Color(0.45f, 0.78f, 1f));
        }
        if (_rocket.Length == 0) return;

        DrawRocket(_world, _rocket, 1f, _flying ? _curThrottle : -1f);

        if (_selected >= 0 && _selected < _rocket.Length)
        {
            var body = _world.RigidBodies[_rocket[_selected]];
            float c = MathF.Cos(body.Angle), s = MathF.Sin(body.Angle);
            for (int g = 0; g < body.GeomCount; g++)
            {
                var geom = _world.RigidBodyGeoms[body.GeomStartIndex + g];
                float wx = body.X + geom.LocalX * c - geom.LocalY * s;
                float wy = body.Y + geom.LocalX * s + geom.LocalY * c;
                DrawArc(ToGodot(wx, wy), geom.Radius + 0.08f, 0f, MathF.Tau, 20, new Color(1f, 0.9f, 0.3f, 0.85f), 0.04f);
            }
        }

        foreach (var j in _spec.Joints)
        {
            var bodyA = _world.RigidBodies[_rocket[j.BodyA]];
            float c = MathF.Cos(bodyA.Angle), s = MathF.Sin(bodyA.Angle);
            float wx = bodyA.X + j.AnchorAX * c - j.AnchorAY * s;
            float wy = bodyA.Y + j.AnchorAX * s + j.AnchorAY * c;
            DrawCircle(ToGodot(wx, wy), 0.06f, new Color(0.3f, 0.9f, 1f, 0.8f));
        }

        if (!_flying)
        {
            var (comX, comY) = Com();
            foreach (var sensor in _spec.Sensors)
            {
                var body = _world.RigidBodies[_rocket[sensor.BodyIndex]];
                float dir = body.Angle + sensor.AngleOffset;
                float ex = comX + MathF.Cos(dir) * 2.5f;
                float ey = comY + MathF.Sin(dir) * 2.5f;
                DrawLine(ToGodot(comX, comY), ToGodot(ex, ey), new Color(0.45f, 0.85f, 0.45f, 0.35f), 0.025f);
                DrawCircle(ToGodot(ex, ey), 0.05f, new Color(0.5f, 0.9f, 0.5f, 0.6f));
            }
        }
    }

    void DrawTrainViz()
    {
        foreach (var ro in _tv)
        {
            if (ro.Rocket.Length == 0) continue;
            Color beacon = ro.Hit ? new Color(0.4f, 1f, 0.55f)
                         : ro.Done ? new Color(0.55f, 0.58f, 0.7f)
                         : new Color(1f, 0.5f, 0.78f);
            float pulse = 0.5f + 0.5f * MathF.Sin(_frame * 0.1f + ro.TargetX);
            DrawPoseBeacon(ro.TargetX, ro.TargetY, ro.TargetAngle, beacon, pulse);
            DrawTrail(ro.Trail, ro.Done ? new Color(0.55f, 0.58f, 0.7f) : new Color(0.5f, 0.82f, 1f));
            DrawRocket(ro.World, ro.Rocket, ro.Done ? 0.45f : 1f, ro.Throttle);
        }
    }

    // A target-pose beacon: a tolerance ring + a tick pointing along the desired nose direction.
    void DrawPoseBeacon(float tx, float ty, float tAngle, Color col, float pulse)
    {
        var c = ToGodot(tx, ty);
        DrawArc(c, PoseHitRadius + 0.1f * pulse, 0f, MathF.Tau, 28,
                new Color(col.R, col.G, col.B, 0.35f + 0.4f * pulse), 0.045f);
        float nx = tx + MathF.Cos(tAngle) * 1.1f;
        float ny = ty + MathF.Sin(tAngle) * 1.1f;
        DrawLine(c, ToGodot(nx, ny), new Color(col.R, col.G, col.B, 0.85f), 0.04f);
        DrawCircle(ToGodot(nx, ny), 0.1f, col);
        DrawCircle(c, 0.06f, new Color(col.R, col.G, col.B, 0.9f));
    }

    void DrawTrail(List<Vector2> pts, Color col)
    {
        if (pts.Count < 2) return;
        for (int i = 1; i < pts.Count; i++)
        {
            float a = i / (float)pts.Count;
            DrawLine(ToGodot(pts[i - 1].X, pts[i - 1].Y), ToGodot(pts[i].X, pts[i].Y),
                     new Color(col.R, col.G, col.B, a * 0.5f), 0.018f + a * 0.03f);
        }
    }

    // Draws a sleek rocket silhouette for body 0 (fuselage + nose + nozzle + fins), tapered
    // strut legs for the remaining bodies, and a layered, throttle-scaled engine plume.
    void DrawRocket(WorldState world, int[] rocket, float alpha, float flameThrottle)
    {
        // --- legs: tapered struts with foot pads ---
        for (int b = 1; b < rocket.Length; b++)
        {
            var leg = world.RigidBodies[rocket[b]];
            float lc = MathF.Cos(leg.Angle), ls = MathF.Sin(leg.Angle);
            Vector2 LW(float lx, float ly) => ToGodot(leg.X + lx * lc - ly * ls, leg.Y + lx * ls + ly * lc);
            GeomExtent(world, leg, out float lmin, out float lmax, out float lrad);
            DrawColoredPolygon(new[]
            {
                LW(lmin, lrad * 0.6f), LW(lmax, lrad * 1.1f),
                LW(lmax, -lrad * 1.1f), LW(lmin, -lrad * 0.6f)
            }, new Color(0.50f, 0.54f, 0.63f, alpha));
            DrawCircle(LW(lmax + lrad * 0.5f, 0f), lrad * 1.6f, new Color(0.66f, 0.70f, 0.80f, alpha));
        }

        // --- fuselage (body 0) ---
        var b0 = world.RigidBodies[rocket[0]];
        float c = MathF.Cos(b0.Angle), s = MathF.Sin(b0.Angle);
        Vector2 W(float lx, float ly) => ToGodot(b0.X + lx * c - ly * s, b0.Y + lx * s + ly * c);
        GeomExtent(world, b0, out float bmin, out float bmax, out float brad);
        float hw = brad * 1.25f;
        float nose = bmax + brad * 1.7f;
        float nz = bmin - brad * 1.5f;   // nozzle mouth

        // engine plume (drawn under the hull) — fly/train only (flameThrottle >= 0)
        if (flameThrottle >= 0.03f)
        {
            float thr = Math.Clamp(flameThrottle, 0f, 1f);
            float flick = 0.82f + 0.18f * MathF.Sin(_frame * 0.9f + b0.X * 2.3f + b0.Y);
            float len = (0.45f + thr * 2.4f) * flick;
            float pw = hw * (0.85f + thr * 0.5f);
            DrawCircle(W(nz - len * 0.12f, 0f), pw * 1.1f, new Color(1f, 0.65f, 0.25f, 0.22f * alpha));
            DrawColoredPolygon(new[] { W(nz, pw), W(nz, -pw), W(nz - len, 0f) },
                               new Color(1f, 0.48f, 0.12f, 0.55f * alpha));
            DrawColoredPolygon(new[] { W(nz, pw * 0.62f), W(nz, -pw * 0.62f), W(nz - len * 0.62f, 0f) },
                               new Color(1f, 0.82f, 0.32f, 0.7f * alpha));
            DrawColoredPolygon(new[] { W(nz, pw * 0.30f), W(nz, -pw * 0.30f), W(nz - len * 0.32f, 0f) },
                               new Color(0.85f, 0.95f, 1f, 0.9f * alpha));
        }

        // tail fins
        var fin = new Color(0.40f, 0.45f, 0.55f, alpha);
        DrawColoredPolygon(new[] { W(bmin + brad * 0.6f, hw), W(bmin - brad * 0.2f, hw * 2.3f), W(bmin - brad * 0.2f, hw) }, fin);
        DrawColoredPolygon(new[] { W(bmin + brad * 0.6f, -hw), W(bmin - brad * 0.2f, -hw * 2.3f), W(bmin - brad * 0.2f, -hw) }, fin);

        // nozzle bell
        DrawColoredPolygon(new[] { W(bmin, hw * 0.55f), W(bmin, -hw * 0.55f), W(nz, -hw), W(nz, hw) },
                           new Color(0.26f, 0.28f, 0.34f, alpha));

        // hull body + nose cone
        DrawColoredPolygon(new[]
        {
            W(bmin, hw * 0.9f), W(bmax - brad * 0.2f, hw), W(bmax + brad * 0.6f, hw * 0.62f),
            W(nose, 0f),
            W(bmax + brad * 0.6f, -hw * 0.62f), W(bmax - brad * 0.2f, -hw), W(bmin, -hw * 0.9f)
        }, new Color(0.82f, 0.86f, 0.94f, alpha));

        // shaded side (gives the tube volume)
        DrawColoredPolygon(new[]
        {
            W(bmin, -hw * 0.2f), W(bmax - brad * 0.2f, -hw * 0.35f),
            W(bmax - brad * 0.2f, -hw), W(bmin, -hw * 0.9f)
        }, new Color(0.52f, 0.57f, 0.68f, alpha * 0.85f));

        // accent stripe + porthole
        float sy = bmax - brad * 1.1f;
        DrawColoredPolygon(new[] { W(sy, hw), W(sy + brad * 0.45f, hw), W(sy + brad * 0.45f, -hw), W(sy, -hw) },
                           new Color(Accent.R, Accent.G, Accent.B, alpha));
        var port = W((bmin + bmax) * 0.5f + brad * 0.8f, 0f);
        DrawCircle(port, brad * 0.5f, new Color(0.10f, 0.16f, 0.26f, alpha));
        DrawArc(port, brad * 0.5f, 0f, MathF.Tau, 14, new Color(Accent.R, Accent.G, Accent.B, alpha * 0.9f), 0.025f);
    }

    static void GeomExtent(WorldState world, RigidBody body, out float min, out float max, out float radius)
    {
        min = float.MaxValue; max = float.MinValue; radius = 0.1f;
        for (int g = 0; g < body.GeomCount; g++)
        {
            var geom = world.RigidBodyGeoms[body.GeomStartIndex + g];
            min = MathF.Min(min, geom.LocalX);
            max = MathF.Max(max, geom.LocalX);
            radius = geom.Radius;
        }
        if (min > max) { min = -0.5f; max = 0.5f; }
    }

    void DrawTarget()
    {
        float pulse = 0.5f + 0.5f * MathF.Sin(_frame * 0.1f);
        var col = new Color(1f, 0.45f, 0.72f);
        var c = ToGodot(_targetX, _targetY);
        DrawArc(c, TargetRingR + 0.18f * pulse, 0f, MathF.Tau, 32, new Color(col.R, col.G, col.B, 0.18f + 0.3f * pulse), 0.05f);
        DrawArc(c, TargetRingR, 0f, MathF.Tau, 32, col, 0.045f);
        DrawCircle(c, 0.07f, new Color(1f, 0.72f, 0.86f, 0.7f + 0.3f * pulse));

        float hx = _targetX + MathF.Cos(_targetAngle) * TargetHandleR;
        float hy = _targetY + MathF.Sin(_targetAngle) * TargetHandleR;
        DrawLine(c, ToGodot(hx, hy), new Color(1f, 0.6f, 0.85f, 0.85f), 0.035f);
        float a = _targetAngle;
        DrawColoredPolygon(new[]
        {
            ToGodot(hx, hy),
            ToGodot(hx - MathF.Cos(a - 0.45f) * 0.32f, hy - MathF.Sin(a - 0.45f) * 0.32f),
            ToGodot(hx - MathF.Cos(a + 0.45f) * 0.32f, hy - MathF.Sin(a + 0.45f) * 0.32f)
        }, new Color(1f, 0.72f, 0.9f));
    }

    // Lunar surface: a faint placement grid (edit only), the regolith fill, a lit rim, and scattered rocks.
    void DrawWorldBackdrop(float groundY, bool editGrid)
    {
        if (editGrid)
        {
            var grid = new Color(0.40f, 0.55f, 0.72f, 0.05f);
            for (int x = -30; x <= 30; x += 2) DrawLine(ToGodot(x, -6f), ToGodot(x, 26f), grid, 0.012f);
            for (int y = -6; y <= 26; y += 2) DrawLine(ToGodot(-30f, y), ToGodot(30f, y), grid, 0.012f);
        }

        const float L = -90f, R = 90f;
        float D = groundY - 70f;
        DrawColoredPolygon(new[] { ToGodot(L, groundY), ToGodot(R, groundY), ToGodot(R, D), ToGodot(L, D) },
                           new Color(0.09f, 0.10f, 0.15f));
        DrawColoredPolygon(new[]
        {
            ToGodot(L, groundY), ToGodot(R, groundY),
            ToGodot(R, groundY - 0.16f), ToGodot(L, groundY - 0.16f)
        }, new Color(0.34f, 0.42f, 0.55f, 0.7f));

        var rng = new Random(7);
        for (int i = 0; i < 30; i++)
        {
            float rx = L + (float)rng.NextDouble() * (R - L);
            float rw = 0.25f + (float)rng.NextDouble() * 1.3f;
            DrawCircle(ToGodot(rx, groundY + rw * 0.12f), rw * 0.55f, new Color(0.14f, 0.16f, 0.22f));
            DrawCircle(ToGodot(rx - rw * 0.12f, groundY + rw * 0.22f), rw * 0.32f, new Color(0.28f, 0.34f, 0.45f, 0.6f));
        }
    }

    (float X, float Y) Com() => ComOf(_world, _rocket);

    static (float X, float Y) ComOf(WorldState world, int[] rocket)
    {
        if (rocket.Length == 0) return (0f, 0f);
        float cx = 0, cy = 0, m = 0;
        foreach (int idx in rocket)
        {
            var b = world.RigidBodies[idx];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            cx += b.X * mass; cy += b.Y * mass; m += mass;
        }
        return m > 0 ? (cx / m, cy / m) : (0f, 0f);
    }

    static (float X, float Y) ComVelOf(WorldState world, int[] rocket)
    {
        float vx = 0, vy = 0, m = 0;
        foreach (int idx in rocket)
        {
            var b = world.RigidBodies[idx];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            vx += b.VelX * mass; vy += b.VelY * mass; m += mass;
        }
        return m > 0 ? (vx / m, vy / m) : (0f, 0f);
    }

    // 10D goal-relative pose observation — mirrors DenseRocketPoseStepKernel exactly.
    void BuildObs(WorldState world, int[] rocket, float tx, float ty, float tAngle,
                  float curThr, float curGim, Span<float> obs)
    {
        var (cx, cy) = ComOf(world, rocket);
        var (vx, vy) = ComVelOf(world, rocket);
        var b0 = world.RigidBodies[rocket[0]];
        obs[0] = (cx - tx) / 20f;
        obs[1] = (cy - ty) / 20f;
        obs[2] = vx / 10f;
        obs[3] = vy / 10f;
        obs[4] = MathF.Cos(b0.Angle);
        obs[5] = MathF.Sin(b0.Angle);
        obs[6] = MathF.Cos(tAngle);
        obs[7] = MathF.Sin(tAngle);
        obs[8] = curGim;
        obs[9] = curThr;
    }

    void ApplyThrustersTo(WorldState world, int[] rocket, float throttle, float gimbal, float dt)
    {
        foreach (var t in _spec.Thrusters)
        {
            int bodyIdx = rocket[t.BodyIndex];
            var body = world.RigidBodies[bodyIdx];
            float c = MathF.Cos(body.Angle), s = MathF.Sin(body.Angle);
            float wdx = t.LocalDirX * c - t.LocalDirY * s;
            float wdy = t.LocalDirX * s + t.LocalDirY * c;
            float thrust = throttle * t.MaxThrust;
            body.VelX += wdx * thrust * body.InvMass * dt;
            body.VelY += wdy * thrust * body.InvMass * dt;
            if (t.Gimbal)
                body.AngularVel += gimbal * t.MaxGimbalTorque * body.InvInertia * dt;
            world.RigidBodies[bodyIdx] = body;
        }
    }

    // ---- input ----------------------------------------------------------------

    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventKey key && key.Pressed && !key.Echo)
        {
            switch (key.Keycode)
            {
                case Key.G: ToggleTraining(); break;
                case Key.T: ToggleFlying(); break;
                case Key.R: ResetAction(); break;
                case Key.S: SaveSpec(); break;
                case Key.L: LoadSpec(); break;
                case Key.Delete: DeleteSelected(); break;
                case Key.Equal or Key.KpAdd: _simSpeed = Math.Min(_simSpeed + 1, 8); break;
                case Key.Minus or Key.KpSubtract: _simSpeed = Math.Max(_simSpeed - 1, 1); break;
                case Key.Escape: GetTree().Quit(); break;
            }
        }
    }

    public override void _UnhandledInput(InputEvent @event)
    {
        if (@event is InputEventMouseButton mb)
        {
            if (mb.ButtonIndex == MouseButton.WheelUp) { _camera.Zoom *= 1.1f; return; }
            if (mb.ButtonIndex == MouseButton.WheelDown) { _camera.Zoom /= 1.1f; return; }
            if (_trainViz) return;
            if (mb.ButtonIndex != MouseButton.Left) return;
            if (!mb.Pressed) { _drag = Drag.None; return; }

            var (mx, my) = MouseWorld();
            if (_flying) BeginTargetDrag(mx, my);
            else BeginBodyDrag(mx, my);
        }
        else if (@event is InputEventMouseMotion && !_trainViz)
        {
            var (mx, my) = MouseWorld();
            switch (_drag)
            {
                case Drag.Body when _selected >= 0:
                    var b = _spec.Bodies[_selected];
                    b.X = mx + _dragOffX;
                    b.Y = (my + _dragOffY) - EditSpawnY;
                    RebuildWorld();
                    OnSpecChanged();
                    break;
                case Drag.TargetPos:
                    _targetX = mx + _dragOffX;
                    _targetY = my + _dragOffY;
                    break;
                case Drag.TargetAngle:
                    _targetAngle = MathF.Atan2(my - _targetY, mx - _targetX);
                    break;
            }
        }
    }

    void BeginBodyDrag(float mx, float my)
    {
        int hit = PickBody(mx, my);
        _selected = hit;
        if (hit >= 0)
        {
            _drag = Drag.Body;
            var b = _spec.Bodies[hit];
            _dragOffX = b.X - mx;
            _dragOffY = (EditSpawnY + b.Y) - my;
            _status = $"Selected body #{hit}.";
        }
        else { _drag = Drag.None; _status = "Nothing under cursor."; }
    }

    void BeginTargetDrag(float mx, float my)
    {
        float hx = _targetX + MathF.Cos(_targetAngle) * TargetHandleR;
        float hy = _targetY + MathF.Sin(_targetAngle) * TargetHandleR;
        if (Dist(mx, my, hx, hy) < 0.45f) { _drag = Drag.TargetAngle; return; }

        if (Dist(mx, my, _targetX, _targetY) < TargetRingR + 0.3f)
        {
            _dragOffX = _targetX - mx;
            _dragOffY = _targetY - my;
        }
        else { _targetX = mx; _targetY = my; _dragOffX = 0f; _dragOffY = 0f; }
        _drag = Drag.TargetPos;
    }

    (float X, float Y) MouseWorld()
    {
        var g = GetGlobalMousePosition();
        return (g.X, -g.Y);
    }

    int PickBody(float mx, float my)
    {
        int best = -1;
        float bestD = float.MaxValue;
        for (int b = 0; b < _rocket.Length; b++)
        {
            var body = _world.RigidBodies[_rocket[b]];
            float c = MathF.Cos(body.Angle), s = MathF.Sin(body.Angle);
            for (int g = 0; g < body.GeomCount; g++)
            {
                var geom = _world.RigidBodyGeoms[body.GeomStartIndex + g];
                float wx = body.X + geom.LocalX * c - geom.LocalY * s;
                float wy = body.Y + geom.LocalX * s + geom.LocalY * c;
                float d = Dist(wx, wy, mx, my) - geom.Radius;
                if (d < PickTolerance && d < bestD) { bestD = d; best = b; }
            }
        }
        return best;
    }

    // ---- editing actions ------------------------------------------------------

    bool BlockEditIfTraining()
    {
        if (_training) { _status = "Stop training (G) before editing the rocket."; return true; }
        if (_trainViz || _flying) { _trainViz = false; _flying = false; RebuildWorld(); UpdateControlsHint(); }
        return false;
    }

    void AddBody()
    {
        if (BlockEditIfTraining()) return;
        var body = new BodySpec { X = 2f, Y = 0f, Angle = 0f, Mass = 2f, Inertia = 1f };
        body.Geoms.Add(new GeomSpec(0f, 0f, 0.3f));
        _spec.Bodies.Add(body);
        _selected = _spec.BodyCount - 1;
        RebuildWorld();
        OnSpecChanged();
        _status = $"Added body #{_selected} — drag it into place.";
    }

    void AddSensor()
    {
        if (BlockEditIfTraining()) return;
        int bidx = _selected >= 0 && _selected < _spec.BodyCount ? _selected : 0;
        int count = _spec.Sensors.Count;
        _spec.Sensors.Add(new SensorSpec { BodyIndex = bidx, AngleOffset = count * 0.5f, MaxRange = 15f });
        OnSpecChanged();
        _status = $"Added sensor on body #{bidx} (now {_spec.SensorCount}).";
    }

    void DeleteSelected()
    {
        if (BlockEditIfTraining()) return;
        if (_selected < 0 || _selected >= _spec.BodyCount) { _status = "Nothing selected to delete."; return; }
        if (_spec.BodyCount <= 1) { _status = "Cannot delete the last body."; return; }

        int del = _selected;
        _spec.Joints.RemoveAll(j => j.BodyA == del || j.BodyB == del);
        foreach (var j in _spec.Joints) { if (j.BodyA > del) j.BodyA--; if (j.BodyB > del) j.BodyB--; }
        _spec.Thrusters.RemoveAll(t => t.BodyIndex == del);
        foreach (var t in _spec.Thrusters) if (t.BodyIndex > del) t.BodyIndex--;
        _spec.Sensors.RemoveAll(s => s.BodyIndex == del);
        foreach (var s in _spec.Sensors) if (s.BodyIndex > del) s.BodyIndex--;
        _spec.Bodies.RemoveAt(del);

        _selected = -1;
        RebuildWorld();
        OnSpecChanged();
        _status = $"Deleted body #{del}.";
    }

    void OnSpecChanged()
    {
        if (_controller != null && _controller.Signature != Signature(_spec))
        {
            _controller = null;
            _flyDirty = true;
            _status = "Controller invalidated (rocket changed) — retrain.";
        }
    }

    void SaveSpec()
    {
        try
        {
            using var f = global::Godot.FileAccess.Open(SavePath, global::Godot.FileAccess.ModeFlags.Write);
            f.StoreString(_spec.ToJson());
            _status = $"Saved → {ProjectSettings.GlobalizePath(SavePath)}";
            GD.Print(_status);
        }
        catch (Exception ex) { _status = $"Save failed: {ex.Message}"; GD.PrintErr(_status); }
    }

    void LoadSpec()
    {
        if (BlockEditIfTraining()) return;
        try
        {
            if (!global::Godot.FileAccess.FileExists(SavePath)) { _status = $"No saved spec at {SavePath}"; return; }
            using var f = global::Godot.FileAccess.Open(SavePath, global::Godot.FileAccess.ModeFlags.Read);
            _spec = RocketSpec.FromJson(f.GetAsText());
            _spec.Validate();
            _selected = -1;
            RebuildWorld();
            OnSpecChanged();
            _status = $"Loaded ← {ProjectSettings.GlobalizePath(SavePath)}";
            GD.Print(_status);
        }
        catch (Exception ex) { _status = $"Load failed: {ex.Message}"; GD.PrintErr(_status); }
    }

    public override void _ExitTree()
    {
        // Stop training and let the background thread finish using the GPU before we tear it down.
        _trainCts?.Cancel();
        if (_trainThread != null && _trainThread.IsAlive) _trainThread.Join(2000);
        // Dispose the ILGPU context exactly once, at scene exit (process is ending anyway).
        Trace("exittree-dispose-begin");
        _eval?.Dispose();
        _eval = null;
        Trace("exittree-dispose-end");
    }

    void HandleShot()
    {
        if (_shotPath == null) return;
        _shotFrame++;
        if (_shotFrame == 24)
        {
            var img = GetViewport().GetTexture().GetImage();
            Error err = img.SavePng(_shotPath);
            GD.Print($"shot {(err == Error.Ok ? "saved" : "FAILED " + err)}: {_shotPath}");
        }
        else if (_shotFrame >= 27) GetTree().Quit();
    }

    // ---- helpers --------------------------------------------------------------

    static float[] RandomParams(DenseTopology topo, int seed)
    {
        var p = new float[topo.TotalParams];
        var r = new Random(seed);
        for (int i = 0; i < p.Length; i++) p[i] = (float)(r.NextDouble() * 2.0 - 1.0) * 0.6f;
        return p;
    }

    static int Signature(RocketSpec s)
    {
        unchecked
        {
            int h = 17;
            h = h * 31 + s.Bodies.Count;
            foreach (var b in s.Bodies)
            {
                h = h * 31 + b.Geoms.Count;
                h = h * 31 + Q(b.X); h = h * 31 + Q(b.Y); h = h * 31 + Q(b.Angle); h = h * 31 + Q(b.Mass);
                foreach (var g in b.Geoms) { h = h * 31 + Q(g.LocalX); h = h * 31 + Q(g.LocalY); h = h * 31 + Q(g.Radius); }
            }
            h = h * 31 + s.Joints.Count;
            foreach (var j in s.Joints) { h = h * 31 + j.BodyA; h = h * 31 + j.BodyB; h = h * 31 + Q(j.ReferenceAngle); }
            h = h * 31 + s.Thrusters.Count;
            foreach (var t in s.Thrusters) { h = h * 31 + t.BodyIndex; h = h * 31 + Q(t.MaxThrust); h = h * 31 + (t.Gimbal ? 1 : 0); }
            h = h * 31 + s.Sensors.Count;
            foreach (var se in s.Sensors) { h = h * 31 + se.BodyIndex; h = h * 31 + Q(se.AngleOffset); }
            return h;
        }
    }

    static int Q(float f) => (int)MathF.Round(f * 1000f);

    static float Dist(float ax, float ay, float bx, float by)
        => MathF.Sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));

    static Vector2 ToGodot(float wx, float wy) => new(wx, -wy);
}
