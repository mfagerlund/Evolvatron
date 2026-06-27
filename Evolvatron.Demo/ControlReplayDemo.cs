using Evolvatron.Core.GPU;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Raylib_cs;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Phase-1 controller replay (see docs/phase1_controller_spec.md).
///
/// Loads a trained maneuvering controller (controller.bin) and replays it across a grid
/// of worlds in free space. Each cell shows the rocket tracking a velocity command:
///   GREEN arrow  = commanded velocity (changes every schedule segment)
///   CYAN  arrow  = actual COM velocity
/// When the two arrows align, the controller is tracking. Watch it tilt-thrust-level to
/// chase sideways commands (under-actuation), and go upright to hover on a (0,0) command.
///
/// Usage: dotnet run --project Evolvatron.Demo -- control [path/to/controller.bin]
/// Controls: SPACE pause, R reset, +/- speed.
/// </summary>
public static class ControlReplayDemo
{
    private const int ScreenWidth = 1920;
    private const int ScreenHeight = 1080;
    private const int GridCols = 4;
    private const int GridRows = 3;
    private const int CellCount = GridCols * GridRows;
    private static readonly int CellWidth = ScreenWidth / GridCols;
    private static readonly int CellHeight = ScreenHeight / GridRows;
    private const int BodiesPerWorld = 3;
    private const int GeomsPerWorld = 19;
    private const int MaxTrailPoints = 400;
    private const int DemoMaxSteps = 1200; // ~10s episodes

    public static void Run(string[]? args = null)
    {
        string binPath = args != null && args.Length > 1
            ? args[1]
            : Path.Combine("scratch", "controller.bin");

        if (!File.Exists(binPath))
        {
            // Try repo-relative fallbacks
            string alt = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "scratch", "controller.bin"));
            if (File.Exists(alt)) binPath = alt;
        }

        if (!File.Exists(binPath))
        {
            Console.WriteLine($"Controller not found: {binPath}");
            Console.WriteLine("Train one first: dotnet run --project Evolvatron.TrainingRunner -- --control");
            return;
        }

        float[] mu = LoadParams(binPath);
        Console.WriteLine($"Loaded controller: {mu.Length} params from {binPath}");

        var topology = DenseTopology.ForRocketController(ParseHidden(args), ParseIntArg(args, "--context", 0));
        if (mu.Length != topology.TotalParams)
        {
            Console.WriteLine($"Param count {mu.Length} != topology {topology} ({topology.TotalParams}). " +
                $"Pass --hidden a,b matching how it was trained (e.g. --hidden 24,24).");
            return;
        }

        GPUDenseRocketControlEvaluator eval;
        try
        {
            eval = new GPUDenseRocketControlEvaluator(topology) { MaxSteps = DemoMaxSteps };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"GPU unavailable: {ex.Message}");
            return;
        }

        RunVisualization(eval, mu);
        eval.Dispose();
    }

    /// <summary>Parse "--flag N" as int; returns def if absent.</summary>
    private static int ParseIntArg(string[]? args, string flag, int def)
    {
        if (args != null)
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == flag) return int.Parse(args[i + 1]);
        return def;
    }

    /// <summary>Parse "--flag X" as float (invariant culture); returns def if absent.</summary>
    private static float ParseFloatArg(string[]? args, string flag, float def)
    {
        if (args != null)
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == flag) return float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture);
        return def;
    }

    /// <summary>Parse "--hidden a,b,c" from args; defaults to {16,16} (matches the runner default).</summary>
    private static int[] ParseHidden(string[]? args)
    {
        if (args != null)
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == "--hidden")
                {
                    var parts = args[i + 1].Split(',');
                    var h = new int[parts.Length];
                    for (int k = 0; k < parts.Length; k++) h[k] = int.Parse(parts[k]);
                    return h;
                }
        return new[] { 16, 16 };
    }

    /// <summary>
    /// Headless replay verification (no window): step a trained controller across worlds
    /// and report mean velocity-tracking error. Used to confirm the replay pipeline before
    /// opening the GUI. Usage: dotnet run --project Evolvatron.Demo -- control-verify [bin] [--hidden a,b]
    /// </summary>
    public static void Verify(string[] args)
    {
        string binPath = args.Length > 1 ? args[1] : Path.Combine("scratch", "controller.bin");
        if (!File.Exists(binPath))
        {
            string alt = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "scratch", "controller.bin"));
            if (File.Exists(alt)) binPath = alt;
        }
        if (!File.Exists(binPath)) { Console.WriteLine($"Controller not found: {binPath}"); return; }

        float[] mu = LoadParams(binPath);
        var topology = DenseTopology.ForRocketController(ParseHidden(args), ParseIntArg(args, "--context", 0));
        if (mu.Length != topology.TotalParams)
        {
            Console.WriteLine($"Param count {mu.Length} != topology {topology.TotalParams}. " +
                $"Pass --hidden a,b matching how it was trained (e.g. --hidden 24,24).");
            return;
        }

        const int count = 16, steps = 600;
        using var eval = new GPUDenseRocketControlEvaluator(topology) { MaxSteps = steps };
        // Match the command distribution the controller was trained on (else we test it
        // on commands it never saw — e.g. an easy-command controller on hard commands).
        eval.SegmentLength = ParseIntArg(args, "--segment-length", eval.SegmentLength);
        eval.CmdSpeedMax = ParseFloatArg(args, "--cmd-speed", eval.CmdSpeedMax);
        Console.WriteLine($"Verify task: segLen={eval.SegmentLength}, CmdSpeedMax={eval.CmdSpeedMax}, steps={steps}");
        eval.PrepareReplay(mu, count, baseSeed: 777000);

        int segLen = eval.SegmentLength;
        var sumErr = new double[count];
        var cnt = new int[count];
        var transSum = new double[count];   // first quarter of each segment (just after a command change)
        var transCnt = new int[count];
        var settleSum = new double[count];  // last quarter of each segment (after the vehicle has re-oriented)
        var settleCnt = new int[count];

        for (int s = 0; s < steps; s++)
        {
            eval.StepReplay();
            eval.ReadReplayState(out var bodies, out _, out var term, out _, out _, out var cvx, out var cvy);
            int posInSeg = s % segLen;
            bool isTransient = posInSeg < segLen / 4;
            bool isSettled = posInSeg >= segLen * 3 / 4;
            for (int i = 0; i < count; i++)
            {
                if (term[i] != 0) continue;  // stop logging once this world terminates (tumble OR MaxSteps)
                var (_, _, vx, vy) = Com(bodies, i);
                double err = Math.Sqrt((vx - cvx[i]) * (vx - cvx[i]) + (vy - cvy[i]) * (vy - cvy[i]));
                sumErr[i] += err; cnt[i]++;
                if (isTransient) { transSum[i] += err; transCnt[i]++; }
                if (isSettled) { settleSum[i] += err; settleCnt[i]++; }
            }
        }

        // Classify by step counter, NOT the terminal flag: the kernel sets IsTerminal=1 both on
        // tumble and on reaching MaxSteps, so a full-episode survivor and an early tumble look
        // identical via the flag alone. Survived iff the world reached the full step budget.
        var finalSteps = eval.ReplaySteps;
        int survived = 0, tumbled = 0;
        double survErr = 0; int survN = 0;
        double transErr = 0; int transN = 0;
        double settleErr = 0; int settleN = 0;
        for (int i = 0; i < count; i++)
        {
            bool ok = finalSteps[i] >= steps;
            if (ok) survived++; else tumbled++;
            if (cnt[i] > 0) { survErr += sumErr[i] / cnt[i]; survN++; }
            if (transCnt[i] > 0) { transErr += transSum[i] / transCnt[i]; transN++; }
            if (settleCnt[i] > 0) { settleErr += settleSum[i] / settleCnt[i]; settleN++; }
        }
        Console.WriteLine($"Replay verify: {count} worlds, {survived} survived / {tumbled} tumbled " +
            $"(full {steps}-step episodes, segLen={segLen}).");
        Console.WriteLine($"  Tracking error  overall={(survN > 0 ? survErr / survN : 0):F2}  " +
            $"transient[first 1/4 of segment]={(transN > 0 ? transErr / transN : 0):F2}  " +
            $"settled[last 1/4]={(settleN > 0 ? settleErr / settleN : 0):F2}  m/s " +
            $"(CmdSpeedMax={eval.CmdSpeedMax}). Settled is the steady-state tracking quality.");
    }

    private static float[] LoadParams(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        int n = br.ReadInt32();
        var mu = new float[n];
        for (int i = 0; i < n; i++) mu[i] = br.ReadSingle();
        return mu;
    }

    private static void RunVisualization(GPUDenseRocketControlEvaluator eval, float[] mu)
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Phase-1 Controller (velocity tracking)");
        Raylib.SetTargetFPS(60);

        var trails = new List<(float x, float y)>[CellCount];
        for (int i = 0; i < CellCount; i++)
            trails[i] = new List<(float x, float y)>(MaxTrailPoints);

        GPURigidBody[]? bodies = null;
        GPURigidBodyGeom[]? geoms = null;
        byte[] terminal = new byte[CellCount];
        float[] throttle = new float[CellCount];
        float[] gimbal = new float[CellCount];
        float[] cmdVx = new float[CellCount];
        float[] cmdVy = new float[CellCount];

        bool paused = false;
        int simulationSpeed = 3;
        int episodeSeed = 1;
        int stepInEpisode = 0;

        PrepareEpisode();

        while (!Raylib.WindowShouldClose())
        {
            if (Raylib.IsKeyPressed(KeyboardKey.Space)) paused = !paused;
            if (Raylib.IsKeyPressed(KeyboardKey.Equal) || Raylib.IsKeyPressed(KeyboardKey.KpAdd))
                simulationSpeed = Math.Min(simulationSpeed + 1, 20);
            if (Raylib.IsKeyPressed(KeyboardKey.Minus) || Raylib.IsKeyPressed(KeyboardKey.KpSubtract))
                simulationSpeed = Math.Max(simulationSpeed - 1, 1);
            if (Raylib.IsKeyPressed(KeyboardKey.R)) PrepareEpisode();

            if (!paused)
            {
                for (int s = 0; s < simulationSpeed; s++)
                {
                    bool allDone = true;
                    for (int i = 0; i < CellCount; i++)
                        if (terminal[i] == 0) { allDone = false; break; }
                    if (allDone || stepInEpisode >= DemoMaxSteps) { PrepareEpisode(); break; }

                    eval.StepReplay();
                    eval.ReadReplayState(out bodies, out geoms, out terminal, out throttle, out gimbal, out cmdVx, out cmdVy);
                    stepInEpisode++;

                    for (int i = 0; i < CellCount; i++)
                    {
                        if (terminal[i] != 0) continue;
                        var (cx, cy, _, _) = Com(bodies!, i);
                        if (trails[i].Count >= MaxTrailPoints) trails[i].RemoveAt(0);
                        trails[i].Add((cx, cy));
                    }
                }
            }

            Raylib.BeginDrawing();
            Raylib.ClearBackground(new Color(10, 10, 28, 255));

            if (bodies != null && geoms != null)
            {
                for (int i = 0; i < CellCount; i++)
                {
                    int col = i % GridCols;
                    int row = i / GridCols;
                    DrawCell(i, bodies, geoms, terminal[i] != 0, throttle[i], gimbal[i],
                        cmdVx[i], cmdVy[i], trails[i],
                        col * CellWidth, row * CellHeight, CellWidth, CellHeight);
                }
            }

            for (int c = 1; c < GridCols; c++)
                Raylib.DrawLine(c * CellWidth, 0, c * CellWidth, ScreenHeight, new Color(40, 40, 60, 255));
            for (int r = 1; r < GridRows; r++)
                Raylib.DrawLine(0, r * CellHeight, ScreenWidth, r * CellHeight, new Color(40, 40, 60, 255));

            DrawHUD(episodeSeed, stepInEpisode, paused, simulationSpeed);
            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();

        void PrepareEpisode()
        {
            episodeSeed++;
            stepInEpisode = 0;
            for (int i = 0; i < CellCount; i++) trails[i].Clear();
            Array.Clear(terminal);
            eval.PrepareReplay(mu, CellCount, episodeSeed * 1000);
            eval.ReadReplayState(out bodies, out geoms, out terminal, out throttle, out gimbal, out cmdVx, out cmdVy);
            for (int i = 0; i < CellCount; i++)
            {
                var (cx, cy, _, _) = Com(bodies!, i);
                trails[i].Add((cx, cy));
            }
        }
    }

    private static (float x, float y, float vx, float vy) Com(GPURigidBody[] bodies, int worldIdx)
    {
        float cx = 0, cy = 0, vx = 0, vy = 0, m = 0;
        int b0 = worldIdx * BodiesPerWorld;
        for (int b = 0; b < BodiesPerWorld; b++)
        {
            var body = bodies[b0 + b];
            if (body.InvMass <= 0) continue;
            float mass = 1f / body.InvMass;
            cx += body.X * mass; cy += body.Y * mass;
            vx += body.VelX * mass; vy += body.VelY * mass;
            m += mass;
        }
        if (m > 0) { cx /= m; cy /= m; vx /= m; vy /= m; }
        return (cx, cy, vx, vy);
    }

    private static void DrawCell(int worldIdx,
        GPURigidBody[] bodies, GPURigidBodyGeom[] geoms,
        bool isTerminal, float curThrottle, float curGimbal,
        float cmdVx, float cmdVy, List<(float x, float y)> trail,
        int cellX, int cellY, int cellW, int cellH)
    {
        var (comX, comY, velX, velY) = Com(bodies, worldIdx);

        // Camera follows COM (free space — rocket drifts), fixed scale.
        const float scale = 26f; // px per metre
        float centerScreenX = cellX + cellW / 2f;
        float centerScreenY = cellY + cellH / 2f + 8f;

        Vector2 W2S(float wx, float wy) => new(
            centerScreenX + (wx - comX) * scale,
            centerScreenY - (wy - comY) * scale);

        // Faint world-up reference at the COM
        Raylib.DrawLineEx(W2S(comX, comY - 1.5f), W2S(comX, comY + 1.5f), 1f, new Color(40, 40, 55, 255));

        // Trail (relative to current COM via W2S — shows the path through space)
        if (trail.Count > 1)
        {
            for (int i = 1; i < trail.Count; i++)
            {
                byte alpha = (byte)(40 + 120 * i / trail.Count);
                Raylib.DrawLineEx(W2S(trail[i - 1].x, trail[i - 1].y), W2S(trail[i].x, trail[i].y),
                    1.5f, new Color((byte)70, (byte)110, (byte)200, alpha));
            }
        }

        // Rocket bodies + geoms
        int bodyBase = worldIdx * BodiesPerWorld;
        int geomBase = worldIdx * GeomsPerWorld;
        byte alphaB = isTerminal ? (byte)110 : (byte)225;
        for (int b = 0; b < BodiesPerWorld; b++)
        {
            var body = bodies[bodyBase + b];
            float cos = MathF.Cos(body.Angle), sin = MathF.Sin(body.Angle);
            bool isMain = b == 0;
            var color = isMain
                ? new Color((byte)225, (byte)225, (byte)245, alphaB)
                : new Color((byte)170, (byte)170, (byte)185, alphaB);
            for (int g = 0; g < body.GeomCount; g++)
            {
                var geom = geoms[geomBase + body.GeomStartIndex + g];
                float wx = body.X + geom.LocalX * cos - geom.LocalY * sin;
                float wy = body.Y + geom.LocalX * sin + geom.LocalY * cos;
                Raylib.DrawCircleV(W2S(wx, wy), geom.Radius * scale, color);
            }
        }

        // Thrust flame (along -body axis from the fuselage base)
        if (!isTerminal && curThrottle > 0.01f)
        {
            var body0 = bodies[bodyBase];
            float bc = MathF.Cos(body0.Angle), bs = MathF.Sin(body0.Angle);
            float botX = body0.X - 0.75f * bc, botY = body0.Y - 0.75f * bs;
            float fa = body0.Angle + curGimbal * 0.5f;
            float fc = MathF.Cos(fa), fs = MathF.Sin(fa);
            float len = curThrottle * 2.0f;
            var tip = W2S(botX - fc * len, botY - fs * len);
            float px = -fs * 0.3f * curThrottle, py = fc * 0.3f * curThrottle;
            var left = W2S(botX + px, botY + py);
            var right = W2S(botX - px, botY - py);
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)170, (byte)30, (byte)(180 * curThrottle)));
        }

        // Command arrow (green) and actual-velocity arrow (cyan) from COM
        DrawArrow(W2S(comX, comY), W2S(comX + cmdVx * 0.35f, comY + cmdVy * 0.35f),
            new Color((byte)60, (byte)220, (byte)90, (byte)230));
        DrawArrow(W2S(comX, comY), W2S(comX + velX * 0.35f, comY + velY * 0.35f),
            new Color((byte)60, (byte)200, (byte)230, (byte)230));

        // Tracking error readout
        float verr = MathF.Sqrt((velX - cmdVx) * (velX - cmdVx) + (velY - cmdVy) * (velY - cmdVy));
        var errColor = verr < 1.5f ? Color.Green : (verr < 4f ? Color.Yellow : new Color((byte)230, (byte)100, (byte)80, (byte)255));
        Raylib.DrawText($"verr {verr:F1}", cellX + 6, cellY + cellH - 22, 16, errColor);
        if (isTerminal)
            Raylib.DrawText("TUMBLED", cellX + 6, cellY + 6, 16, new Color((byte)230, (byte)90, (byte)90, (byte)220));
    }

    private static void DrawArrow(Vector2 from, Vector2 to, Color color)
    {
        Raylib.DrawLineEx(from, to, 2.5f, color);
        Vector2 d = to - from;
        float len = d.Length();
        if (len < 1f) return;
        d /= len;
        Vector2 n = new(-d.Y, d.X);
        float h = 8f;
        Raylib.DrawTriangle(to, to - d * h + n * (h * 0.6f), to - d * h - n * (h * 0.6f), color);
    }

    private static void DrawHUD(int episodeSeed, int step, bool paused, int speed)
    {
        Raylib.DrawRectangle(0, 0, ScreenWidth, 34, new Color(0, 0, 0, 180));
        Raylib.DrawText("PHASE-1 CONTROLLER — velocity tracking", 10, 7, 20, Color.White);
        Raylib.DrawText("GREEN = command", 520, 8, 18, new Color((byte)60, (byte)220, (byte)90, (byte)255));
        Raylib.DrawText("CYAN = actual", 720, 8, 18, new Color((byte)60, (byte)200, (byte)230, (byte)255));
        Raylib.DrawText($"Episode {episodeSeed}  step {step}", 900, 8, 18, Color.LightGray);
        Raylib.DrawText($"Speed {speed}x", 1130, 8, 18, Color.LightGray);
        if (paused) Raylib.DrawText("PAUSED", ScreenWidth / 2 - 40, 7, 22, Color.Red);
        Raylib.DrawText("SPACE:Pause  R:Reset  +/-:Speed", ScreenWidth - 360, 8, 16, Color.Gray);
        Raylib.DrawText($"FPS:{Raylib.GetFPS()}", ScreenWidth - 90, 8, 16, Color.Green);
    }
}
