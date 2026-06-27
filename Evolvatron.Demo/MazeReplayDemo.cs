using Evolvatron.Core.GPU;
using Evolvatron.Evolvion.ES;
using Evolvatron.Evolvion.GPU;
using Raylib_cs;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Phase-2 maze navigator replay (see docs/phase2_maze_spec.md).
///
/// Loads an evolved maze policy + the FROZEN Phase-1 controller and replays the hierarchy
/// across a grid of worlds, each with its own goal. Per cell:
///   green target ring = goal              GREEN arrow = commanded velocity (maze policy output)
///   rocket + flame    = frozen controller CYAN  arrow = actual COM velocity
/// "REACHED" when the rocket arrives. The maze policy never touches the controller — it only
/// hands it gentle velocity commands, and the controller flies the rocket.
///
/// Usage: dotnet run --project Evolvatron.Demo -- maze [maze_policy.bin] [--controller controller_easy.bin]
///        [--hidden 24,24] [--controller-hidden 16,16] [--sensors 0]
/// Controls: SPACE pause, R reset, +/- speed.
/// </summary>
public static class MazeReplayDemo
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
    private const int MaxTrailPoints = 600;
    private const int DemoMaxSteps = 900;

    public static void Run(string[]? args = null)
    {
        string binPath = args != null && args.Length > 1 && !args[1].StartsWith("--")
            ? args[1] : Path.Combine("scratch", "maze_policy.bin");
        if (!File.Exists(binPath))
        {
            string alt = Path.Combine("scratch", "maze_smoke.bin");
            if (File.Exists(alt)) binPath = alt;
        }
        if (!File.Exists(binPath)) { Console.WriteLine($"Maze policy not found: {binPath}"); return; }

        string controllerPath = ParseStrArg(args, "--controller", Path.Combine("scratch", "controller_easy.bin"));
        if (!File.Exists(controllerPath)) { Console.WriteLine($"Frozen controller not found: {controllerPath}"); return; }

        int[] mazeHidden = ParseHidden(args, "--hidden", new[] { 24, 24 });
        int[] ctrlHidden = ParseHidden(args, "--controller-hidden", new[] { 16, 16 });
        int sensors = ParseIntArg(args, "--sensors", 0);
        int obstacles = ParseIntArg(args, "--obstacles", 0);

        float[] mazeMu = LoadParams(binPath);
        float[] ctrlParams = LoadParams(controllerPath);

        var ctrlTopology = DenseTopology.ForRocketController(ctrlHidden);
        if (ctrlParams.Length != ctrlTopology.TotalParams)
        {
            Console.WriteLine($"Controller params {ctrlParams.Length} != topology {ctrlTopology.TotalParams}. Pass --controller-hidden.");
            return;
        }

        int mazeInput = 6 + (sensors >= 8 ? 8 : (sensors >= 4 ? 4 : 0));
        var mazeLayers = new int[mazeHidden.Length + 2];
        mazeLayers[0] = mazeInput;
        for (int i = 0; i < mazeHidden.Length; i++) mazeLayers[i + 1] = mazeHidden[i];
        mazeLayers[^1] = 2;
        var mazeTopology = new DenseTopology(mazeLayers);
        if (mazeMu.Length != mazeTopology.TotalParams)
        {
            Console.WriteLine($"Maze policy params {mazeMu.Length} != topology {mazeTopology} ({mazeTopology.TotalParams}). Pass --hidden / --sensors.");
            return;
        }

        GPUDenseMazeEvaluator eval;
        try
        {
            eval = new GPUDenseMazeEvaluator(mazeTopology, ctrlTopology, ctrlParams)
            { MaxSteps = DemoMaxSteps, SensorCount = sensors, NumObstacles = obstacles };
        }
        catch (Exception ex) { Console.WriteLine($"GPU unavailable: {ex.Message}"); return; }

        RunVisualization(eval, mazeMu);
        eval.Dispose();
    }

    private static void RunVisualization(GPUDenseMazeEvaluator eval, float[] mu)
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Phase-2 Maze Navigator (frozen controller)");
        Raylib.SetTargetFPS(60);

        var trails = new List<(float x, float y)>[CellCount];
        for (int i = 0; i < CellCount; i++) trails[i] = new List<(float x, float y)>(MaxTrailPoints);

        GPURigidBody[]? bodies = null;
        GPURigidBodyGeom[]? geoms = null;
        byte[] terminal = new byte[CellCount];
        byte[] reached = new byte[CellCount];
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
                    eval.ReadReplayState(out bodies, out geoms, out terminal, out reached, out throttle, out gimbal, out cmdVx, out cmdVy);
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
                var mazes = eval.ReplayMazes;
                float goalRadius = eval.GoalRadius;
                for (int i = 0; i < CellCount; i++)
                {
                    int col = i % GridCols, row = i / GridCols;
                    DrawCell(i, bodies, geoms, terminal[i] != 0, reached[i] != 0,
                        throttle[i], gimbal[i], cmdVx[i], cmdVy[i],
                        mazes[i].gx, mazes[i].gy, mazes[i].sx, mazes[i].sy, goalRadius, mazes[i].obstacles, trails[i],
                        col * CellWidth, row * CellHeight, CellWidth, CellHeight);
                }
            }

            for (int c = 1; c < GridCols; c++)
                Raylib.DrawLine(c * CellWidth, 0, c * CellWidth, ScreenHeight, new Color(40, 40, 60, 255));
            for (int r = 1; r < GridRows; r++)
                Raylib.DrawLine(0, r * CellHeight, ScreenWidth, r * CellHeight, new Color(40, 40, 60, 255));

            int reachedCount = 0;
            for (int i = 0; i < CellCount; i++) if (reached[i] != 0) reachedCount++;
            DrawHUD(episodeSeed, stepInEpisode, paused, simulationSpeed, reachedCount);
            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();

        void PrepareEpisode()
        {
            episodeSeed++;
            stepInEpisode = 0;
            for (int i = 0; i < CellCount; i++) trails[i].Clear();
            Array.Clear(terminal);
            Array.Clear(reached);
            eval.PrepareReplay(mu, CellCount, episodeSeed * 1000);
            eval.ReadReplayState(out bodies, out geoms, out terminal, out reached, out throttle, out gimbal, out cmdVx, out cmdVy);
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
            cx += body.X * mass; cy += body.Y * mass; vx += body.VelX * mass; vy += body.VelY * mass; m += mass;
        }
        if (m > 0) { cx /= m; cy /= m; vx /= m; vy /= m; }
        return (cx, cy, vx, vy);
    }

    private static void DrawCell(int worldIdx,
        GPURigidBody[] bodies, GPURigidBodyGeom[] geoms,
        bool isTerminal, bool reachedGoal, float curThrottle, float curGimbal,
        float cmdVx, float cmdVy, float goalX, float goalY, float startX, float startY,
        float goalRadius, GPUOBBCollider[] obstacles, List<(float x, float y)> trail,
        int cellX, int cellY, int cellW, int cellH)
    {
        var (comX, comY, velX, velY) = Com(bodies, worldIdx);

        // Fixed camera framing the whole maze region (rocket travels start → goal).
        const float viewCenterX = 0f, viewCenterY = 5.5f;
        float scale = Math.Min(cellW / 27f, cellH / 16f);
        float cxS = cellX + cellW / 2f, cyS = cellY + cellH / 2f;
        Vector2 W2S(float wx, float wy) => new(cxS + (wx - viewCenterX) * scale, cyS - (wy - viewCenterY) * scale);

        // Obstacles (walls) — filled boxes with a brighter edge, drawn behind everything else.
        if (obstacles != null)
            foreach (var o in obstacles)
            {
                Vector2 c0 = W2S(o.CX - o.HalfExtentX, o.CY + o.HalfExtentY);   // top-left in screen space
                float w = 2f * o.HalfExtentX * scale, h = 2f * o.HalfExtentY * scale;
                Raylib.DrawRectangleV(c0, new Vector2(w, h), new Color((byte)120, (byte)45, (byte)55, (byte)210));
                Raylib.DrawRectangleLinesEx(new Rectangle(c0.X, c0.Y, w, h), 1.5f, new Color((byte)210, (byte)90, (byte)100, (byte)235));
            }

        // Start marker (faint)
        Raylib.DrawCircleLinesV(W2S(startX, startY), 0.4f * scale, new Color(70, 70, 90, 200));

        // Goal target. The SOLID ring is the actual success boundary (COM must enter it to count as
        // reached); the faint outer ring is just an approach halo. Center dot marks the exact goal.
        var goalColor = reachedGoal ? new Color((byte)90, (byte)230, (byte)120, (byte)255)
                                    : new Color((byte)80, (byte)200, (byte)110, (byte)220);
        var haloColor = new Color((byte)60, (byte)140, (byte)90, (byte)110);
        Vector2 gS = W2S(goalX, goalY);
        Raylib.DrawCircleLinesV(gS, (goalRadius + 0.75f) * scale, haloColor);
        Raylib.DrawCircleLinesV(gS, goalRadius * scale, goalColor);
        Raylib.DrawCircleV(gS, 0.15f * scale, goalColor);

        // Trail
        if (trail.Count > 1)
            for (int i = 1; i < trail.Count; i++)
            {
                byte alpha = (byte)(40 + 120 * i / trail.Count);
                Raylib.DrawLineEx(W2S(trail[i - 1].x, trail[i - 1].y), W2S(trail[i].x, trail[i].y),
                    1.5f, new Color((byte)70, (byte)110, (byte)200, alpha));
            }

        // Rocket bodies + geoms
        int bodyBase = worldIdx * BodiesPerWorld;
        int geomBase = worldIdx * GeomsPerWorld;
        byte alphaB = isTerminal && !reachedGoal ? (byte)110 : (byte)225;
        for (int b = 0; b < BodiesPerWorld; b++)
        {
            var body = bodies[bodyBase + b];
            float cos = MathF.Cos(body.Angle), sin = MathF.Sin(body.Angle);
            var color = b == 0 ? new Color((byte)225, (byte)225, (byte)245, alphaB)
                               : new Color((byte)170, (byte)170, (byte)185, alphaB);
            for (int g = 0; g < body.GeomCount; g++)
            {
                var geom = geoms[geomBase + body.GeomStartIndex + g];
                float wx = body.X + geom.LocalX * cos - geom.LocalY * sin;
                float wy = body.Y + geom.LocalX * sin + geom.LocalY * cos;
                Raylib.DrawCircleV(W2S(wx, wy), geom.Radius * scale, color);
            }
        }

        // Thrust flame
        if (!isTerminal && curThrottle > 0.01f)
        {
            var body0 = bodies[bodyBase];
            float bc = MathF.Cos(body0.Angle), bs = MathF.Sin(body0.Angle);
            float botX = body0.X - 0.75f * bc, botY = body0.Y - 0.75f * bs;
            float fa = body0.Angle + curGimbal * 0.5f;
            float fc = MathF.Cos(fa), fs = MathF.Sin(fa);
            float len = curThrottle * 2.0f;
            Raylib.DrawLineEx(W2S(botX, botY), W2S(botX - len * fc, botY - len * fs), 3f,
                new Color((byte)255, (byte)170, (byte)40, (byte)220));
        }

        // Velocity arrows from COM: GREEN = commanded, CYAN = actual
        Vector2 comS = W2S(comX, comY);
        const float arrowScale = 0.5f; // metres of arrow per (m/s)
        if (!isTerminal)
        {
            DrawArrow(comS, new Vector2(comS.X + cmdVx * arrowScale * scale, comS.Y - cmdVy * arrowScale * scale),
                new Color((byte)90, (byte)230, (byte)110, (byte)235));
            DrawArrow(comS, new Vector2(comS.X + velX * arrowScale * scale, comS.Y - velY * arrowScale * scale),
                new Color((byte)80, (byte)200, (byte)230, (byte)235));
        }

        if (reachedGoal)
            Raylib.DrawText("REACHED", cellX + 6, cellY + 6, 16, new Color((byte)90, (byte)230, (byte)120, (byte)230));
        else if (isTerminal)
            Raylib.DrawText("--", cellX + 6, cellY + 6, 16, new Color((byte)200, (byte)110, (byte)110, (byte)200));
    }

    private static void DrawArrow(Vector2 from, Vector2 to, Color color)
    {
        Raylib.DrawLineEx(from, to, 2.5f, color);
        Vector2 d = to - from;
        float len = d.Length();
        if (len < 3f) return;
        d /= len;
        Vector2 perp = new(-d.Y, d.X);
        float h = 7f;
        Raylib.DrawTriangle(to, to - d * h + perp * h * 0.6f, to - d * h - perp * h * 0.6f, color);
    }

    private static void DrawHUD(int seed, int step, bool paused, int speed, int reachedCount)
    {
        Raylib.DrawRectangle(0, 0, ScreenWidth, 26, new Color(0, 0, 0, 150));
        Raylib.DrawText($"Phase-2 maze navigator (frozen controller)   reached {reachedCount}/{CellCount}   " +
            $"seed {seed}   step {step}   speed {speed}x   {(paused ? "PAUSED" : "")}   " +
            $"[SPACE pause  R reset  +/- speed]   green=command  cyan=actual  green ring=goal",
            10, 5, 16, new Color(220, 220, 235, 255));
    }

    private static int[] ParseHidden(string[]? args, string flag, int[] def)
    {
        if (args != null)
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == flag)
                {
                    var parts = args[i + 1].Split(',');
                    var h = new int[parts.Length];
                    for (int k = 0; k < parts.Length; k++) h[k] = int.Parse(parts[k]);
                    return h;
                }
        return def;
    }

    private static int ParseIntArg(string[]? args, string flag, int def)
    {
        if (args != null)
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == flag) return int.Parse(args[i + 1]);
        return def;
    }

    private static string ParseStrArg(string[]? args, string flag, string def)
    {
        if (args != null)
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == flag) return args[i + 1];
        return def;
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
}
