using Evolvatron.Core;
using Evolvatron.Core.Templates;
using Evolvatron.Evolvion.TrajectoryOptimization;
using Raylib_cs;
using System.Collections.Concurrent;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Lunar lander demo with live optimization visualization.
///
/// The optimizer runs on a background thread. Each time an iteration completes,
/// its trajectory is queued. The render loop plays back the current iteration's
/// trajectory via interpolation. When playback finishes, it jumps to the latest
/// queued iteration (skipping intermediates). The final converged trajectory
/// is always played back in full with real physics.
///
/// Controls:
///   R     - Re-randomize start + re-optimize
///   SPACE - Pause/Resume
///   +/-   - Change speed
///   I     - Toggle past iteration trails
/// </summary>
public static class LunarLanderDemo
{
    private const int ScreenWidth = 1920;
    private const int ScreenHeight = 1080;
    private const float MetersToPixels = 40f;

    private const float GroundY = -5f;
    private const float PadX = 0f;
    private const float PadY = -4.5f;
    private const float PadHalfWidth = 2f;

    private enum Phase { Optimizing, PlayingFinal, Done }

    public static void Run()
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Lunar Lander (LM Trajectory Optimization)");
        Raylib.SetTargetFPS(60);

        var random = new Random(42);
        int seed = 42;

        // Shared state between optimizer thread and render thread
        var iterationQueue = new ConcurrentQueue<IterationSnapshot>();
        int optimizerDoneFlag = 0; // 0 = running, 1 = done
        TrajectoryResult? finalResult = null;

        // Render state
        Phase phase = Phase.Optimizing;
        IterationSnapshot? currentIteration = null;
        float playbackT = 0f;              // 0..N interpolation through states
        var pastTrails = new List<IterationSnapshot>();
        bool showPastTrails = true;
        bool paused = false;
        int speedMultiplier = 2;

        // Final physics playback
        WorldState? finalWorld = null;
        int[]? finalRocket = null;
        CPUStepper? finalStepper = null;
        int finalPlaybackStep = 0;
        int finalPhysicsSubstep = 0;
        var finalTrail = new List<(float x, float y, float throttle)>();
        int controlSteps = 40;
        int physicsStepsPerControl = 15;

        // Will be set from optimizer's config to ensure trajectory matches
        SimulationConfig? playbackConfig = null;

        float startX = 0f, startY = 0f, startVelX = 0f, startVelY = 0f;
        Thread? optimizerThread = null;

        StartOptimization();

        while (!Raylib.WindowShouldClose())
        {
            float dt = Raylib.GetFrameTime();

            // Input
            if (Raylib.IsKeyPressed(KeyboardKey.R))
            {
                seed = random.Next();
                StartOptimization();
            }
            if (Raylib.IsKeyPressed(KeyboardKey.Space))
                paused = !paused;
            if (Raylib.IsKeyPressed(KeyboardKey.I))
                showPastTrails = !showPastTrails;
            if (Raylib.IsKeyPressed(KeyboardKey.Equal) || Raylib.IsKeyPressed(KeyboardKey.KpAdd))
                speedMultiplier = Math.Min(speedMultiplier + 1, 16);
            if (Raylib.IsKeyPressed(KeyboardKey.Minus) || Raylib.IsKeyPressed(KeyboardKey.KpSubtract))
                speedMultiplier = Math.Max(speedMultiplier - 1, 1);

            // Phase: Optimizing — play back iterations as they arrive
            if (phase == Phase.Optimizing && !paused)
            {
                if (currentIteration == null)
                {
                    // Try to grab the latest iteration (skip intermediates)
                    TakeLatest();
                }

                if (currentIteration != null)
                {
                    int stateCount = currentIteration.States.Length;
                    float maxT = stateCount - 1;

                    // Advance playback — each control step takes ~0.125s real-time at 1x
                    // Speed it up so iterations play in ~1 second
                    float playbackSpeed = maxT / 1.0f * speedMultiplier;
                    playbackT += dt * playbackSpeed;

                    if (playbackT >= maxT)
                    {
                        // This iteration's playback is done
                        playbackT = maxT;

                        // Check if optimizer is finished
                        if (Volatile.Read(ref optimizerDoneFlag) == 1)
                        {
                            // Archive this trail, then go to final playback
                            pastTrails.Add(currentIteration);
                            currentIteration = null;
                            TransitionToFinalPlayback();
                        }
                        else
                        {
                            // Try to grab the next iteration
                            var prev = currentIteration;
                            TakeLatest();
                            if (currentIteration != prev)
                            {
                                // New iteration arrived — archive the old one
                                pastTrails.Add(prev);
                                playbackT = 0f;
                            }
                            // else: no new iteration yet, just hold at end
                        }
                    }
                }
                else if (Volatile.Read(ref optimizerDoneFlag) == 1)
                {
                    TransitionToFinalPlayback();
                }
            }

            // Phase: PlayingFinal — full physics playback
            if (phase == Phase.PlayingFinal && !paused &&
                finalWorld != null && finalRocket != null && finalResult != null && playbackConfig != null)
            {
                // Total physics ticks: optimized controls + settling phase
                int totalOptimizedTicks = controlSteps * physicsStepsPerControl;
                int settlingTicks = 3 * 120; // 3 seconds of settling at 120Hz
                int totalTicks = totalOptimizedTicks + settlingTicks;

                for (int tick = 0; tick < speedMultiplier; tick++)
                {
                    int currentTick = finalPlaybackStep * physicsStepsPerControl + finalPhysicsSubstep;
                    if (currentTick >= totalTicks)
                    {
                        phase = Phase.Done;
                        break;
                    }

                    float throttle, gimbal;
                    if (finalPlaybackStep < controlSteps)
                    {
                        // Optimized controls phase
                        throttle = finalResult.Throttles[finalPlaybackStep];
                        gimbal = finalResult.Gimbals[finalPlaybackStep];
                    }
                    else
                    {
                        // Settling phase: zero thrust, let gravity bring it down
                        throttle = 0f;
                        gimbal = 0f;
                    }

                    if (throttle > 0f)
                        RigidBodyRocketTemplate.ApplyThrust(finalWorld, finalRocket,
                            throttle, 200f, playbackConfig.Dt);
                    if (MathF.Abs(gimbal) > 0.001f)
                        RigidBodyRocketTemplate.ApplyGimbal(finalWorld, finalRocket,
                            gimbal * 50f, playbackConfig.Dt);
                    finalStepper!.Step(finalWorld, playbackConfig);

                    finalPhysicsSubstep++;
                    RigidBodyRocketTemplate.GetCenterOfMass(finalWorld, finalRocket,
                        out float tx, out float ty);
                    finalTrail.Add((tx, ty, throttle));

                    if (finalPhysicsSubstep >= physicsStepsPerControl)
                    {
                        finalPhysicsSubstep = 0;
                        finalPlaybackStep++;
                    }
                }
            }

            // === DRAW ===
            Raylib.BeginDrawing();
            Raylib.ClearBackground(new Color(10, 10, 30, 255));

            var camera = new Vector2(PadX, (GroundY + startY) / 2f);

            DrawGround(camera);
            DrawLandingPad(camera);

            // Past iteration trails
            if (showPastTrails)
            {
                int trailCount = pastTrails.Count;
                for (int i = 0; i < trailCount; i++)
                {
                    float progress = trailCount > 1 ? (float)i / (trailCount - 1) : 1f;
                    DrawIterationTrail(camera, pastTrails[i].States, progress, false);
                }
            }

            // Current iteration being played back (during optimization phase)
            if (phase == Phase.Optimizing && currentIteration != null)
            {
                int totalPast = pastTrails.Count;
                float progress = totalPast > 0 ? 1f : 0.5f;
                DrawIterationTrail(camera, currentIteration.States, progress, true);
                DrawInterpolatedRocket(camera, currentIteration.States, playbackT);
            }

            // Final physics playback
            if (phase >= Phase.PlayingFinal)
            {
                DrawPhysicsTrail(camera, finalTrail);

                if (finalWorld != null && finalRocket != null)
                {
                    float currentThrottle = (finalResult != null && finalPlaybackStep < controlSteps)
                        ? finalResult.Throttles[finalPlaybackStep] : 0f;
                    DrawRocket(finalWorld, finalRocket, camera, currentThrottle);
                }
            }

            DrawHUD(phase, currentIteration, pastTrails.Count, finalResult,
                finalPlaybackStep, controlSteps, finalWorld, finalRocket,
                paused, speedMultiplier, showPastTrails, Volatile.Read(ref optimizerDoneFlag) == 1);

            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();

        // --- Local functions ---

        void TakeLatest()
        {
            IterationSnapshot? latest = null;
            while (iterationQueue.TryDequeue(out var snap))
                latest = snap;
            if (latest != null)
            {
                if (currentIteration != null && currentIteration != latest)
                    pastTrails.Add(currentIteration);
                currentIteration = latest;
                playbackT = 0f;
            }
        }

        void TransitionToFinalPlayback()
        {
            if (finalResult == null) return;

            // Drain any remaining from queue
            while (iterationQueue.TryDequeue(out var snap))
                pastTrails.Add(snap);

            phase = Phase.PlayingFinal;
            finalWorld = new WorldState(64);
            finalWorld.Obbs.Add(OBBCollider.AxisAligned(0f, GroundY, 30f, 0.5f));
            finalRocket = RigidBodyRocketTemplate.CreateRocket(
                finalWorld, startX, startY,
                bodyHeight: 1.5f, bodyRadius: 0.2f,
                legLength: 1.0f, legRadius: 0.1f,
                bodyMass: 8f, legMass: 1.5f);

            for (int i = 0; i < finalRocket.Length; i++)
            {
                var rb = finalWorld.RigidBodies[finalRocket[i]];
                rb.VelX = startVelX;
                rb.VelY = startVelY;
                finalWorld.RigidBodies[finalRocket[i]] = rb;
            }

            finalStepper = new CPUStepper();
            finalPlaybackStep = 0;
            finalPhysicsSubstep = 0;
            finalTrail.Clear();
        }

        void StartOptimization()
        {
            // Reset all state
            while (iterationQueue.TryDequeue(out _)) { }
            Volatile.Write(ref optimizerDoneFlag, 0);
            finalResult = null;
            currentIteration = null;
            playbackT = 0f;
            pastTrails.Clear();
            phase = Phase.Optimizing;
            finalWorld = null;
            finalRocket = null;
            finalTrail.Clear();
            finalPlaybackStep = 0;

            var rng = new Random(seed);
            startX = (float)(rng.NextDouble() * 6f - 3f);
            startY = 10f + (float)(rng.NextDouble() * 8f);
            startVelX = (float)(rng.NextDouble() * 4f - 2f);
            startVelY = (float)(rng.NextDouble() * -3f);

            Console.WriteLine($"\n--- Optimizing trajectory (seed={seed}) ---");
            Console.WriteLine($"  Start: ({startX:F1}, {startY:F1}), vel=({startVelX:F1}, {startVelY:F1})");

            float capturedStartX = startX, capturedStartY = startY;
            float capturedVelX = startVelX, capturedVelY = startVelY;

            optimizerThread = new Thread(() =>
            {
                var optimizer = new TrajectoryOptimizer(new TrajectoryOptimizerOptions
                {
                    MaxIterations = 80,
                    ControlSteps = controlSteps,
                    PhysicsStepsPerControl = physicsStepsPerControl,
                    LogCallback = msg => Console.WriteLine($"  [LM] {msg}"),
                    OnIterationComplete = snap => iterationQueue.Enqueue(snap)
                });

                // Use the optimizer's physics config for playback (must match exactly)
                playbackConfig = optimizer.Config;

                var result = optimizer.Optimize(capturedStartX, capturedStartY,
                    capturedVelX, capturedVelY);

                Console.WriteLine($"  Result: cost={result.FinalCost:F2}, " +
                                  $"iters={result.Iterations}, " +
                                  $"time={result.ComputationTimeMs:F0}ms, " +
                                  $"reason={result.ConvergenceReason}");

                var final = result.States[^1];
                Console.WriteLine($"  Final: pos=({final.X:F2}, {final.Y:F2}), " +
                                  $"vel=({final.VelX:F2}, {final.VelY:F2}), " +
                                  $"tilt={((final.Angle - MathF.PI / 2f) * 180f / MathF.PI):F1} deg");

                finalResult = result;
                Volatile.Write(ref optimizerDoneFlag, 1);
            });
            optimizerThread.IsBackground = true;
            optimizerThread.Start();
        }
    }

    // --- Drawing helpers ---

    private static Vector2 WorldToScreen(float wx, float wy, Vector2 camera)
    {
        float sx = ScreenWidth / 2f + (wx - camera.X) * MetersToPixels;
        float sy = ScreenHeight / 2f - (wy - camera.Y) * MetersToPixels;
        return new Vector2(sx, sy);
    }

    // The ground OBB has half-height 0.5, so the collision surface is 0.5 above GroundY
    private const float GroundSurfaceY = GroundY + 0.5f;

    private static void DrawGround(Vector2 camera)
    {
        // Draw ground line at the actual collision surface, not the OBB center
        var left = WorldToScreen(-30f, GroundSurfaceY, camera);
        var right = WorldToScreen(30f, GroundSurfaceY, camera);
        Raylib.DrawLineEx(left, right, 3f, new Color(100, 100, 100, 255));

        for (float x = -28f; x <= 28f; x += 2f)
        {
            var top = WorldToScreen(x, GroundSurfaceY, camera);
            var bot = WorldToScreen(x, GroundSurfaceY - 0.5f, camera);
            Raylib.DrawLineV(top, bot, new Color(80, 80, 80, 255));
        }
    }

    private static void DrawLandingPad(Vector2 camera)
    {
        var topLeft = WorldToScreen(PadX - PadHalfWidth, PadY + 0.3f, camera);
        var botRight = WorldToScreen(PadX + PadHalfWidth, PadY - 0.1f, camera);
        float w = botRight.X - topLeft.X;
        float h = botRight.Y - topLeft.Y;
        Raylib.DrawRectangleV(topLeft, new Vector2(w, h), new Color(200, 60, 20, 200));
        Raylib.DrawRectangleLinesEx(new Rectangle(topLeft.X, topLeft.Y, w, h), 2f, new Color(255, 100, 50, 255));
    }

    private static void DrawIterationTrail(Vector2 camera, TrajectoryState[] states, float progress, bool isActive)
    {
        // Color: red (early) -> yellow (mid) -> green (final)
        byte r, g, b;
        if (progress < 0.5f)
        {
            float t = progress * 2f;
            r = (byte)(200 - (int)(100 * t));
            g = (byte)(40 + (int)(180 * t));
            b = 40;
        }
        else
        {
            float t = (progress - 0.5f) * 2f;
            r = (byte)(100 - (int)(80 * t));
            g = (byte)(220 + (int)(35 * t));
            b = (byte)(40 + (int)(60 * t));
        }

        byte alpha = isActive ? (byte)220 : (byte)(40 + (int)(60 * progress));
        float thickness = isActive ? 2.5f : 1.2f;

        for (int i = 1; i < states.Length; i++)
        {
            var p0 = WorldToScreen(states[i - 1].X, states[i - 1].Y, camera);
            var p1 = WorldToScreen(states[i].X, states[i].Y, camera);
            Raylib.DrawLineEx(p0, p1, thickness, new Color(r, g, b, alpha));
        }
    }

    // Rocket geometry constants (must match TrajectoryOptimizer / LunarLanderDemo setup)
    private const float RocketBodyHeight = 1.5f;
    private const float RocketBodyRadius = 0.2f;
    private const float RocketLegLength = 1.0f;
    private const float RocketLegRadius = 0.1f;
    private const float RocketBodyMass = 8f;
    private const float RocketLegMass = 1.5f;

    // COM is offset from body center by this amount along body axis (legs pull COM down)
    // Computed: (2 * legMass * (halfBody + halfLeg * cos(45°))) / totalMass
    private static readonly float ComToBodyOffset =
        (2f * RocketLegMass * (RocketBodyHeight * 0.5f + RocketLegLength * 0.5f * MathF.Cos(MathF.PI / 4f)))
        / (RocketBodyMass + 2f * RocketLegMass);

    private static void DrawInterpolatedRocket(Vector2 camera, TrajectoryState[] states, float t)
    {
        if (states.Length < 2) return;

        int maxIdx = states.Length - 1;
        t = Math.Clamp(t, 0f, maxIdx);
        int idx = Math.Min((int)t, maxIdx - 1);
        float frac = t - idx;

        float comX = states[idx].X + (states[idx + 1].X - states[idx].X) * frac;
        float comY = states[idx].Y + (states[idx + 1].Y - states[idx].Y) * frac;
        float angle = states[idx].Angle + (states[idx + 1].Angle - states[idx].Angle) * frac;
        float throttle = states[idx + 1].Throttle;

        // Convert COM to body center
        float bodyX = comX + ComToBodyOffset * MathF.Cos(angle);
        float bodyY = comY + ComToBodyOffset * MathF.Sin(angle);

        DrawRocketFromPose(camera, bodyX, bodyY, angle, throttle, 220);
    }

    private static void DrawPhysicsTrail(Vector2 camera, List<(float x, float y, float throttle)> trail)
    {
        for (int i = 1; i < trail.Count; i++)
        {
            var p0 = WorldToScreen(trail[i - 1].x, trail[i - 1].y, camera);
            var p1 = WorldToScreen(trail[i].x, trail[i].y, camera);

            float t = trail[i].throttle;
            byte r = (byte)(50 + (int)(205 * t));
            byte g = (byte)(100 + (int)(100 * t));
            byte b = (byte)(200 - (int)(150 * t));
            Raylib.DrawLineEx(p0, p1, 3f, new Color(r, g, b, (byte)220));
        }
    }

    private static void DrawRocket(WorldState world, int[] rocketIndices, Vector2 camera, float throttle)
    {
        // Draw actual circle geoms at their true world positions from the physics state.
        // This is more accurate than the kinematic DrawRocketFromPose because it reflects
        // actual joint flexion and contact response.
        foreach (int idx in rocketIndices)
        {
            var rb = world.RigidBodies[idx];
            float cos = MathF.Cos(rb.Angle);
            float sin = MathF.Sin(rb.Angle);
            bool isBody = (idx == rocketIndices[0]);
            var color = isBody
                ? new Color((byte)220, (byte)220, (byte)240, (byte)220)
                : new Color((byte)180, (byte)180, (byte)180, (byte)220);

            for (int g = 0; g < rb.GeomCount; g++)
            {
                var geom = world.RigidBodyGeoms[rb.GeomStartIndex + g];
                float wx = rb.X + geom.LocalX * cos - geom.LocalY * sin;
                float wy = rb.Y + geom.LocalX * sin + geom.LocalY * cos;
                var sp = WorldToScreen(wx, wy, camera);
                float sr = geom.Radius * MetersToPixels;
                Raylib.DrawCircleV(sp, sr, color);
                if (isBody)
                    Raylib.DrawCircleLinesV(sp, sr, new Color((byte)255, (byte)255, (byte)255, (byte)110));
            }
        }

        // Flame from body bottom
        if (throttle > 0.01f)
        {
            var body = world.RigidBodies[rocketIndices[0]];
            float bodyCos = MathF.Cos(body.Angle);
            float bodySin = MathF.Sin(body.Angle);
            float halfBody = RocketBodyHeight * 0.5f;
            float botX = body.X - halfBody * bodyCos;
            float botY = body.Y - halfBody * bodySin;

            float flameLen = throttle * 2.0f;
            float flameTipX = botX - bodyCos * flameLen;
            float flameTipY = botY - bodySin * flameLen;
            float perpX = -bodySin * 0.3f * throttle;
            float perpY = bodyCos * 0.3f * throttle;

            var tip = WorldToScreen(flameTipX, flameTipY, camera);
            var left = WorldToScreen(botX + perpX, botY + perpY, camera);
            var right = WorldToScreen(botX - perpX, botY - perpY, camera);

            byte fa1 = (byte)(180 * throttle);
            byte fa2 = (byte)(100 * throttle);
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)180, (byte)30, fa1));
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)100, (byte)20, fa2));
        }
    }

    /// <summary>
    /// Draws the rocket at the given body center position and angle using the known geom layout.
    /// Used by both interpolated (optimization) and physics (final) rendering.
    /// </summary>
    private static void DrawRocketFromPose(Vector2 camera, float bodyX, float bodyY, float bodyAngle, float throttle, byte alpha)
    {
        float halfBody = RocketBodyHeight * 0.5f;
        float halfLeg = RocketLegLength * 0.5f;

        // Draw body capsule: circles along local X axis (which points "up" at bodyAngle)
        int bodyCircles = 5;
        float bodyCos = MathF.Cos(bodyAngle);
        float bodySin = MathF.Sin(bodyAngle);
        var bodyColor = new Color((byte)220, (byte)220, (byte)240, alpha);

        for (int i = 0; i < bodyCircles; i++)
        {
            float t = (float)i / (bodyCircles - 1);
            float localX = -halfBody + t * RocketBodyHeight;
            float wx = bodyX + localX * bodyCos;
            float wy = bodyY + localX * bodySin;
            var sp = WorldToScreen(wx, wy, camera);
            float sr = RocketBodyRadius * MetersToPixels;
            Raylib.DrawCircleV(sp, sr, bodyColor);
            Raylib.DrawCircleLinesV(sp, sr, new Color((byte)255, (byte)255, (byte)255, (byte)(alpha / 2)));
        }

        // Body bottom in world coords (attachment point for legs)
        float botX = bodyX - halfBody * bodyCos;
        float botY = bodyY - halfBody * bodySin;

        // Left leg: angle = bodyAngle + 3π/4 (135° relative)
        float leftAngle = bodyAngle + 3f * MathF.PI / 4f;
        float leftCos = MathF.Cos(leftAngle);
        float leftSin = MathF.Sin(leftAngle);
        float leftCenterX = botX + halfLeg * leftCos;
        float leftCenterY = botY + halfLeg * leftSin;

        // Right leg: angle = bodyAngle + 5π/4 (225° relative)
        float rightAngle = bodyAngle + 5f * MathF.PI / 4f;
        float rightCos = MathF.Cos(rightAngle);
        float rightSin = MathF.Sin(rightAngle);
        float rightCenterX = botX + halfLeg * rightCos;
        float rightCenterY = botY + halfLeg * rightSin;

        var legColor = new Color((byte)180, (byte)180, (byte)180, alpha);
        int legCircles = 7;

        // Draw left leg circles
        for (int i = 0; i < legCircles; i++)
        {
            float lt = (float)i / (legCircles - 1);
            float localX = -halfLeg + lt * RocketLegLength;
            float wx = leftCenterX + localX * leftCos;
            float wy = leftCenterY + localX * leftSin;
            var sp = WorldToScreen(wx, wy, camera);
            float sr = RocketLegRadius * MetersToPixels;
            Raylib.DrawCircleV(sp, sr, legColor);
        }

        // Draw right leg circles
        for (int i = 0; i < legCircles; i++)
        {
            float lt = (float)i / (legCircles - 1);
            float localX = -halfLeg + lt * RocketLegLength;
            float wx = rightCenterX + localX * rightCos;
            float wy = rightCenterY + localX * rightSin;
            var sp = WorldToScreen(wx, wy, camera);
            float sr = RocketLegRadius * MetersToPixels;
            Raylib.DrawCircleV(sp, sr, legColor);
        }

        // Flame
        if (throttle > 0.01f)
        {
            float engineX = botX;
            float engineY = botY;
            float flameLen = throttle * 2.0f;
            float flameTipX = engineX - bodyCos * flameLen;
            float flameTipY = engineY - bodySin * flameLen;
            float perpX = -bodySin * 0.3f * throttle;
            float perpY = bodyCos * 0.3f * throttle;

            var tip = WorldToScreen(flameTipX, flameTipY, camera);
            var left = WorldToScreen(engineX + perpX, engineY + perpY, camera);
            var right = WorldToScreen(engineX - perpX, engineY - perpY, camera);

            byte fa1 = (byte)(180 * throttle);
            byte fa2 = (byte)(100 * throttle);
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)180, (byte)30, fa1));
            Raylib.DrawTriangle(tip, right, left, new Color((byte)255, (byte)100, (byte)20, fa2));
        }
    }

    private static void DrawHUD(
        Phase phase, IterationSnapshot? currentIter, int pastCount,
        TrajectoryResult? result,
        int step, int totalSteps,
        WorldState? world, int[]? rocket,
        bool paused, int speed, bool showTrails, bool optDone)
    {
        int y = 10;
        Raylib.DrawText("LUNAR LANDER - LM Trajectory Optimization", 10, y, 24, Color.White);
        y += 30;

        if (phase == Phase.Optimizing)
        {
            string status = optDone ? "CONVERGED" : "OPTIMIZING...";
            var statusColor = optDone ? new Color(80, 255, 80, 255) : new Color(255, 200, 50, 255);
            Raylib.DrawText(status, 10, y, 22, statusColor);
            y += 26;

            if (currentIter != null)
            {
                Raylib.DrawText($"Iteration {pastCount + 1}  |  Cost: {currentIter.Cost:F1}",
                                10, y, 18, new Color(255, 200, 50, 200));
                y += 22;
            }
        }

        if (result != null && phase >= Phase.PlayingFinal)
        {
            Raylib.DrawText($"Optimization: {result.Iterations} iters, " +
                            $"{result.ComputationTimeMs:F0}ms, cost={result.FinalCost:F2}",
                            10, y, 18, Color.LightGray);
            y += 22;
            Raylib.DrawText($"Convergence: {result.ConvergenceReason}", 10, y, 18, Color.LightGray);
            y += 22;

            float progress = totalSteps > 0 ? (float)step / totalSteps : 0f;
            int barW = 300;
            Raylib.DrawRectangle(10, y, barW, 14, new Color(40, 40, 40, 255));
            Raylib.DrawRectangle(10, y, (int)(barW * progress), 14, new Color(80, 200, 80, 255));
            Raylib.DrawText($"Playback {step}/{totalSteps}", barW + 20, y - 2, 18, Color.LightGray);
            y += 22;

            if (phase == Phase.Done)
            {
                Raylib.DrawText("LANDED", 10, y, 24, new Color(80, 255, 80, 255));
                y += 28;
            }
        }

        if (world != null && rocket != null && phase >= Phase.PlayingFinal)
        {
            RigidBodyRocketTemplate.GetCenterOfMass(world, rocket, out float cx, out float cy);
            RigidBodyRocketTemplate.GetVelocity(world, rocket, out float vx, out float vy);
            var body = world.RigidBodies[rocket[0]];
            float tilt = (body.Angle - MathF.PI / 2f) * 180f / MathF.PI;

            Raylib.DrawText($"Pos: ({cx:F1}, {cy:F1})  Vel: ({vx:F1}, {vy:F1})  Tilt: {tilt:F1} deg",
                            10, y, 18, Color.LightGray);
            y += 22;
        }

        y += 10;
        Raylib.DrawText($"Speed: {speed}x  {(paused ? "[PAUSED]" : "")}  " +
                        $"Trails: {(showTrails ? "ON" : "OFF")}  ({pastCount} iterations)",
                        10, y, 18, paused ? Color.Yellow : Color.LightGray);

        Raylib.DrawText("R: re-optimize  SPACE: pause  +/-: speed  I: toggle trails",
                        10, ScreenHeight - 30, 18, new Color(120, 120, 120, 255));
    }
}
