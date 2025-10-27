using Raylib_cs;
using System.Numerics;
using RaylibVector2 = System.Numerics.Vector2;
using GodotVector2 = Godot.Vector2;
using RaylibColor = Raylib_cs.Color;
using Evolvatron.Evolvion.Environments;
using Colonel.Tests.HagridTests.FollowTheCorridor;
using static Colonel.Tests.HagridTests.FollowTheCorridor.SimpleCarWorld;

namespace Evolvatron.Demo;

public static class FollowTheCorridorDemo
{
    private const int ScreenWidth = 1600;
    private const int ScreenHeight = 900;
    private const float Scale = 2.5f;

    public static void Run()
    {
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Follow The Corridor");
        Raylib.SetTargetFPS(60);

        var baseEnvironment = new FollowTheCorridorEnvironment(maxSteps: 320);
        RaylibVector2 cameraOffset = new RaylibVector2(200, ScreenHeight - 100);

        object snapshotLock = new object();
        CorridorEvaluationRunner.GenerationSnapshot? latestSnapshot = null;
        bool evolutionRunning = true;
        bool solved = false;

        var runner = new CorridorEvaluationRunner(
            config: new CorridorEvaluationRunner.RunConfig(),
            progressCallback: update =>
            {
                if (update.Generation % 100 == 0)
                {
                    Console.WriteLine($"Gen {update.Generation}: Best={update.BestFitness:F3} ({update.BestFitness * 100:F1}%)");
                }
            },
            snapshotProvider: null
        );

        var evolutionTask = Task.Run(() =>
        {
            var result = runner.Run();
            lock (snapshotLock)
            {
                solved = result.solved;
                evolutionRunning = false;
            }

            Console.WriteLine($"\n=== SUMMARY ===");
            Console.WriteLine($"Generations: {result.generation}");
            Console.WriteLine($"Final fitness: {result.bestFitness:F3} ({result.bestFitness * 100:F1}%)");
            Console.WriteLine($"Status: {(result.solved ? "SOLVED!" : "Stopped")}");
            Console.WriteLine($"Total time: {result.elapsedMs / 1000.0:F1}s");
        });

        while (!Raylib.WindowShouldClose() && !solved)
        {
            lock (snapshotLock)
            {
                latestSnapshot = runner.CreateSnapshot();
            }

            Raylib.BeginDrawing();
            Raylib.ClearBackground(RaylibColor.Black);

            RenderTrack(baseEnvironment.World, cameraOffset);

            if (latestSnapshot != null)
            {
                lock (snapshotLock)
                {
                    foreach (var env in latestSnapshot.Environments)
                    {
                        RaylibColor color;
                        if (!env.IsTerminal)
                        {
                            color = new RaylibColor(100, 200, 255, 180); // Blue - alive
                        }
                        else
                        {
                            color = env.DeathCause switch
                            {
                                Evolvion.Environments.DeathCause.WallCollision => new RaylibColor(255, 100, 100, 120), // Red
                                Evolvion.Environments.DeathCause.TooSlowTo4thMarker => new RaylibColor(255, 165, 0, 120), // Orange
                                Evolvion.Environments.DeathCause.TooSlowAfter4thMarker => new RaylibColor(255, 255, 0, 120), // Yellow
                                Evolvion.Environments.DeathCause.Timeout => new RaylibColor(128, 128, 128, 120), // Gray
                                Evolvion.Environments.DeathCause.Finished => new RaylibColor(0, 255, 0, 180), // Lime
                                _ => new RaylibColor(255, 100, 100, 120)
                            };
                        }
                        RenderCar(env.Position, env.Heading, cameraOffset, color);
                    }

                    RenderUI(
                        latestSnapshot.Generation,
                        latestSnapshot.BestFitness,
                        latestSnapshot.CurrentStep,
                        latestSnapshot.ActiveCount,
                        evolutionRunning,
                        TimeSpan.FromMilliseconds(latestSnapshot.ElapsedMs),
                        latestSnapshot.DeathCounts
                    );
                }
            }

            Raylib.EndDrawing();
        }

        evolutionTask.Wait();
        Raylib.CloseWindow();
    }

    private static void RenderTrack(SimpleCarWorld world, RaylibVector2 cameraOffset)
    {
        // Draw walls in white
        foreach (var lineSegment in world.WallGrid.LineSegments)
        {
            RaylibVector2 start = WorldToScreen(lineSegment.Start, cameraOffset);
            RaylibVector2 end = WorldToScreen(lineSegment.End, cameraOffset);
            Raylib.DrawLineV(start, end, RaylibColor.White);
        }

        // Draw start line in blue
        RaylibVector2 startBegin = WorldToScreen(world.Start.Start, cameraOffset);
        RaylibVector2 startEnd = WorldToScreen(world.Start.End, cameraOffset);
        Raylib.DrawLineEx(startBegin, startEnd, 3f, RaylibColor.SkyBlue);

        // Draw finish line in green
        RaylibVector2 finishBegin = WorldToScreen(world.Finish.Start, cameraOffset);
        RaylibVector2 finishEnd = WorldToScreen(world.Finish.End, cameraOffset);
        Raylib.DrawLineEx(finishBegin, finishEnd, 3f, RaylibColor.Lime);

        // Draw progress markers (every 10th)
        for (int i = 0; i < world.ProgressMarkers.Count; i += 10)
        {
            var marker = world.ProgressMarkers[i];
            RaylibVector2 screenPos = WorldToScreen(marker.Position, cameraOffset);
            Raylib.DrawCircleV(screenPos, 2f, new RaylibColor(100, 100, 100, 100));
        }
    }

    private static void RenderCar(GodotVector2 position, float heading, RaylibVector2 cameraOffset, RaylibColor color)
    {
        RaylibVector2 screenPos = WorldToScreen(position, cameraOffset);

        // Draw car body (circle)
        Raylib.DrawCircleV(screenPos, SimpleCar.Radius * Scale, color);

        // Draw heading indicator (nose) in white
        RaylibVector2 nose = new RaylibVector2(
            position.X + MathF.Cos(heading) * SimpleCar.Radius,
            position.Y + MathF.Sin(heading) * SimpleCar.Radius
        );
        RaylibVector2 screenNose = WorldToScreen(nose, cameraOffset);
        Raylib.DrawLineV(screenPos, screenNose, RaylibColor.White);
    }

    private static void RenderUI(int generation, float bestFitness, int step, int activeCars, bool isSimulating, TimeSpan elapsed, Dictionary<Evolvion.Environments.DeathCause, int> deathCounts)
    {
        int y = 10;
        int lineHeight = 25;

        // Format elapsed time as mm:ss
        string timeString = $"{(int)elapsed.TotalMinutes}:{elapsed.Seconds:D2}";

        Raylib.DrawText($"Generation: {generation} | Time: {timeString}", 10, y, 20, RaylibColor.White);
        y += lineHeight;

        Raylib.DrawText($"Best Fitness: {bestFitness:F3} ({bestFitness * 100:F0}% of track)", 10, y, 20, RaylibColor.White);
        y += lineHeight;

        Raylib.DrawText($"Step: {step} / 320", 10, y, 20, RaylibColor.White);
        y += lineHeight;

        Raylib.DrawText($"Active agents: {activeCars} / 1600", 10, y, 18, RaylibColor.White);
        y += lineHeight;

        if (isSimulating)
        {
            Raylib.DrawText("Status: Simulating MAX SPEED...", 10, y, 20, RaylibColor.Yellow);
        }
        else
        {
            Raylib.DrawText("Status: Ready for next generation", 10, y, 20, RaylibColor.Lime);
        }
        y += lineHeight;

        // Death statistics
        y += 5;
        Raylib.DrawText("Agent Status:", 10, y, 18, RaylibColor.Gray);
        y += lineHeight;

        int wallCount = deathCounts.GetValueOrDefault(Evolvion.Environments.DeathCause.WallCollision, 0);
        int slowTo4th = deathCounts.GetValueOrDefault(Evolvion.Environments.DeathCause.TooSlowTo4thMarker, 0);
        int slowAfter4th = deathCounts.GetValueOrDefault(Evolvion.Environments.DeathCause.TooSlowAfter4thMarker, 0);
        int finished = deathCounts.GetValueOrDefault(Evolvion.Environments.DeathCause.Finished, 0);
        int timeout = deathCounts.GetValueOrDefault(Evolvion.Environments.DeathCause.Timeout, 0);

        Raylib.DrawText($"  Alive: {activeCars}", 10, y, 16, new RaylibColor(100, 200, 255, 255));
        y += 20;
        Raylib.DrawText($"  Wall Collision: {wallCount}", 10, y, 16, RaylibColor.Red);
        y += 20;
        Raylib.DrawText($"  Too Slow (start): {slowTo4th}", 10, y, 16, RaylibColor.Orange);
        y += 20;
        Raylib.DrawText($"  Too Slow (4th+): {slowAfter4th}", 10, y, 16, RaylibColor.Yellow);
        y += 20;
        Raylib.DrawText($"  Timeout: {timeout}", 10, y, 16, RaylibColor.Gray);
        y += 20;
        Raylib.DrawText($"  Finished: {finished}", 10, y, 16, RaylibColor.Lime);
    }

    private static RaylibVector2 WorldToScreen(GodotVector2 worldPos, RaylibVector2 cameraOffset)
    {
        return new RaylibVector2(
            cameraOffset.X + worldPos.X * Scale,
            cameraOffset.Y - worldPos.Y * Scale
        );
    }

    private static RaylibVector2 WorldToScreen(RaylibVector2 worldPos, RaylibVector2 cameraOffset)
    {
        return new RaylibVector2(
            cameraOffset.X + worldPos.X * Scale,
            cameraOffset.Y - worldPos.Y * Scale
        );
    }
}
