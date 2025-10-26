using Evolvatron.Core;
using Raylib_cs;
using System;

namespace Evolvatron.Demo;

/// <summary>
/// Demo showing 30 L-shapes with random angles falling and landing.
/// Tests the angle constraint solver under varied conditions.
/// </summary>
public static class AngleConstraintDemo
{
    private const int WindowWidth = 1600;
    private const int WindowHeight = 900;
    private const float PixelsPerMeter = 40f;

    public static void Run()
    {
        // Suppress Raylib initialization spam
        Raylib.SetTraceLogLevel(TraceLogLevel.Warning);

        Raylib.InitWindow(WindowWidth, WindowHeight, "Angle Constraint Demo - 30 Random L-Shapes");
        Raylib.SetTargetFPS(60);

        var world = CreateDemoWorld();
        var config = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 20,  // Use production settings (not 40 like tests)
            Substeps = 1,
            GravityY = -9.81f,
            RodCompliance = 0f,
            AngleCompliance = 0f,  // Rigid angles
            ContactCompliance = 1e-8f,
            FrictionMu = 0.5f,
            VelocityStabilizationBeta = 1.0f,
            GlobalDamping = 0.02f
        };

        var stepper = new CPUStepper();
        var camera = new Camera2D
        {
            Offset = new System.Numerics.Vector2(WindowWidth / 2f, WindowHeight - 100),
            Target = new System.Numerics.Vector2(0f, 0f),
            Rotation = 0f,
            Zoom = 1f
        };

        bool paused = false;
        int stepCount = 0;

        while (!Raylib.WindowShouldClose())
        {
            // Input
            if (Raylib.IsKeyPressed(KeyboardKey.Space))
                paused = !paused;

            if (Raylib.IsKeyPressed(KeyboardKey.R))
            {
                world = CreateDemoWorld();
                stepCount = 0;
            }

            // Physics
            if (!paused)
            {
                stepper.Step(world, config);
                stepCount++;
            }

            // Render
            Raylib.BeginDrawing();
            Raylib.ClearBackground(Color.Black);

            Raylib.BeginMode2D(camera);

            // Draw ground
            DrawGround(world);

            // Draw all L-shapes
            DrawLShapes(world);

            Raylib.EndMode2D();

            // UI
            DrawUI(stepCount, paused, world);

            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();
    }

    private static WorldState CreateDemoWorld()
    {
        var world = new WorldState(initialCapacity: 200);
        var random = new Random(42);

        // Ground platform
        float platformWidth = 30f;
        float platformY = -2f;
        world.Obbs.Add(OBBCollider.AxisAligned(0f, platformY, platformWidth, 0.5f));

        // Spawn 30 L-shapes with random angles
        int shapesPerRow = 10;
        int rows = 3;
        float spacing = 2.5f;
        float startX = -(shapesPerRow - 1) * spacing / 2f;
        float startY = 5f;

        int shapeIndex = 0;
        Console.WriteLine("\n=== Initial Shape Configurations ===");
        Console.WriteLine("Shape# | Row | Col | Target Angle | Rotation | Notes");
        Console.WriteLine("-------|-----|-----|--------------|----------|------");

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < shapesPerRow; col++)
            {
                float x = startX + col * spacing;
                float y = startY + row * 4f;

                float angleDegrees;

                // Make every 3rd shape a 90-degree angle
                if (shapeIndex % 3 == 0)
                {
                    angleDegrees = 90f;  // Perfect right angle
                }
                else
                {
                    // Random angle between 30° and 150°
                    angleDegrees = 30f + (float)random.NextDouble() * 120f;
                }

                float angleRadians = angleDegrees * MathF.PI / 180f;

                // Random rotation of the entire shape
                float rotationDegrees = (float)random.NextDouble() * 360f;
                float rotationRadians = rotationDegrees * MathF.PI / 180f;

                // Flag known problem shapes based on user observation
                // Shape indices: 9 (last of row 0), 14 (5th of row 1), 28 (2nd to last of row 2)
                string notes = "";
                if (shapeIndex == 9 || shapeIndex == 14 || shapeIndex == 28)
                {
                    notes = "*** SPINS ***";
                }

                Console.WriteLine($"  {shapeIndex,3}  | {row,3} | {col,3} | {angleDegrees,8:F1}° | {rotationDegrees,7:F1}° | {notes}");

                CreateLShape(world, x, y, angleRadians, rotationRadians);
                shapeIndex++;
            }
        }

        Console.WriteLine("\n=== Checking Initial Angle Errors ===");
        for (int idx = 0; idx < world.Angles.Count; idx++)
        {
            var angle = world.Angles[idx];
            int i = angle.I, j = angle.J, k = angle.K;

            float ux = world.PosX[i] - world.PosX[j];
            float uy = world.PosY[i] - world.PosY[j];
            float vx = world.PosX[k] - world.PosX[j];
            float vy = world.PosY[k] - world.PosY[j];

            float currentAngle = MathF.Atan2(ux * vy - uy * vx, ux * vx + uy * vy);
            float error = MathF.Abs(WrapAngle(currentAngle - angle.Theta0));
            float errorDeg = error * 180f / MathF.PI;

            if (idx == 9 || idx == 14 || idx == 28)
            {
                Console.WriteLine($"Shape {idx}: Target={angle.Theta0 * 180f / MathF.PI:F1}°, Initial={currentAngle * 180f / MathF.PI:F1}°, Error={errorDeg:F2}° *** SPINS ***");
            }
            else if (errorDeg > 5f)
            {
                Console.WriteLine($"Shape {idx}: Target={angle.Theta0 * 180f / MathF.PI:F1}°, Initial={currentAngle * 180f / MathF.PI:F1}°, Error={errorDeg:F2}° (large initial error)");
            }
        }

        Console.WriteLine();
        return world;
    }

    private static void CreateLShape(WorldState world, float centerX, float centerY, float targetAngle, float rotation)
    {
        float armLength = 0.8f;
        float mass = 1f;
        float radius = 0.05f;

        // Create L-shape: p0 (left), p1 (vertex), p2 (right at targetAngle)
        // Initially create horizontal (p0-p1) and angled (p1-p2) arms
        float p0x = -armLength;
        float p0y = 0f;

        float p1x = 0f;
        float p1y = 0f;

        float p2x = armLength * MathF.Cos(targetAngle);
        float p2y = armLength * MathF.Sin(targetAngle);

        // Apply rotation
        float cos = MathF.Cos(rotation);
        float sin = MathF.Sin(rotation);

        void RotateAndTranslate(ref float x, ref float y)
        {
            float rx = x * cos - y * sin;
            float ry = x * sin + y * cos;
            x = rx + centerX;
            y = ry + centerY;
        }

        RotateAndTranslate(ref p0x, ref p0y);
        RotateAndTranslate(ref p1x, ref p1y);
        RotateAndTranslate(ref p2x, ref p2y);

        // Add particles
        int i0 = world.AddParticle(p0x, p0y, 0f, 0f, mass, radius);
        int i1 = world.AddParticle(p1x, p1y, 0f, 0f, mass, radius);
        int i2 = world.AddParticle(p2x, p2y, 0f, 0f, mass, radius);

        // Add rods
        world.Rods.Add(new Rod(i0, i1, armLength, 0f));
        world.Rods.Add(new Rod(i1, i2, armLength, 0f));

        // Add angle constraint
        world.Angles.Add(new Angle(i0, i1, i2, targetAngle, 0f));
    }

    private static void DrawGround(WorldState world)
    {
        foreach (var obb in world.Obbs)
        {
            float x = obb.CX * PixelsPerMeter;
            float y = -obb.CY * PixelsPerMeter;
            float w = obb.HalfExtentX * 2f * PixelsPerMeter;
            float h = obb.HalfExtentY * 2f * PixelsPerMeter;

            Raylib.DrawRectangle(
                (int)(x - w / 2f),
                (int)(y - h / 2f),
                (int)w,
                (int)h,
                new Color(60, 60, 60, 255));
        }
    }

    private static void DrawLShapes(WorldState world)
    {
        // Draw rods (edges)
        for (int i = 0; i < world.Rods.Count; i++)
        {
            var rod = world.Rods[i];
            float x1 = world.PosX[rod.I] * PixelsPerMeter;
            float y1 = -world.PosY[rod.I] * PixelsPerMeter;
            float x2 = world.PosX[rod.J] * PixelsPerMeter;
            float y2 = -world.PosY[rod.J] * PixelsPerMeter;

            // Color based on length error (for debugging)
            float currentLen = MathF.Sqrt(
                (world.PosX[rod.I] - world.PosX[rod.J]) * (world.PosX[rod.I] - world.PosX[rod.J]) +
                (world.PosY[rod.I] - world.PosY[rod.J]) * (world.PosY[rod.I] - world.PosY[rod.J]));
            float error = MathF.Abs(currentLen - rod.RestLength);

            Color lineColor = error < 0.01f ? Color.White : Color.Red;

            Raylib.DrawLineEx(
                new System.Numerics.Vector2(x1, y1),
                new System.Numerics.Vector2(x2, y2),
                2f,
                lineColor);
        }

        // Draw particles (vertices)
        for (int i = 0; i < world.ParticleCount; i++)
        {
            float x = world.PosX[i] * PixelsPerMeter;
            float y = -world.PosY[i] * PixelsPerMeter;
            float r = world.Radius[i] * PixelsPerMeter;

            Raylib.DrawCircleV(
                new System.Numerics.Vector2(x, y),
                r,
                Color.SkyBlue);
        }

        // Draw angle constraint indicators (small arcs at vertices)
        for (int idx = 0; idx < world.Angles.Count; idx++)
        {
            var angle = world.Angles[idx];
            int i = angle.I;
            int j = angle.J;  // vertex
            int k = angle.K;

            float vx = world.PosX[j] * PixelsPerMeter;
            float vy = -world.PosY[j] * PixelsPerMeter;

            // Compute current angle
            float ux = world.PosX[i] - world.PosX[j];
            float uy = world.PosY[i] - world.PosY[j];
            float vx2 = world.PosX[k] - world.PosX[j];
            float vy2 = world.PosY[k] - world.PosY[j];

            float currentAngle = MathF.Atan2(ux * vy2 - uy * vx2, ux * vx2 + uy * vy2);
            float error = MathF.Abs(WrapAngle(currentAngle - angle.Theta0));

            // Color: Green if within tolerance, yellow if close, red if bad
            Color arcColor;
            if (error < 0.05f)  // < ~3 degrees
                arcColor = Color.Green;
            else if (error < 0.15f)  // < ~8.6 degrees
                arcColor = Color.Yellow;
            else
                arcColor = Color.Red;

            // Draw arc showing the ACTUAL current angle (not target)
            // Arc goes from edge i->j to edge j->k
            float arcRadius = 15f;

            // Convert world angles to screen angles (flip Y coordinate)
            // Direction from j to i in screen space
            float startAngleDeg = MathF.Atan2(-uy, ux) * 180f / MathF.PI;

            // Direction from j to k in screen space
            float endAngleDeg = MathF.Atan2(-vy2, vx2) * 180f / MathF.PI;

            // Make sure we draw the arc in the correct direction (smaller arc)
            // Handle wraparound
            float sweep = endAngleDeg - startAngleDeg;
            if (sweep > 180f) sweep -= 360f;
            if (sweep < -180f) sweep += 360f;

            if (sweep < 0f)
            {
                // Swap to draw counter-clockwise
                float temp = startAngleDeg;
                startAngleDeg = endAngleDeg;
                endAngleDeg = temp;
            }

            Raylib.DrawRing(
                new System.Numerics.Vector2(vx, vy),
                arcRadius - 2f,
                arcRadius,
                startAngleDeg,
                endAngleDeg,
                10,
                arcColor);

            // Also draw a small line showing the target angle for comparison
            // Target is relative to the first edge (from j to i)
            float targetAngleWorld = MathF.Atan2(uy, ux) + angle.Theta0;  // World space target
            float tx = vx + MathF.Cos(targetAngleWorld) * (arcRadius + 5f);
            float ty = vy - MathF.Sin(targetAngleWorld) * (arcRadius + 5f);  // Flip Y for screen

            Raylib.DrawLineEx(
                new System.Numerics.Vector2(vx, vy),
                new System.Numerics.Vector2(tx, ty),
                2f,
                new Color(0, 255, 0, 150));  // Semi-transparent bright green line for target
        }
    }

    private static void DrawUI(int stepCount, bool paused, WorldState world)
    {
        Raylib.DrawText($"Step: {stepCount}", 10, 10, 20, Color.White);
        Raylib.DrawText(paused ? "PAUSED (Space to resume)" : "Running (Space to pause)", 10, 35, 20, Color.White);
        Raylib.DrawText("R: Reset", 10, 60, 20, Color.White);
        Raylib.DrawText($"Particles: {world.ParticleCount}", 10, 85, 20, Color.White);
        Raylib.DrawText($"Angle Constraints: {world.Angles.Count}", 10, 110, 20, Color.White);

        // Count angle errors
        int good = 0, ok = 0, bad = 0;
        for (int idx = 0; idx < world.Angles.Count; idx++)
        {
            var angle = world.Angles[idx];
            int i = angle.I, j = angle.J, k = angle.K;

            float ux = world.PosX[i] - world.PosX[j];
            float uy = world.PosY[i] - world.PosY[j];
            float vx = world.PosX[k] - world.PosX[j];
            float vy = world.PosY[k] - world.PosY[j];

            float currentAngle = MathF.Atan2(ux * vy - uy * vx, ux * vx + uy * vy);
            float error = MathF.Abs(WrapAngle(currentAngle - angle.Theta0));

            if (error < 0.05f) good++;
            else if (error < 0.15f) ok++;
            else bad++;
        }

        Raylib.DrawText("Angle Error Status:", 10, 140, 20, Color.White);
        Raylib.DrawText($"  Good (<3°): {good}", 10, 165, 18, Color.Green);
        Raylib.DrawText($"  OK (3-8°): {ok}", 10, 185, 18, Color.Yellow);
        Raylib.DrawText($"  Bad (>8°): {bad}", 10, 205, 18, Color.Red);

        // Legend
        Raylib.DrawText("Legend:", WindowWidth - 200, 10, 20, Color.White);
        Raylib.DrawCircleV(new System.Numerics.Vector2(WindowWidth - 180, 45), 5f, Color.SkyBlue);
        Raylib.DrawText("Particle", WindowWidth - 165, 38, 16, Color.White);
        Raylib.DrawLineEx(new System.Numerics.Vector2(WindowWidth - 185, 70), new System.Numerics.Vector2(WindowWidth - 155, 70), 2f, Color.White);
        Raylib.DrawText("Rod", WindowWidth - 145, 63, 16, Color.White);
        Raylib.DrawRing(new System.Numerics.Vector2(WindowWidth - 175, 100), 8f, 10f, 0f, 90f, 10, Color.Green);
        Raylib.DrawText("Angle (good)", WindowWidth - 155, 93, 16, Color.White);
    }

    private static float WrapAngle(float angle)
    {
        return MathF.Atan2(MathF.Sin(angle), MathF.Cos(angle));
    }
}
