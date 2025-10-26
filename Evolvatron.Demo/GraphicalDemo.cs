using Evolvatron.Core;
using Evolvatron.Core.Templates;
using Raylib_cs;
using System.Numerics;

namespace Evolvatron.Demo;

/// <summary>
/// Real-time graphical demo using Raylib.
/// Shows particles and rigid bodies with interactive controls.
/// </summary>
public static class GraphicalDemo
{
    private const int ScreenWidth = 1280;
    private const int ScreenHeight = 720;
    private const float MetersToPixels = 40f; // 40 pixels = 1 meter

    public static void Run()
    {
        Raylib.InitWindow(ScreenWidth, ScreenHeight, "Evolvatron - Physics Simulation");
        Raylib.SetTargetFPS(60);

        var world = new WorldState(initialCapacity: 256);
        var config = new SimulationConfig
        {
            Dt = 1f / 240f,
            Substeps = 1,
            XpbdIterations = 12,
            GravityX = 0f,
            GravityY = -9.81f,
            ContactCompliance = 1e-8f,
            RodCompliance = 0f,
            AngleCompliance = 0f,
            MotorCompliance = 1e-6f,
            FrictionMu = 0.6f,
            VelocityStabilizationBeta = 1.0f,
            GlobalDamping = 0.05f  // Normal damping - friction now works properly
        };

        var stepper = new CPUStepper();

        // Camera settings (world space)
        Vector2 cameraPos = new Vector2(0f, 0f); // Center of view in world coords
        float cameraZoom = 1.0f;

        // Simulation state
        bool paused = false;
        bool singleStep = false;
        int sceneIndex = 0;
        float simTime = 0f;

        // Build initial scene
        BuildScene(world, sceneIndex);

        while (!Raylib.WindowShouldClose())
        {
            // === INPUT ===
            if (Raylib.IsKeyPressed(KeyboardKey.Space))
                paused = !paused;

            if (Raylib.IsKeyPressed(KeyboardKey.S))
                singleStep = true;

            if (Raylib.IsKeyPressed(KeyboardKey.R))
            {
                world.Clear();
                BuildScene(world, sceneIndex);
                simTime = 0f;
            }

            if (Raylib.IsKeyPressed(KeyboardKey.Right))
            {
                sceneIndex = (sceneIndex + 1) % 6;
                world.Clear();
                BuildScene(world, sceneIndex);
                simTime = 0f;
            }

            if (Raylib.IsKeyPressed(KeyboardKey.Left))
            {
                sceneIndex = (sceneIndex - 1 + 6) % 6;
                world.Clear();
                BuildScene(world, sceneIndex);
                simTime = 0f;
            }

            // Camera zoom
            float wheel = Raylib.GetMouseWheelMove();
            if (wheel != 0)
            {
                cameraZoom *= (1f + wheel * 0.1f);
                cameraZoom = Math.Clamp(cameraZoom, 0.2f, 5f);
            }

            // Camera pan (WASD)
            float panSpeed = 0.3f / cameraZoom;
            if (Raylib.IsKeyDown(KeyboardKey.W)) cameraPos.Y += panSpeed;
            if (Raylib.IsKeyDown(KeyboardKey.A)) cameraPos.X -= panSpeed;
            if (Raylib.IsKeyDown(KeyboardKey.S)) cameraPos.Y -= panSpeed;
            if (Raylib.IsKeyDown(KeyboardKey.D)) cameraPos.X += panSpeed;

            // === SIMULATION ===
            if (!paused || singleStep)
            {
                // Run multiple sim steps per frame for real-time
                int stepsPerFrame = 4;
                for (int i = 0; i < stepsPerFrame; i++)
                {
                    stepper.Step(world, config);
                    simTime += config.Dt;
                }
                singleStep = false;
            }

            // === RENDERING ===
            Raylib.BeginDrawing();
            Raylib.ClearBackground(Color.Black);

            // Draw world
            DrawWorld(world, cameraPos, cameraZoom);

            // Draw UI
            DrawUI(world, simTime, paused, sceneIndex, cameraZoom);

            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();
    }

    private static void BuildScene(WorldState world, int index)
    {
        switch (index)
        {
            case 0:
                BuildSceneCapsuleTest(world);
                break;
            case 1:
                BuildSceneRigidBodies(world);
                break;
            case 2:
                BuildSceneFallingStick(world);
                break;
            case 3:
                BuildScenePendulum(world);
                break;
            case 4:
                BuildSceneMixed(world);
                break;
            case 5:
                BuildSceneRigidBodyRocket(world);
                break;
        }
    }

    private static void BuildSceneCapsuleTest(WorldState world)
    {
        // Simple test: capsule falls on sphere, then on box
        // Should come to rest on the box without spinning wildly

        // Ground (far below to catch if it falls through)
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -10f, 20f, 0.5f));

        // Sphere obstacle
        world.Circles.Add(new CircleCollider(0f, -2f, 1.5f));

        // Box platform below sphere
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -6f, 3f, 0.5f));

        // Single capsule starting above
        RigidBodyFactory.CreateCapsule(world, 0f, 3f, halfLength: 0.8f, radius: 0.3f, mass: 1f, angle: 0.3f);
    }

    private static void BuildSceneRigidBodies(WorldState world)
    {
        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));

        // Ramps
        float angle = MathF.PI / 6f;
        world.Obbs.Add(new OBBCollider(-10f, -4f, MathF.Cos(angle), MathF.Sin(angle), 3f, 0.3f));
        world.Obbs.Add(new OBBCollider(10f, -4f, MathF.Cos(-angle), MathF.Sin(-angle), 3f, 0.3f));

        // Central obstacle
        world.Circles.Add(new CircleCollider(0f, -3f, 2f));

        // Rigid bodies
        RigidBodyFactory.CreateBox(world, -8f, 5f, 0.5f, 0.5f, 1f, 0.3f);
        RigidBodyFactory.CreateBox(world, -5f, 8f, 0.6f, 0.4f, 1.2f, -0.2f);

        RigidBodyFactory.CreateCircle(world, 2f, 6f, 0.5f, 1f);
        RigidBodyFactory.CreateCircle(world, 5f, 9f, 0.4f, 0.8f);

        RigidBodyFactory.CreateCapsule(world, -2f, 10f, 0.8f, 0.3f, 1f, 0.5f);
        RigidBodyFactory.CreateCapsule(world, 8f, 7f, 0.6f, 0.25f, 0.9f, -0.3f);
    }

    private static void BuildSceneRocket(WorldState world)
    {
        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));

        // Landing platform
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -6f, 3f, 0.3f));

        // Rocket starting high up
        RocketTemplate.CreateRocket(world, centerX: 0f, centerY: 5f);
    }

    private static void BuildSceneFallingStick(WorldState world)
    {
        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));

        // Multiple ramps and obstacles
        world.Obbs.Add(new OBBCollider(-8f, -3f, MathF.Cos(0.3f), MathF.Sin(0.3f), 4f, 0.3f));
        world.Obbs.Add(new OBBCollider(8f, -3f, MathF.Cos(-0.3f), MathF.Sin(-0.3f), 4f, 0.3f));
        world.Circles.Add(new CircleCollider(-3f, 0f, 1.5f));
        world.Circles.Add(new CircleCollider(3f, 0f, 1.5f));

        // Several boxes, circles and capsules raining down
        for (int i = 0; i < 3; i++)
        {
            float x = -6f + i * 6f;
            RigidBodyFactory.CreateBox(world, x, 8f + i * 2f, 0.4f, 0.6f, 1.5f, i * 0.4f);
        }

        for (int i = 0; i < 3; i++)
        {
            float x = -5f + i * 5f;
            RigidBodyFactory.CreateCircle(world, x, 12f + i * 2f, 0.4f, 1.2f);
        }

        for (int i = 0; i < 2; i++)
        {
            float x = -3f + i * 6f;
            RigidBodyFactory.CreateCapsule(world, x, 16f + i * 2f, 0.7f, 0.3f, 1.3f, i * 0.5f);
        }
    }

    private static void BuildScenePendulum(WorldState world)
    {
        // Anchor point (pinned particle)
        int anchor = world.AddPinnedParticle(0f, 5f, 0.1f);

        // Pendulum bob
        int bob = world.AddParticle(-3f, 2f, 0f, 0f, 2f, 0.3f);

        // Rod connecting them
        float len = Math2D.Distance(world.PosX[anchor], world.PosY[anchor],
            world.PosX[bob], world.PosY[bob]);
        world.Rods.Add(new Rod(anchor, bob, len, 0f));

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));
    }

    private static void BuildSceneMixed(WorldState world)
    {
        // Terrain
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));
        world.Circles.Add(new CircleCollider(-5f, -3f, 1.5f));
        world.Circles.Add(new CircleCollider(5f, -3f, 1.5f));

        // Particle chain
        float chainY = 3f;
        int prev = world.AddPinnedParticle(-8f, chainY, 0.1f);
        for (int i = 0; i < 5; i++)
        {
            int curr = world.AddParticle(-8f + (i + 1) * 0.8f, chainY, 0f, 0f, 0.5f, 0.15f);
            float len = Math2D.Distance(world.PosX[prev], world.PosY[prev],
                world.PosX[curr], world.PosY[curr]);
            world.Rods.Add(new Rod(prev, curr, len, 0f));
            prev = curr;
        }

        // Rigid bodies
        RigidBodyFactory.CreateBox(world, 0f, 8f, 0.6f, 0.6f, 1.5f);
        RigidBodyFactory.CreateCircle(world, 3f, 10f, 0.5f, 1f);
        RigidBodyFactory.CreateCapsule(world, -3f, 12f, 1f, 0.3f, 1.2f);
    }

    private static void BuildSceneTiltedPlane(WorldState world)
    {
        // TILTED PLANE TEST - Boxes on a tilted ramp
        // Tests that multi-circle box approximation doesn't fall through angled surfaces

        // Create tilted plane at 20 degrees
        float tiltAngle = 20f * MathF.PI / 180f;
        float cos = MathF.Cos(tiltAngle);
        float sin = MathF.Sin(tiltAngle);

        // Tilted OBB - halfExtentY is the "thickness" perpendicular to the surface
        float planeHalfExtentX = 10f;
        float planeHalfExtentY = 0.5f;
        world.Obbs.Add(new OBBCollider(0f, -2f, cos, sin, planeHalfExtentX, planeHalfExtentY));

        // Ground below to catch anything that falls through
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -10f, 20f, 0.5f));

        // Create 3 boxes on the plane, tilted to match plane angle
        // Box halfExtent = 0.5, so we need clearance of planeHalfExtentY + boxDiagonal/2
        // For a tilted box, the diagonal in the perpendicular direction is approximately sqrt(0.5^2 + 0.5^2) = 0.707
        float planeY = -2f;
        float boxHalfExtent = 0.5f;
        float clearance = planeHalfExtentY + boxHalfExtent * 1.5f; // 0.5 + 0.75 = 1.25m clearance

        // Box 1 (left side of ramp)
        float box1AlongRamp = -5f;
        float box1X = box1AlongRamp * cos - clearance * sin;
        float box1Y = planeY + box1AlongRamp * sin + clearance * cos;
        RigidBodyFactory.CreateBox(world, box1X, box1Y, boxHalfExtent, boxHalfExtent, 2f, tiltAngle);

        // Box 2 (center of ramp)
        float box2AlongRamp = 0f;
        float box2X = box2AlongRamp * cos - clearance * sin;
        float box2Y = planeY + box2AlongRamp * sin + clearance * cos;
        RigidBodyFactory.CreateBox(world, box2X, box2Y, boxHalfExtent, boxHalfExtent, 2f, tiltAngle);

        // Box 3 (right side of ramp)
        float box3AlongRamp = 5f;
        float box3X = box3AlongRamp * cos - clearance * sin;
        float box3Y = planeY + box3AlongRamp * sin + clearance * cos;
        RigidBodyFactory.CreateBox(world, box3X, box3Y, boxHalfExtent, boxHalfExtent, 2f, tiltAngle);
    }

    private static void BuildSceneRigidBodyRocket(WorldState world)
    {
        // RIGID BODY ROCKET WITH JOINTS
        // Shows a rocket made from rigid bodies (capsules) connected by revolute joints
        // Unlike the particle-based rocket, this should be stable without explosions

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));

        // Landing platform
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -6f, 8f, 0.3f));

        // Create rigid body rocket (body + 2 legs connected by joints)
        RigidBodyRocketTemplate.CreateRocket(world,
            centerX: 0f,
            centerY: 5f,
            bodyHeight: 2f,
            bodyRadius: 0.3f,
            legLength: 1.5f,
            legRadius: 0.15f);

        // Add some obstacles for testing
        world.Circles.Add(new CircleCollider(-3f, -6f, 0.5f));
        world.Circles.Add(new CircleCollider(3f, -6f, 0.5f));
    }

    private static void DrawWorld(WorldState world, Vector2 cameraPos, float zoom)
    {
        // Draw static colliders
        foreach (var obb in world.Obbs)
            DrawOBB(obb, cameraPos, zoom, Color.Gray);

        foreach (var circle in world.Circles)
            DrawCircleCollider(circle, cameraPos, zoom, Color.Gray);

        foreach (var capsule in world.Capsules)
            DrawCapsuleCollider(capsule, cameraPos, zoom, Color.Gray);

        // Draw rods (constraints)
        for (int i = 0; i < world.Rods.Count; i++)
        {
            var rod = world.Rods[i];
            var p1 = WorldToScreen(world.PosX[rod.I], world.PosY[rod.I], cameraPos, zoom);
            var p2 = WorldToScreen(world.PosX[rod.J], world.PosY[rod.J], cameraPos, zoom);
            Raylib.DrawLineEx(p1, p2, 2f, Color.SkyBlue);
        }

        // Draw particles
        for (int i = 0; i < world.ParticleCount; i++)
        {
            var pos = WorldToScreen(world.PosX[i], world.PosY[i], cameraPos, zoom);
            float r = world.Radius[i] * MetersToPixels * zoom;
            bool isPinned = world.InvMass[i] == 0f;
            Color color = isPinned ? Color.Red : Color.Lime;
            Raylib.DrawCircleV(pos, r, color);
        }

        // Draw rigid bodies (as multi-circle geoms)
        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            DrawRigidBodyGeoms(rb, world, cameraPos, zoom);
        }

        // Draw revolute joints
        foreach (var joint in world.RevoluteJoints)
        {
            DrawRevoluteJoint(joint, world, cameraPos, zoom);
        }
    }

    private static void DrawRigidBodyGeoms(RigidBody rb, WorldState world, Vector2 cameraPos, float zoom)
    {
        float cos = MathF.Cos(rb.Angle);
        float sin = MathF.Sin(rb.Angle);

        // Draw each circle geom
        for (int g = 0; g < rb.GeomCount; g++)
        {
            var geom = world.RigidBodyGeoms[rb.GeomStartIndex + g];

            // Transform to world space
            float worldX = rb.X + geom.LocalX * cos - geom.LocalY * sin;
            float worldY = rb.Y + geom.LocalX * sin + geom.LocalY * cos;

            var pos = WorldToScreen(worldX, worldY, cameraPos, zoom);
            float r = geom.Radius * MetersToPixels * zoom;

            // Draw circle with semi-transparency to see overlaps
            Raylib.DrawCircleV(pos, r, new Color(255, 165, 0, 150));
            Raylib.DrawCircleLinesV(pos, r, Color.Orange);
        }

        // Draw "theoretical" box outline if this looks like a box (5 geoms)
        if (rb.GeomCount == 5)
        {
            // Assume it's a box - draw the theoretical rectangle
            // Corner circles are positioned inward by circleRadius, so:
            // actualHalfExtent = cornerOffset + circleRadius
            var geom1 = world.RigidBodyGeoms[rb.GeomStartIndex + 1]; // First corner circle
            float circleRadius = geom1.Radius;
            float cornerOffsetX = MathF.Abs(geom1.LocalX);
            float cornerOffsetY = MathF.Abs(geom1.LocalY);
            float halfExtentX = cornerOffsetX + circleRadius;
            float halfExtentY = cornerOffsetY + circleRadius;

            Vector2[] corners = new Vector2[4];
            Vector2[] localCorners = new[]
            {
                new Vector2(-halfExtentX, -halfExtentY),
                new Vector2(halfExtentX, -halfExtentY),
                new Vector2(halfExtentX, halfExtentY),
                new Vector2(-halfExtentX, halfExtentY)
            };

            for (int i = 0; i < 4; i++)
            {
                float wx = rb.X + localCorners[i].X * cos - localCorners[i].Y * sin;
                float wy = rb.Y + localCorners[i].X * sin + localCorners[i].Y * cos;
                corners[i] = WorldToScreen(wx, wy, cameraPos, zoom);
            }

            // Draw theoretical box outline in sky blue
            for (int i = 0; i < 4; i++)
                Raylib.DrawLineEx(corners[i], corners[(i + 1) % 4], 1f, Color.SkyBlue);
        }
        // Draw "theoretical" capsule outline if this looks like a capsule (3-7 geoms)
        else if (rb.GeomCount >= 3 && rb.GeomCount <= 7)
        {
            // Assume it's a capsule - estimate from first and last geom
            var firstGeom = world.RigidBodyGeoms[rb.GeomStartIndex];
            var lastGeom = world.RigidBodyGeoms[rb.GeomStartIndex + rb.GeomCount - 1];

            // Estimate halfLength from distance between endpoints
            float halfLength = MathF.Abs(lastGeom.LocalX - firstGeom.LocalX) * 0.5f;
            float radius = firstGeom.Radius;

            // Draw capsule as two circles connected by lines
            float x1 = rb.X + (-halfLength) * cos;
            float y1 = rb.Y + (-halfLength) * sin;
            float x2 = rb.X + halfLength * cos;
            float y2 = rb.Y + halfLength * sin;

            var p1 = WorldToScreen(x1, y1, cameraPos, zoom);
            var p2 = WorldToScreen(x2, y2, cameraPos, zoom);
            float r = radius * MetersToPixels * zoom;

            // Draw circles at ends
            Raylib.DrawCircleLinesV(p1, r, Color.SkyBlue);
            Raylib.DrawCircleLinesV(p2, r, Color.SkyBlue);

            // Draw connecting lines (perpendicular to capsule axis)
            float perpX = -sin;
            float perpY = cos;
            var p1Top = WorldToScreen(x1 + perpX * radius, y1 + perpY * radius, cameraPos, zoom);
            var p1Bot = WorldToScreen(x1 - perpX * radius, y1 - perpY * radius, cameraPos, zoom);
            var p2Top = WorldToScreen(x2 + perpX * radius, y2 + perpY * radius, cameraPos, zoom);
            var p2Bot = WorldToScreen(x2 - perpX * radius, y2 - perpY * radius, cameraPos, zoom);

            Raylib.DrawLineV(p1Top, p2Top, Color.SkyBlue);
            Raylib.DrawLineV(p1Bot, p2Bot, Color.SkyBlue);
        }

        // Draw orientation line from center of mass
        var centerPos = WorldToScreen(rb.X, rb.Y, cameraPos, zoom);
        float lineEndX = rb.X + MathF.Cos(rb.Angle) * 0.3f;
        float lineEndY = rb.Y + MathF.Sin(rb.Angle) * 0.3f;
        var lineEnd = WorldToScreen(lineEndX, lineEndY, cameraPos, zoom);
        Raylib.DrawLineEx(centerPos, lineEnd, 2f, Color.White);
    }

    private static void DrawOBB(OBBCollider obb, Vector2 cameraPos, float zoom, Color color)
    {
        Vector2[] corners = new Vector2[4];

        // Local corners
        Vector2[] localCorners = new[]
        {
            new Vector2(-obb.HalfExtentX, -obb.HalfExtentY),
            new Vector2(obb.HalfExtentX, -obb.HalfExtentY),
            new Vector2(obb.HalfExtentX, obb.HalfExtentY),
            new Vector2(-obb.HalfExtentX, obb.HalfExtentY)
        };

        for (int i = 0; i < 4; i++)
        {
            float wx = obb.CX + localCorners[i].X * obb.UX - localCorners[i].Y * obb.UY;
            float wy = obb.CY + localCorners[i].X * obb.UY + localCorners[i].Y * obb.UX;
            corners[i] = WorldToScreen(wx, wy, cameraPos, zoom);
        }

        for (int i = 0; i < 4; i++)
            Raylib.DrawLineEx(corners[i], corners[(i + 1) % 4], 3f, color);
    }

    private static void DrawCircleCollider(CircleCollider circle, Vector2 cameraPos, float zoom, Color color)
    {
        var pos = WorldToScreen(circle.CX, circle.CY, cameraPos, zoom);
        float r = circle.Radius * MetersToPixels * zoom;
        Raylib.DrawCircleLinesV(pos, r, color);
    }

    private static void DrawCapsuleCollider(CapsuleCollider capsule, Vector2 cameraPos, float zoom, Color color)
    {
        float x1 = capsule.CX - capsule.UX * capsule.HalfLength;
        float y1 = capsule.CY - capsule.UY * capsule.HalfLength;
        float x2 = capsule.CX + capsule.UX * capsule.HalfLength;
        float y2 = capsule.CY + capsule.UY * capsule.HalfLength;

        var p1 = WorldToScreen(x1, y1, cameraPos, zoom);
        var p2 = WorldToScreen(x2, y2, cameraPos, zoom);
        float r = capsule.Radius * MetersToPixels * zoom;

        Raylib.DrawLineEx(p1, p2, 3f, color);
        Raylib.DrawCircleLinesV(p1, r, color);
        Raylib.DrawCircleLinesV(p2, r, color);
    }

    private static void DrawRevoluteJoint(RevoluteJoint joint, WorldState world, Vector2 cameraPos, float zoom)
    {
        var bodyA = world.RigidBodies[joint.BodyA];
        var bodyB = world.RigidBodies[joint.BodyB];

        // Transform anchors to world space
        float cosA = MathF.Cos(bodyA.Angle);
        float sinA = MathF.Sin(bodyA.Angle);
        float anchorAX = bodyA.X + joint.LocalAnchorAX * cosA - joint.LocalAnchorAY * sinA;
        float anchorAY = bodyA.Y + joint.LocalAnchorAX * sinA + joint.LocalAnchorAY * cosA;

        float cosB = MathF.Cos(bodyB.Angle);
        float sinB = MathF.Sin(bodyB.Angle);
        float anchorBX = bodyB.X + joint.LocalAnchorBX * cosB - joint.LocalAnchorBY * sinB;
        float anchorBY = bodyB.Y + joint.LocalAnchorBX * sinB + joint.LocalAnchorBY * cosB;

        var posA = WorldToScreen(anchorAX, anchorAY, cameraPos, zoom);
        var posB = WorldToScreen(anchorBX, anchorBY, cameraPos, zoom);

        // Draw anchor points
        Raylib.DrawCircleV(posA, 5f, Color.Red);
        Raylib.DrawCircleV(posB, 5f, Color.Blue);

        // Draw connection line (should be very short if joint is working correctly)
        Raylib.DrawLineEx(posA, posB, 2f, Color.Yellow);

        // Draw lines to body centers to visualize the joint structure
        var centerA = WorldToScreen(bodyA.X, bodyA.Y, cameraPos, zoom);
        var centerB = WorldToScreen(bodyB.X, bodyB.Y, cameraPos, zoom);
        Raylib.DrawLineEx(centerA, posA, 1f, new Color(255, 0, 0, 100));
        Raylib.DrawLineEx(centerB, posB, 1f, new Color(0, 0, 255, 100));
    }

    private static Vector2 WorldToScreen(float worldX, float worldY, Vector2 cameraPos, float zoom)
    {
        float screenX = (worldX - cameraPos.X) * MetersToPixels * zoom + ScreenWidth / 2f;
        float screenY = ScreenHeight / 2f - (worldY - cameraPos.Y) * MetersToPixels * zoom;
        return new Vector2(screenX, screenY);
    }

    private static void DrawUI(WorldState world, float simTime, bool paused, int sceneIndex, float zoom)
    {
        string[] sceneNames = { "Capsule Test", "Rigid Bodies", "RB Rain", "Pendulum", "Mixed", "RB Rocket+Joints" };

        Raylib.DrawText($"FPS: {Raylib.GetFPS()}", 10, 10, 20, Color.Green);
        Raylib.DrawText($"Time: {simTime:F2}s", 10, 35, 20, Color.Green);
        Raylib.DrawText($"Scene: {sceneNames[sceneIndex]} ({sceneIndex + 1}/6)", 10, 60, 20, Color.Green);
        Raylib.DrawText($"Particles: {world.ParticleCount}, Rigid Bodies: {world.RigidBodies.Count}", 10, 85, 20, Color.Green);
        Raylib.DrawText($"Zoom: {zoom:F1}x", 10, 110, 20, Color.Green);

        // Debug info for rigid bodies
        if (world.RigidBodies.Count > 0)
        {
            var rb = world.RigidBodies[0];
            Raylib.DrawText($"RB #0: pos=({rb.X:F2}, {rb.Y:F2})", 10, 135, 18, Color.Yellow);
            Raylib.DrawText($"  vel=({rb.VelX:F2}, {rb.VelY:F2}) ω={rb.AngularVel:F2}", 10, 155, 18, Color.Yellow);
            float speed = MathF.Sqrt(rb.VelX * rb.VelX + rb.VelY * rb.VelY);
            Raylib.DrawText($"  speed={speed:F2} m/s", 10, 175, 18, Color.Yellow);
        }

        if (paused)
            Raylib.DrawText("PAUSED", ScreenWidth / 2 - 80, 20, 40, Color.Red);

        // Controls
        int y = ScreenHeight - 150;
        Raylib.DrawText("Controls:", 10, y, 20, Color.White);
        Raylib.DrawText("  SPACE - Pause/Resume", 10, y + 25, 18, Color.LightGray);
        Raylib.DrawText("  S - Single Step", 10, y + 45, 18, Color.LightGray);
        Raylib.DrawText("  R - Reset Scene", 10, y + 65, 18, Color.LightGray);
        Raylib.DrawText("  LEFT/RIGHT - Change Scene", 10, y + 85, 18, Color.LightGray);
        Raylib.DrawText("  WASD - Pan Camera", 10, y + 105, 18, Color.LightGray);
        Raylib.DrawText("  MOUSE WHEEL - Zoom", 10, y + 125, 18, Color.LightGray);
    }
}
