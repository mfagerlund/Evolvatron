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

        // Particle coloring metadata (for scene 7 - particle boxes)
        List<int> angleConstraintParticles = new List<int>();
        List<int> diagonalBracingParticles = new List<int>();

        // Build initial scene
        BuildScene(world, sceneIndex, angleConstraintParticles, diagonalBracingParticles);

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
                angleConstraintParticles.Clear();
                diagonalBracingParticles.Clear();
                BuildScene(world, sceneIndex, angleConstraintParticles, diagonalBracingParticles);
                simTime = 0f;
            }

            if (Raylib.IsKeyPressed(KeyboardKey.Right))
            {
                sceneIndex = (sceneIndex + 1) % 8;  // Updated to 8 scenes
                world.Clear();
                angleConstraintParticles.Clear();
                diagonalBracingParticles.Clear();
                BuildScene(world, sceneIndex, angleConstraintParticles, diagonalBracingParticles);
                simTime = 0f;
            }

            if (Raylib.IsKeyPressed(KeyboardKey.Left))
            {
                sceneIndex = (sceneIndex - 1 + 8) % 8;  // Updated to 8 scenes
                world.Clear();
                angleConstraintParticles.Clear();
                diagonalBracingParticles.Clear();
                BuildScene(world, sceneIndex, angleConstraintParticles, diagonalBracingParticles);
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
                    // Update snake virtual particles for scene 7 (before physics step)
                    if (sceneIndex == 7)
                    {
                        UpdateMorphingBoxMotors(world, simTime);
                        CancelGravityOnVirtualParticles(world, config);
                    }

                    stepper.Step(world, config);
                    simTime += config.Dt;
                }
                singleStep = false;
            }

            // === RENDERING ===
            Raylib.BeginDrawing();
            Raylib.ClearBackground(Color.Black);

            // Draw world
            DrawWorld(world, cameraPos, cameraZoom, sceneIndex, angleConstraintParticles, diagonalBracingParticles);

            // Draw UI
            DrawUI(world, simTime, paused, sceneIndex, cameraZoom);

            Raylib.EndDrawing();
        }

        Raylib.CloseWindow();
    }

    private static void CancelGravityOnVirtualParticles(WorldState world, SimulationConfig config)
    {
        // Virtual particles should not be affected by gravity
        // They're control particles, not physical mass
        // Cancel out gravity that was applied during integration

        for (int i = 0; i < world.ParticleCount; i++)
        {
            // Check if this is a virtual particle (small radius)
            if (world.Radius[i] < 0.06f && world.InvMass[i] > 0f)
            {
                // Cancel gravity by removing what was added
                world.VelY[i] -= config.GravityY * config.Dt;
            }
        }
    }

    private static void UpdateMorphingBoxMotors(WorldState world, float time)
    {
        // Make snakes slither by moving their virtual particles in a wave pattern
        // Virtual particles are placed perpendicular to joints
        // Moving them creates bending motion

        float frequency = 1.0f; // Wave frequency
        float amplitude = 0.3f; // How far virtual particles move

        // We need to identify which particles are virtual particles
        // They're the smaller ones (radius 0.04)
        // For now, just apply forces to all small particles

        for (int i = 0; i < world.ParticleCount; i++)
        {
            // Check if this is a virtual particle (small radius)
            if (world.Radius[i] < 0.06f)
            {
                // This is a virtual particle
                // Apply sinusoidal force perpendicular to its current position

                // Use particle position as phase offset for traveling wave
                float phase = world.PosX[i] * 0.5f;
                float wave = MathF.Sin(2f * MathF.PI * frequency * time + phase);

                // Apply force in Y direction (perpendicular to snake body)
                float targetOffset = amplitude * wave;

                // Find the nearest segment particle to get its Y position
                // For simplicity, just oscillate around initial position
                float targetY = world.PosY[i] + targetOffset;

                // Apply gentle force to reach target
                float dy = targetOffset * 10f; // Force proportional to desired movement
                world.VelY[i] += dy * 0.01f;
            }
        }
    }

    private static void BuildScene(WorldState world, int index, List<int> angleConstraintParticles, List<int> diagonalBracingParticles)
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
            case 6:
                BuildSceneParticleGrid(world);
                break;
            case 7:
                BuildSceneParticleBoxes(world, angleConstraintParticles, diagonalBracingParticles);
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

    private static void BuildSceneParticleGrid(WorldState world)
    {
        // 8x8 PARTICLE GRID (SOFT-BODY CLOTH)
        // Shows interconnected particles forming a deformable cloth-like grid
        // Uses distance constraints (rods) for structure

        // Ground platform
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));

        // Landing platform (angled)
        float platformAngle = 0.2f;
        world.Obbs.Add(new OBBCollider(0f, -5f, MathF.Cos(platformAngle), MathF.Sin(platformAngle), 6f, 0.3f));

        // Create 8x8 grid
        const int gridWidth = 8;
        const int gridHeight = 8;
        float spacing = 0.4f; // Distance between adjacent particles
        float particleMass = 0.1f;
        float particleRadius = 0.08f;
        float startX = -(gridWidth - 1) * spacing / 2f;
        float startY = 5f;

        // Create particles
        int[,] particleIndices = new int[gridWidth, gridHeight];
        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                float px = startX + x * spacing;
                float py = startY + y * spacing;

                particleIndices[x, y] = world.AddParticle(
                    x: px,
                    y: py,
                    vx: 0f,
                    vy: 0f,
                    mass: particleMass,
                    radius: particleRadius);
            }
        }

        // Connect with structural rods (horizontal and vertical)
        float rodCompliance = 5e-4f; // Soft for cloth-like behavior (was 1e-5 = too stiff!)

        // Horizontal connections
        for (int x = 0; x < gridWidth - 1; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                world.Rods.Add(new Rod(
                    particleIndices[x, y],
                    particleIndices[x + 1, y],
                    restLength: spacing,
                    compliance: rodCompliance));
            }
        }

        // Vertical connections
        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight - 1; y++)
            {
                world.Rods.Add(new Rod(
                    particleIndices[x, y],
                    particleIndices[x, y + 1],
                    restLength: spacing,
                    compliance: rodCompliance));
            }
        }

        // Diagonal connections (shear resistance - prevents sliding)
        float diagonalLength = spacing * MathF.Sqrt(2f);
        for (int x = 0; x < gridWidth - 1; x++)
        {
            for (int y = 0; y < gridHeight - 1; y++)
            {
                // Diagonal 1: top-left to bottom-right
                world.Rods.Add(new Rod(
                    particleIndices[x, y],
                    particleIndices[x + 1, y + 1],
                    restLength: diagonalLength,
                    compliance: rodCompliance * 2f)); // Softer diagonals

                // Diagonal 2: top-right to bottom-left
                world.Rods.Add(new Rod(
                    particleIndices[x + 1, y],
                    particleIndices[x, y + 1],
                    restLength: diagonalLength,
                    compliance: rodCompliance * 2f));
            }
        }

        // Optional: Pin top corners to make it hang like a cloth
        // Uncomment these lines to pin the top-left and top-right corners
        // world.InvMass[particleIndices[0, gridHeight - 1]] = 0f;  // Pin top-left
        // world.InvMass[particleIndices[gridWidth - 1, gridHeight - 1]] = 0f;  // Pin top-right

        // Add some obstacles for the cloth to interact with
        world.Circles.Add(new CircleCollider(-2f, -3f, 1.2f));
        world.Circles.Add(new CircleCollider(2f, -2f, 0.8f));
    }

    private static void BuildSceneParticleBoxes(WorldState world, List<int> angleConstraintParticles, List<int> diagonalBracingParticles)
    {
        // SNAKES WITH VIRTUAL PARTICLE ANGLE CONTROL
        // Creates articulated chains (snakes) that slither using angle constraints
        // Uses virtual particles perpendicular to joints to control angles

        // Ground
        world.Obbs.Add(OBBCollider.AxisAligned(0f, -8f, 20f, 0.5f));

        // Create a random number generator with fixed seed for reproducible results
        var random = new Random(42);

        // Create 5 snakes with different lengths
        int numSnakes = 5;
        for (int snakeIdx = 0; snakeIdx < numSnakes; snakeIdx++)
        {
            // Random snake parameters
            int segments = 4 + snakeIdx; // 4 to 8 segments
            float segmentLength = 0.5f;
            float startX = -10f + snakeIdx * 4.5f;
            float startY = 8f + snakeIdx * 1.5f;
            float phaseOffset = snakeIdx * MathF.PI / 2.5f;

            var snakeParticles = CreateSnakeWithVirtualParticles(
                world, startX, startY, segments, segmentLength, phaseOffset);

            angleConstraintParticles.AddRange(snakeParticles);
        }

        // Add some obstacles
        world.Circles.Add(new CircleCollider(-8f, -4f, 1.2f));
        world.Circles.Add(new CircleCollider(0f, -5f, 1.5f));
        world.Circles.Add(new CircleCollider(8f, -3f, 1.0f));
        world.Obbs.Add(new OBBCollider(-4f, -6f, MathF.Cos(0.2f), MathF.Sin(0.2f), 3f, 0.3f));
        world.Obbs.Add(new OBBCollider(4f, -6f, MathF.Cos(-0.2f), MathF.Sin(-0.2f), 3f, 0.3f));
    }

    private static List<int> CreateSnakeWithVirtualParticles(
        WorldState world,
        float startX,
        float startY,
        int numSegments,
        float segmentLength,
        float phaseOffset)
    {
        // Create a snake chain using virtual particles for angle control
        // Snake structure: p0--p1--p2--p3--p4...
        // At each joint (p1, p2, p3...), add a virtual particle perpendicular
        // The virtual particle controls the bend angle at that joint

        var allParticles = new List<int>();

        float particleMass = 0.2f;
        float particleRadius = 0.08f;
        float virtualMass = 0.05f;      // Virtual particles are lighter
        float virtualRadius = 0.04f;     // Smaller visual
        float rodCompliance = 0f;        // Rigid segments
        float virtualCompliance = 1e-5f; // Slightly soft angle control

        // Create segment particles in a horizontal chain
        var segments = new List<int>();
        for (int i = 0; i <= numSegments; i++)
        {
            float px = startX + i * segmentLength;
            float py = startY;
            int p = world.AddParticle(px, py, 0f, 0f, particleMass, particleRadius);
            segments.Add(p);
            allParticles.Add(p);
        }

        // Connect segments with rigid rods
        for (int i = 0; i < numSegments; i++)
        {
            world.Rods.Add(new Rod(segments[i], segments[i + 1], segmentLength, rodCompliance));
        }

        // Add virtual particles at each INTERIOR joint (not endpoints)
        // Each virtual particle is placed perpendicular to the joint
        float virtualDist = 0.15f; // Distance perpendicular to joint

        for (int i = 1; i < numSegments; i++) // Interior joints only
        {
            int prev = segments[i - 1];
            int curr = segments[i];
            int next = segments[i + 1];

            // Place virtual particle perpendicular to current joint
            // Initial position: offset perpendicular (down from the chain)
            float vx = world.PosX[curr];
            float vy = world.PosY[curr] - virtualDist;

            int v = world.AddParticle(vx, vy, 0f, 0f, virtualMass, virtualRadius);
            allParticles.Add(v);

            // Connect virtual particle to the two arms of the angle
            // This forms a triangle: prev--curr--next with V controlling the angle
            float armDist = MathF.Sqrt(segmentLength * segmentLength + virtualDist * virtualDist);

            world.Rods.Add(new Rod(v, prev, armDist, virtualCompliance)); // V to previous segment
            world.Rods.Add(new Rod(v, next, armDist, virtualCompliance)); // V to next segment
        }

        return allParticles;
    }

    private static int[] CreateMorphingParticleBox(
        WorldState world,
        float centerX,
        float centerY,
        float width,
        float height,
        float angle,
        float angularVel,
        float phaseOffset)
    {
        // Create a box using VIRTUAL PARTICLES for angle control
        // Each corner gets a virtual particle placed inside the angle
        // The virtual particle position defines the corner angle

        float halfW = width / 2f;
        float halfH = height / 2f;
        float particleMass = 0.3f;
        float particleRadius = 0.1f;
        float virtualMass = 0.1f; // Virtual particles are lighter
        float virtualRadius = 0.05f; // Smaller visual
        float rodCompliance = 1e-6f; // Very stiff for edges

        // Calculate rotated corner positions
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);

        // Local corner positions (before rotation)
        Vector2[] localCorners = new[]
        {
            new Vector2(-halfW, -halfH), // Bottom-left
            new Vector2(halfW, -halfH),  // Bottom-right
            new Vector2(halfW, halfH),   // Top-right
            new Vector2(-halfW, halfH)   // Top-left
        };

        // Create corner particles
        int[] corners = new int[4];
        for (int i = 0; i < 4; i++)
        {
            float localX = localCorners[i].X;
            float localY = localCorners[i].Y;
            float worldX = centerX + localX * cos - localY * sin;
            float worldY = centerY + localX * sin + localY * cos;

            float velX = -angularVel * localY;
            float velY = angularVel * localX;

            corners[i] = world.AddParticle(worldX, worldY, velX, velY, particleMass, particleRadius);
        }

        // Add edge rods (4 rods connecting corners in a loop)
        world.Rods.Add(new Rod(corners[0], corners[1], width, rodCompliance));  // Bottom edge
        world.Rods.Add(new Rod(corners[1], corners[2], height, rodCompliance)); // Right edge
        world.Rods.Add(new Rod(corners[2], corners[3], width, rodCompliance));  // Top edge
        world.Rods.Add(new Rod(corners[3], corners[0], height, rodCompliance)); // Left edge

        // Add VIRTUAL PARTICLES at two opposing corners (bottom-left and top-right)
        // Place them inside the box at a fixed distance from the corner
        float virtualDist = 0.2f; // Distance from corner to virtual particle

        // Virtual particle for bottom-left corner (inside box)
        float vbl_x = world.PosX[corners[0]] + virtualDist * MathF.Cos(MathF.PI * 0.25f);
        float vbl_y = world.PosY[corners[0]] + virtualDist * MathF.Sin(MathF.PI * 0.25f);
        int vbl = world.AddParticle(vbl_x, vbl_y, 0f, 0f, virtualMass, virtualRadius);

        // Virtual particle for top-right corner (inside box)
        float vtr_x = world.PosX[corners[2]] - virtualDist * MathF.Cos(MathF.PI * 0.25f);
        float vtr_y = world.PosY[corners[2]] - virtualDist * MathF.Sin(MathF.PI * 0.25f);
        int vtr = world.AddParticle(vtr_x, vtr_y, 0f, 0f, virtualMass, virtualRadius);

        // Connect virtual particles to their corner arms
        // Bottom-left corner: connect V to left-arm (p3) and bottom-arm (p1)
        float armDist = MathF.Sqrt(virtualDist * virtualDist + virtualDist * virtualDist); // Distance to arms
        world.Rods.Add(new Rod(vbl, corners[3], armDist, 1e-5f)); // V to top-left
        world.Rods.Add(new Rod(vbl, corners[1], armDist, 1e-5f)); // V to bottom-right

        // Top-right corner: connect V to right-arm (p1) and top-arm (p3)
        world.Rods.Add(new Rod(vtr, corners[1], armDist, 1e-5f)); // V to bottom-right
        world.Rods.Add(new Rod(vtr, corners[3], armDist, 1e-5f)); // V to top-left

        return corners;
    }

    private static int[] CreateParticleBoxWithDirectAngleConstraints(
        WorldState world,
        float centerX,
        float centerY,
        float width,
        float height,
        float angle,
        float angularVel)
    {
        // Create a box from 4 corner particles
        // Use rods for edges and DIRECT 3-point angle constraints (world.Angles.Add)
        // WARNING: This version will fail/explode due to over-constraint!

        float halfW = width / 2f;
        float halfH = height / 2f;
        float particleMass = 0.3f;
        float particleRadius = 0.1f;
        float rodCompliance = 1e-6f; // Very stiff for edges

        // Calculate rotated corner positions
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);

        // Local corner positions (before rotation)
        Vector2[] localCorners = new[]
        {
            new Vector2(-halfW, -halfH), // Bottom-left
            new Vector2(halfW, -halfH),  // Bottom-right
            new Vector2(halfW, halfH),   // Top-right
            new Vector2(-halfW, halfH)   // Top-left
        };

        // Create particles at rotated positions with initial velocity from angular motion
        int[] corners = new int[4];
        for (int i = 0; i < 4; i++)
        {
            // Rotate corner position
            float localX = localCorners[i].X;
            float localY = localCorners[i].Y;
            float worldX = centerX + localX * cos - localY * sin;
            float worldY = centerY + localX * sin + localY * cos;

            // Calculate velocity from angular motion: v = ω × r
            float velX = -angularVel * localY;
            float velY = angularVel * localX;

            corners[i] = world.AddParticle(
                x: worldX,
                y: worldY,
                vx: velX,
                vy: velY,
                mass: particleMass,
                radius: particleRadius);
        }

        // Add edge rods (4 rods connecting corners in a loop)
        world.Rods.Add(new Rod(corners[0], corners[1], width, rodCompliance));  // Bottom edge
        world.Rods.Add(new Rod(corners[1], corners[2], height, rodCompliance)); // Right edge
        world.Rods.Add(new Rod(corners[2], corners[3], width, rodCompliance));  // Top edge
        world.Rods.Add(new Rod(corners[3], corners[0], height, rodCompliance)); // Left edge

        // Add DIRECT 3-point angle constraints (the problematic approach!)
        // These will conflict with the rods and contacts, causing instability

        // Bottom-left corner angle (90 degrees)
        world.Angles.Add(new Angle(
            i: corners[3],       // Left side endpoint (top-left)
            j: corners[0],       // Vertex of angle (bottom-left corner)
            k: corners[1],       // Bottom side endpoint (bottom-right)
            theta0: MathF.PI / 2f,
            compliance: rodCompliance));

        // Top-right corner angle (90 degrees)
        world.Angles.Add(new Angle(
            i: corners[1],       // Bottom side endpoint (bottom-right)
            j: corners[2],       // Vertex of angle (top-right corner)
            k: corners[3],       // Top side endpoint (top-left)
            theta0: MathF.PI / 2f,
            compliance: rodCompliance));

        return corners;
    }

    private static int[] CreateParticleBoxWithAngleConstraints(
        WorldState world,
        float centerX,
        float centerY,
        float width,
        float height,
        float angle,
        float angularVel)
    {
        // Create a box from 4 corner particles
        // Use rods for edges and angle constraints (as diagonal rods) on opposing corners
        // This version uses the AddAngleConstraintAsRod API

        float halfW = width / 2f;
        float halfH = height / 2f;
        float particleMass = 0.3f;
        float particleRadius = 0.1f;
        float rodCompliance = 1e-6f; // Very stiff for edges

        // Calculate rotated corner positions
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);

        // Local corner positions (before rotation)
        Vector2[] localCorners = new[]
        {
            new Vector2(-halfW, -halfH), // Bottom-left
            new Vector2(halfW, -halfH),  // Bottom-right
            new Vector2(halfW, halfH),   // Top-right
            new Vector2(-halfW, halfH)   // Top-left
        };

        // Create particles at rotated positions with initial velocity from angular motion
        int[] corners = new int[4];
        for (int i = 0; i < 4; i++)
        {
            // Rotate corner position
            float localX = localCorners[i].X;
            float localY = localCorners[i].Y;
            float worldX = centerX + localX * cos - localY * sin;
            float worldY = centerY + localX * sin + localY * cos;

            // Calculate velocity from angular motion: v = ω × r
            float velX = -angularVel * localY;
            float velY = angularVel * localX;

            corners[i] = world.AddParticle(
                x: worldX,
                y: worldY,
                vx: velX,
                vy: velY,
                mass: particleMass,
                radius: particleRadius);
        }

        // Add edge rods (4 rods connecting corners in a loop)
        world.Rods.Add(new Rod(corners[0], corners[1], width, rodCompliance));  // Bottom edge
        world.Rods.Add(new Rod(corners[1], corners[2], height, rodCompliance)); // Right edge
        world.Rods.Add(new Rod(corners[2], corners[3], width, rodCompliance));  // Top edge
        world.Rods.Add(new Rod(corners[3], corners[0], height, rodCompliance)); // Left edge

        // Add angle constraints on two opposing corners (bottom-left and top-right)
        // These prevent the box from shearing without using diagonal crossbracing
        // We use the diagonal rod method which is stable for rigid structures

        // Bottom-left corner angle (90 degrees)
        // Constrains the angle at corner[0] between edges to corner[3] and corner[1]
        world.AddAngleConstraintAsRod(
            i: corners[3],       // Left side endpoint (top-left)
            j: corners[0],       // Vertex of angle (bottom-left corner)
            k: corners[1],       // Bottom side endpoint (bottom-right)
            targetAngle: MathF.PI / 2f,
            len1: height,        // Left edge length (corner[0] to corner[3])
            len2: width,         // Bottom edge length (corner[0] to corner[1])
            compliance: rodCompliance);

        // Top-right corner angle (90 degrees)
        // Constrains the angle at corner[2] between edges to corner[1] and corner[3]
        world.AddAngleConstraintAsRod(
            i: corners[1],       // Bottom side endpoint (bottom-right)
            j: corners[2],       // Vertex of angle (top-right corner)
            k: corners[3],       // Top side endpoint (top-left)
            targetAngle: MathF.PI / 2f,
            len1: height,        // Right edge length (corner[2] to corner[1])
            len2: width,         // Top edge length (corner[2] to corner[3])
            compliance: rodCompliance);

        return corners;
    }

    private static void DrawWorld(WorldState world, Vector2 cameraPos, float zoom, int sceneIndex, List<int> angleConstraintParticles, List<int> diagonalBracingParticles)
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

        // Draw particles with color-coding for scene 7 (particle boxes comparison)
        for (int i = 0; i < world.ParticleCount; i++)
        {
            var pos = WorldToScreen(world.PosX[i], world.PosY[i], cameraPos, zoom);
            float r = world.Radius[i] * MetersToPixels * zoom;
            bool isPinned = world.InvMass[i] == 0f;

            Color color;
            if (isPinned)
            {
                color = Color.Red; // Pinned particles are always red
            }
            else if (sceneIndex == 7)
            {
                // Scene 7: Color-code particles by box type
                if (angleConstraintParticles.Contains(i))
                    color = new Color(0, 255, 255, 255);  // Cyan - LEFT side: Angle constraints
                else if (diagonalBracingParticles.Contains(i))
                    color = new Color(255, 0, 255, 255);  // Magenta - RIGHT side: Diagonal bracing
                else
                    color = Color.Lime;  // Default (shouldn't happen in scene 7)
            }
            else
            {
                color = Color.Lime; // Default color for other scenes
            }

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
        string[] sceneNames = { "Capsule Test", "Rigid Bodies", "RB Rain", "Pendulum", "Mixed", "RB Rocket+Joints", "Particle Grid (Cloth)", "Slithering Snakes (Virtual Particles)" };

        Raylib.DrawText($"FPS: {Raylib.GetFPS()}", 10, 10, 20, Color.Green);
        Raylib.DrawText($"Time: {simTime:F2}s", 10, 35, 20, Color.Green);
        Raylib.DrawText($"Scene: {sceneNames[sceneIndex]} ({sceneIndex + 1}/8)", 10, 60, 20, Color.Green);
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
