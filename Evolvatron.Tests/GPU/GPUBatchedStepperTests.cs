using Xunit;
using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace Evolvatron.Tests.GPU;

public class GPUBatchedStepperTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    public GPUBatchedStepperTests()
    {
        _context = Context.CreateDefault();
        _accelerator = _context.CreateCPUAccelerator(0);
    }

    [Fact]
    public void BatchedStepper_CanStepMultipleWorlds()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 10);
        using var worldState = new GPUBatchedWorldState(_accelerator, config);
        using var stepper = new GPUBatchedStepper(_accelerator);

        var simConfig = new SimulationConfig
        {
            Dt = 1f / 120f,
            GravityX = 0f,
            GravityY = -10f,
            XpbdIterations = 4,
            FrictionMu = 0.6f,
            Restitution = 0.0f,
            GlobalDamping = 0.01f,
            AngularDamping = 0.1f,
            VelocityStabilizationBeta = 1.0f,
            MaxVelocity = 20f
        };

        // Create a simple rocket template (just bodies, no collision for this test)
        var templateBodies = new GPURigidBody[3];
        templateBodies[0] = new GPURigidBody
        {
            X = 0f, Y = 5f, Angle = MathF.PI / 2f,
            VelX = 0f, VelY = 0f, AngularVel = 0f,
            InvMass = 1f / 5f, InvInertia = 1f / 2f
        };
        templateBodies[1] = new GPURigidBody
        {
            X = -0.5f, Y = 4f, Angle = 0f,
            VelX = 0f, VelY = 0f, AngularVel = 0f,
            InvMass = 1f / 1f, InvInertia = 1f / 0.5f
        };
        templateBodies[2] = new GPURigidBody
        {
            X = 0.5f, Y = 4f, Angle = 0f,
            VelX = 0f, VelY = 0f, AngularVel = 0f,
            InvMass = 1f / 1f, InvInertia = 1f / 0.5f
        };

        var templateGeoms = new GPURigidBodyGeom[config.GeomsPerWorld];
        var templateJoints = new GPURevoluteJoint[config.JointsPerWorld];

        worldState.UploadRocketTemplate(templateBodies, templateGeoms, templateJoints);

        // Upload ground collider (GPUOBBCollider uses CX, CY, UX, UY, HalfExtentX, HalfExtentY)
        var colliders = new GPUOBBCollider[config.SharedColliderCount];
        colliders[0] = new GPUOBBCollider
        {
            CX = 0f, CY = -5f,
            UX = 1f, UY = 0f,  // Axis-aligned: X-axis is (1,0)
            HalfExtentX = 20f, HalfExtentY = 0.5f
        };
        worldState.UploadSharedColliders(colliders);

        // Step multiple times (Step takes only worldState and simConfig)
        for (int i = 0; i < 100; i++)
        {
            stepper.Step(worldState, simConfig);
        }

        // Verify all worlds progressed (bodies should have fallen due to gravity)
        var allBodies = worldState.DownloadAllBodies();

        for (int w = 0; w < 10; w++)
        {
            int idx = config.GetRigidBodyIndex(w, 0);
            // After 100 steps with gravity, body should have fallen
            Assert.True(allBodies[idx].Y < 5f, $"World {w} body didn't fall");
        }
    }

    [Fact]
    public void BatchedStepper_AllWorldsProduceSameResult_WhenIdentical()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 5);
        using var worldState = new GPUBatchedWorldState(_accelerator, config);
        using var stepper = new GPUBatchedStepper(_accelerator);

        var simConfig = new SimulationConfig
        {
            Dt = 1f / 120f,
            GravityY = -10f,
            XpbdIterations = 4
        };

        // Create identical templates
        var templateBodies = new GPURigidBody[3];
        templateBodies[0] = new GPURigidBody
        {
            X = 0f, Y = 10f,
            InvMass = 0.2f, InvInertia = 0.5f
        };
        templateBodies[1] = new GPURigidBody { X = -1f, Y = 9f, InvMass = 1f, InvInertia = 2f };
        templateBodies[2] = new GPURigidBody { X = 1f, Y = 9f, InvMass = 1f, InvInertia = 2f };

        worldState.UploadRocketTemplate(templateBodies,
            new GPURigidBodyGeom[config.GeomsPerWorld],
            new GPURevoluteJoint[config.JointsPerWorld]);
        worldState.UploadSharedColliders(new GPUOBBCollider[config.SharedColliderCount]);

        // Step
        for (int i = 0; i < 50; i++)
        {
            stepper.Step(worldState, simConfig);
        }

        // All worlds should have identical state
        var allBodies = worldState.DownloadAllBodies();

        var world0Body0 = allBodies[config.GetRigidBodyIndex(0, 0)];

        for (int w = 1; w < 5; w++)
        {
            var body = allBodies[config.GetRigidBodyIndex(w, 0)];
            Assert.Equal(world0Body0.X, body.X, 4);
            Assert.Equal(world0Body0.Y, body.Y, 4);
            Assert.Equal(world0Body0.VelX, body.VelX, 4);
            Assert.Equal(world0Body0.VelY, body.VelY, 4);
        }
    }

    [Fact]
    public void BatchedStepper_ScalesTo100Worlds()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 100);
        using var worldState = new GPUBatchedWorldState(_accelerator, config);
        using var stepper = new GPUBatchedStepper(_accelerator);

        var simConfig = new SimulationConfig { Dt = 1f / 120f, GravityY = -10f };

        var templateBodies = new GPURigidBody[3];
        templateBodies[0] = new GPURigidBody { X = 0f, Y = 5f, InvMass = 0.2f, InvInertia = 0.5f };
        templateBodies[1] = new GPURigidBody { X = -1f, Y = 4f, InvMass = 1f, InvInertia = 2f };
        templateBodies[2] = new GPURigidBody { X = 1f, Y = 4f, InvMass = 1f, InvInertia = 2f };

        worldState.UploadRocketTemplate(templateBodies,
            new GPURigidBodyGeom[config.GeomsPerWorld],
            new GPURevoluteJoint[config.JointsPerWorld]);
        worldState.UploadSharedColliders(new GPUOBBCollider[config.SharedColliderCount]);

        // Step 100 worlds for 200 steps - should complete without error
        for (int i = 0; i < 200; i++)
        {
            stepper.Step(worldState, simConfig);
        }

        // Verify completion
        var bodies = worldState.DownloadAllBodies();
        Assert.Equal(config.TotalRigidBodies, bodies.Length);
    }

    public void Dispose()
    {
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
