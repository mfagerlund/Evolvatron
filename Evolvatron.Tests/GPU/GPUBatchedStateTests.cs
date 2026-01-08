using Xunit;
using Evolvatron.Core.GPU.Batched;
using Evolvatron.Core.GPU;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace Evolvatron.Tests.GPU;

public class GPUBatchedStateTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    public GPUBatchedStateTests()
    {
        _context = Context.CreateDefault();
        // Use CPU accelerator for testing (works without GPU)
        _accelerator = _context.CreateCPUAccelerator(0);
    }

    [Fact]
    public void GPUBatchedWorldConfig_IndexComputation_IsCorrect()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 10);

        // Test forward index computation
        Assert.Equal(0, config.GetRigidBodyIndex(0, 0));
        Assert.Equal(1, config.GetRigidBodyIndex(0, 1));
        Assert.Equal(3, config.GetRigidBodyIndex(1, 0)); // world 1, local 0

        // Test reverse computation
        Assert.Equal(0, config.GetWorldFromRigidBodyIndex(0));
        Assert.Equal(0, config.GetWorldFromRigidBodyIndex(2));
        Assert.Equal(1, config.GetWorldFromRigidBodyIndex(3));
    }

    [Fact]
    public void GPUBatchedWorldConfig_TotalCounts_AreCorrect()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 100, targetsPerWorld: 5);

        Assert.Equal(100, config.WorldCount);
        Assert.Equal(300, config.TotalRigidBodies); // 100 * 3
        Assert.Equal(500, config.TotalTargets); // 100 * 5
    }

    [Fact]
    public void GPUBatchedWorldState_CanAllocateAndDispose()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 10);

        using var state = new GPUBatchedWorldState(_accelerator, config);

        // Verify buffers were allocated with correct sizes
        Assert.Equal(config.TotalRigidBodies, state.RigidBodies.Length);
        Assert.Equal(config.TotalGeoms, state.Geoms.Length);
        Assert.Equal(config.TotalJoints, state.Joints.Length);
    }

    [Fact]
    public void GPUBatchedWorldState_UploadTemplate_ExpandsToAllWorlds()
    {
        var config = GPUBatchedWorldConfig.ForRocketChase(worldCount: 5);
        using var state = new GPUBatchedWorldState(_accelerator, config);

        // Create a simple template (just set X position to identify each body)
        var templateBodies = new GPURigidBody[3];
        templateBodies[0] = new GPURigidBody { X = 1.0f, Y = 2.0f };
        templateBodies[1] = new GPURigidBody { X = 3.0f, Y = 4.0f };
        templateBodies[2] = new GPURigidBody { X = 5.0f, Y = 6.0f };

        var templateGeoms = new GPURigidBodyGeom[config.GeomsPerWorld];
        var templateJoints = new GPURevoluteJoint[config.JointsPerWorld];

        state.UploadRocketTemplate(templateBodies, templateGeoms, templateJoints);

        // Download and verify all worlds have the template
        var allBodies = state.DownloadAllBodies();

        for (int w = 0; w < 5; w++)
        {
            int idx0 = config.GetRigidBodyIndex(w, 0);
            int idx1 = config.GetRigidBodyIndex(w, 1);

            Assert.Equal(1.0f, allBodies[idx0].X);
            Assert.Equal(3.0f, allBodies[idx1].X);
        }
    }

    [Fact]
    public void GPUBatchedEnvironmentState_CanAllocateAndDispose()
    {
        using var state = new GPUBatchedEnvironmentState(
            _accelerator,
            worldCount: 10,
            targetsPerWorld: 5,
            observationsPerWorld: 8,
            actionsPerWorld: 2);

        Assert.Equal(10, state.WorldCount);
        Assert.Equal(5, state.TargetsPerWorld);
        Assert.Equal(10 * 5 * 2, state.TargetPositions.Length); // x,y per target
        Assert.Equal(10 * 8, state.Observations.Length);
    }

    [Fact]
    public void GPUBatchedEnvironmentState_ResetAll_ZerosState()
    {
        using var state = new GPUBatchedEnvironmentState(
            _accelerator,
            worldCount: 10,
            targetsPerWorld: 5,
            observationsPerWorld: 8,
            actionsPerWorld: 2);

        state.ResetAll();

        var fitness = state.DownloadFitnessValues();
        var terminals = state.DownloadTerminalFlags();

        Assert.All(fitness, f => Assert.Equal(0f, f));
        Assert.All(terminals, t => Assert.Equal((byte)0, t));
    }

    [Fact]
    public void GPUBatchedEnvironmentState_AllTerminal_DetectsCorrectly()
    {
        using var state = new GPUBatchedEnvironmentState(
            _accelerator,
            worldCount: 3,
            targetsPerWorld: 2,
            observationsPerWorld: 4,
            actionsPerWorld: 2);

        state.ResetAll();

        // Initially none are terminal
        Assert.False(state.AllTerminal());

        // Mark all as terminal
        var terminals = new byte[] { 1, 1, 1 };
        state.IsTerminal.CopyFromCPU(terminals);

        Assert.True(state.AllTerminal());
    }

    public void Dispose()
    {
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
